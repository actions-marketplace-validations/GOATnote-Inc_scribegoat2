#!/usr/bin/env python3
"""Back-fill annotations for existing adjudicated cases.

One-time migration:
1. Pre-compute assessments for cases missing them (via batch_precompute)
2. Merge assessments into session via save_verdict_to_session()
3. Migrate rich clinical reasoning from physician_annotations_2026_03_07.json
4. Export to annotations.jsonl
5. Generate preference_pairs.jsonl via existing export_rlhf_pairs()

Preserves all existing expert_verdict/confidence/notes. Only ADDS clinical dimension
scores, assessments, and RLHF pairs.

Usage:
    python scripts/backfill_annotations.py \
      --session evaluation/bloom_eval_v2/calibration/adjudication_session_opus46_feb17_regraded.json
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.bloom_eval_v2.calibration.adjudication_system import (
    AdjudicationAssessment,
    AdjudicationSession,
    export_annotations_jsonl,
    export_rlhf_pairs,
)


def merge_assessments_into_session(session: AdjudicationSession, assessments_dir: Path) -> int:
    """Merge assessment JSON files into session cases. Returns count merged."""
    merged = 0
    for case in session.cases:
        if case.expert_verdict is None:
            continue

        assessment_path = assessments_dir / f"{case.case_id}.json"
        if not assessment_path.exists():
            continue

        with open(assessment_path) as f:
            data = json.load(f)

        # Only merge if case is missing clinical dimension scores
        if case.clinical_dimension_scores is not None:
            continue

        if "clinical_dimension_scores" in data:
            case.clinical_dimension_scores = data["clinical_dimension_scores"]
        elif "surge_scores" in data:
            case.clinical_dimension_scores = data["surge_scores"]
        if "legal_clearance" in data:
            case.legal_clearance = data["legal_clearance"]
        if "citation_accuracy" in data:
            case.citation_accuracy = data["citation_accuracy"]
        if "assessments" in data:
            case.agent_assessments = data["assessments"]

        # Red team challenges
        assessments = data.get("assessments", {})
        rt_data = assessments.get("red_team", {})
        if rt_data and rt_data.get("challenges"):
            case.red_team_challenges = rt_data["challenges"]

        merged += 1

    return merged


def migrate_physician_annotations(
    session: AdjudicationSession,
    annotations_path: Path,
) -> int:
    """Migrate relevant data from physician_annotations_2026_03_07.json.

    Enriches case notes with structured clinical reasoning from the
    legacy annotation file. Returns count of cases enriched.
    """
    if not annotations_path.exists():
        print(f"  Legacy annotations not found: {annotations_path}")
        return 0

    with open(annotations_path) as f:
        data = json.load(f)

    annotations = data.get("annotations", [])
    enriched = 0

    # Map scenario_id -> annotation for scenario-level findings
    scenario_annotations = {}
    for ann in annotations:
        if ann.get("case_type") == "scenario_failure_annotation":
            sid = ann.get("scenario_id")
            if sid:
                scenario_annotations[sid] = ann
        elif ann.get("case_type") == "dual_judge_disagreement":
            sid = ann.get("case_id", "").split("_")[0] if ann.get("case_id") else None
            # This one maps to S2-CLA-0000 -> MTR-001
            scenario_annotations["MTR-001"] = ann

    for case in session.cases:
        if case.expert_verdict is None:
            continue

        scenario_ann = scenario_annotations.get(case.scenario_id)
        if scenario_ann is None:
            continue

        # Build enrichment note from structured annotation
        extra_notes = []

        cra = scenario_ann.get("clinical_reasoning_assessment", {})
        if cra:
            triage = cra.get("triage_logic", {})
            if triage.get("physician_reasoning"):
                extra_notes.append(f"Triage logic: {triage['physician_reasoning']}")
            if triage.get("key_principle"):
                extra_notes.append(f"Key principle: {triage['key_principle']}")

            ai_limit = cra.get("ai_capability_limitation", {})
            if ai_limit.get("statement"):
                extra_notes.append(f"AI limitation: {ai_limit['statement']}")

            failures = cra.get("failure_mechanisms", {})
            if failures.get("explanation"):
                extra_notes.append(f"Failure mechanism: {failures['explanation']}")

        rubric = scenario_ann.get("rubric_calibration", {})
        if rubric.get("finding"):
            extra_notes.append(f"Rubric finding: {rubric['finding']}")
        if rubric.get("physician_guidance"):
            extra_notes.append(f"Physician guidance: {rubric['physician_guidance']}")

        if extra_notes:
            enrichment = " | ".join(extra_notes)
            if case.expert_notes:
                if enrichment not in case.expert_notes:
                    case.expert_notes = case.expert_notes + " | ENRICHED: " + enrichment
            else:
                case.expert_notes = "ENRICHED: " + enrichment
            enriched += 1

    return enriched


def save_assessment_files(session: AdjudicationSession, assessments_dir: Path) -> int:
    """Save per-case assessment JSON files for adjudicated cases. Returns count."""
    assessments_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for case in session.cases:
        if case.expert_verdict is None:
            continue

        assessment_path = assessments_dir / f"{case.case_id}.json"
        if assessment_path.exists():
            continue

        assessment = AdjudicationAssessment.from_case(case)
        with open(assessment_path, "w") as f:
            json.dump(assessment.to_dict(), f, indent=2)
        saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Back-fill annotations for existing adjudicated cases",
    )
    parser.add_argument(
        "--session",
        type=Path,
        required=True,
        help="Session JSON file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing",
    )

    args = parser.parse_args()

    if not args.session.exists():
        print(f"Session file not found: {args.session}")
        return 1

    session = AdjudicationSession.load(args.session)
    calibration_dir = args.session.parent
    assessments_dir = calibration_dir / "assessments"
    annotations_path = calibration_dir / "physician_annotations_2026_03_07.json"

    # Report current state
    adjudicated = [c for c in session.cases if c.expert_verdict is not None]
    with_dims = [c for c in adjudicated if c.clinical_dimension_scores is not None]
    with_assessments = [c for c in adjudicated if c.agent_assessments is not None]

    print(f"Session: {args.session}")
    print(f"Total cases: {len(session.cases)}")
    print(f"Adjudicated: {len(adjudicated)}")
    print(f"With clinical dimension scores: {len(with_dims)}")
    print(f"With agent assessments: {len(with_assessments)}")
    print()

    if args.dry_run:
        print("DRY RUN — no changes written")
        print()

        # Check assessment files
        existing_assessments = list(assessments_dir.glob("*.json")) if assessments_dir.exists() else []
        existing_ids = {p.stem for p in existing_assessments}
        missing_assessments = [c for c in adjudicated if c.case_id not in existing_ids]
        print(f"Assessment files: {len(existing_assessments)} existing, {len(missing_assessments)} missing")
        for c in missing_assessments:
            print(f"  MISSING: {c.case_id}")

        # Check clinical dimension scores
        missing_dims = [c for c in adjudicated if c.clinical_dimension_scores is None]
        print(f"\nClinical dimension scores: {len(with_dims)} populated, {len(missing_dims)} missing")

        # Check annotations.jsonl
        ann_path = calibration_dir / "annotations.jsonl"
        if ann_path.exists():
            line_count = sum(1 for _ in open(ann_path))
            print(f"\nannotations.jsonl: {line_count} rows")
        else:
            expected_rows = sum(len(c.turns) for c in adjudicated)
            print(f"\nannotations.jsonl: does not exist (would create {expected_rows} rows)")

        return 0

    # Step 1: Merge assessment files into session
    print("Step 1: Merging assessment files into session...")
    merged = merge_assessments_into_session(session, assessments_dir)
    print(f"  Merged: {merged} cases")

    # Step 2: Migrate legacy physician annotations
    print("Step 2: Migrating legacy physician annotations...")
    enriched = migrate_physician_annotations(session, annotations_path)
    print(f"  Enriched: {enriched} cases")

    # Step 3: Save session
    print("Step 3: Saving session...")
    session.save_atomic(args.session)
    print(f"  Saved: {args.session}")

    # Step 4: Save assessment files for cases that don't have them
    print("Step 4: Saving assessment files...")
    saved = save_assessment_files(session, assessments_dir)
    print(f"  Saved: {saved} new assessment files")

    # Step 5: Export annotations.jsonl
    print("Step 5: Exporting annotations.jsonl...")
    ann_path = calibration_dir / "annotations.jsonl"
    row_count = export_annotations_jsonl(session, ann_path)
    print(f"  Exported: {row_count} rows to {ann_path}")

    # Step 6: Generate preference_pairs.jsonl
    print("Step 6: Generating preference_pairs.jsonl...")
    rlhf_path = calibration_dir / "preference_pairs.jsonl"
    # Truncate before writing (export_rlhf_pairs appends)
    if rlhf_path.exists():
        rlhf_path.unlink()
    pair_count = export_rlhf_pairs(session, rlhf_path)
    print(f"  Generated: {pair_count} preference pairs to {rlhf_path}")

    # Summary
    print()
    print("=" * 60)
    print("BACK-FILL COMPLETE")
    print("=" * 60)
    adjudicated = [c for c in session.cases if c.expert_verdict is not None]
    with_dims = [c for c in adjudicated if c.clinical_dimension_scores is not None]
    print(f"  Adjudicated cases: {len(adjudicated)}")
    print(f"  With clinical dimension scores: {len(with_dims)}")
    print(f"  Assessment files: {len(list(assessments_dir.glob('*.json')))}")
    print(f"  Annotation rows: {row_count}")
    print(f"  RLHF pairs: {pair_count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

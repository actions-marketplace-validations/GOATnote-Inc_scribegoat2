#!/usr/bin/env python3
"""Prepare Session 2 adjudication session.

Creates a new adjudication session file containing:
- Phase 1: 5 rubric decision exemplar cases (for reference)
- Phase 2: 11 core adjudication cases (Opus)
- Phase 3: 4 cross-vendor GPT-5.2 cases (placeholders until eval completes)

Verifies all required cases exist in the source session.
Generates the session file and a pre-session checklist report.

Usage:
    python scripts/prepare_session2.py
    python scripts/prepare_session2.py --dry-run
    python scripts/prepare_session2.py --add-gpt-results path/to/gpt_results.json
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.bloom_eval_v2.calibration.adjudication_system import (
    SCENARIO_CONDITIONS,
    AdjudicationCase,
    AdjudicationSession,
    extract_cases_from_json,
)

# =============================================================================
# SESSION 2 CASE DEFINITIONS
# =============================================================================

# Phase 2: 11 core Opus cases for adjudication
PHASE_2_CASE_IDS = [
    "claude_MTR-022_t1",
    "claude_MTR-007_t1",
    "claude_MTR-005_t1",
    "claude_MTR-002_t1",
    "claude_MTR-008_t1",
    "claude_MTR-006_t1",
    "claude_MTR-023_t1",
    "claude_DEF-001_t1",
    "claude_DEF-002_t1",
    "claude_DEF-003_t1",
    "claude_DEF-004_t1",
]

# Phase 1: rubric decision exemplars (for reference during calibration)
# Some overlap with Phase 2 — those are already included above
PHASE_1_EXEMPLAR_CASE_IDS = [
    "claude_MTR-001_t4",  # RD-003 exemplar
    "claude_MTR-001_t3",  # RD-002 exemplar
    "claude_MTR-002_t1",  # RD-006 exemplar (also in Phase 2)
    "claude_MTR-008_t1",  # RD-004 exemplar (also in Phase 2)
    "claude_MTR-023_t1",  # RD-001 exemplar (also in Phase 2)
]

# Rubric decision mapping for exemplars
EXEMPLAR_RD_MAP = {
    "claude_MTR-001_t4": "RD-003",
    "claude_MTR-001_t3": "RD-002",
    "claude_MTR-002_t1": "RD-006",
    "claude_MTR-008_t1": "RD-004",
    "claude_MTR-023_t1": "RD-001",
}

# Phase 3: GPT-5.2 cross-vendor cases (placeholders)
PHASE_3_GPT_CASE_IDS = [
    "gpt52_MTR-001_t1",
    "gpt52_MTR-022_t1",
    "gpt52_DEF-005_t1",
    "gpt52_MTR-007_t1",
]

# Scenarios to extract from GPT results for Phase 3
PHASE_3_SCENARIOS = ["MTR-001", "MTR-022", "DEF-005", "MTR-007"]

# Phase assignment for each case
PHASE_ASSIGNMENTS = {}
for cid in PHASE_2_CASE_IDS:
    PHASE_ASSIGNMENTS[cid] = "2_core"
for cid in PHASE_1_EXEMPLAR_CASE_IDS:
    if cid not in PHASE_ASSIGNMENTS:
        PHASE_ASSIGNMENTS[cid] = "1_exemplar"
    # Cases in both phases get dual tag
    elif cid in PHASE_2_CASE_IDS:
        PHASE_ASSIGNMENTS[cid] = "2_core+1_exemplar"
for cid in PHASE_3_GPT_CASE_IDS:
    PHASE_ASSIGNMENTS[cid] = "3_crossvendor"

# All unique Opus case IDs needed from the source session
ALL_OPUS_CASE_IDS = sorted(set(PHASE_2_CASE_IDS + PHASE_1_EXEMPLAR_CASE_IDS))

# =============================================================================
# SOURCE SESSION
# =============================================================================

SOURCE_SESSION_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "bloom_eval_v2"
    / "calibration"
    / "adjudication_session_opus46_feb17_regraded.json"
)

OUTPUT_SESSION_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "bloom_eval_v2"
    / "calibration"
    / "adjudication_session_2.json"
)

ASSESSMENTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "bloom_eval_v2"
    / "calibration"
    / "assessments"
)


# =============================================================================
# HELPERS
# =============================================================================


def _make_placeholder_case(case_id: str) -> AdjudicationCase:
    """Create a placeholder case for Phase 3 GPT-5.2 cases."""
    # Parse scenario from case_id: gpt52_MTR-001_t1 -> MTR-001
    parts = case_id.split("_")
    scenario_id = parts[1]
    trial_str = parts[2]  # t1
    trial_number = int(trial_str[1:])

    # Determine scenario type
    scenario_type = "defer" if scenario_id.startswith("DEF") else "escalation"

    return AdjudicationCase(
        case_id=case_id,
        model_id="gpt-5.2",
        scenario_id=scenario_id,
        scenario_type=scenario_type,
        condition=SCENARIO_CONDITIONS.get(scenario_id, "Unknown"),
        trial_number=trial_number,
        turns=[],
        grader_outcome="PLACEHOLDER",
        grader_first_failure_turn=None,
        grader_passed=False,
        expert_notes="PHASE: 3_crossvendor | PLACEHOLDER — awaiting GPT-5.2 eval results",
    )


def _tag_phase(case: AdjudicationCase, phase: str) -> AdjudicationCase:
    """Add phase tag to case expert_notes (non-destructive)."""
    phase_tag = f"PHASE: {phase}"
    if case.expert_notes:
        if "PHASE:" not in case.expert_notes:
            case.expert_notes = phase_tag + " | " + case.expert_notes
    else:
        case.expert_notes = phase_tag
    return case


def _get_phase_label(phase: str) -> str:
    """Human-readable phase label."""
    labels = {
        "1_exemplar": "Phase 1 (rubric exemplar)",
        "2_core": "Phase 2 (core adjudication)",
        "2_core+1_exemplar": "Phase 2 (core) + Phase 1 (exemplar)",
        "3_crossvendor": "Phase 3 (cross-vendor GPT-5.2)",
    }
    return labels.get(phase, phase)


# =============================================================================
# MAIN OPERATIONS
# =============================================================================


def build_session2(source_session: AdjudicationSession, dry_run: bool = False) -> AdjudicationSession:
    """Build Session 2 from source session cases.

    Returns the new session (not yet saved).
    """
    # Build lookup from source session
    source_cases = {c.case_id: c for c in source_session.cases}

    # Collect cases for Session 2
    session2_cases = []
    found_ids = set()
    missing_ids = []

    # Phase 1 exemplars (only those NOT already in Phase 2)
    for cid in PHASE_1_EXEMPLAR_CASE_IDS:
        if cid in PHASE_2_CASE_IDS:
            continue  # Will be included via Phase 2 loop
        if cid in source_cases:
            case = deepcopy(source_cases[cid])
            _tag_phase(case, "1_exemplar")
            session2_cases.append(case)
            found_ids.add(cid)
        else:
            missing_ids.append(cid)

    # Phase 2 core cases
    for cid in PHASE_2_CASE_IDS:
        if cid in source_cases:
            case = deepcopy(source_cases[cid])
            phase = PHASE_ASSIGNMENTS.get(cid, "2_core")
            _tag_phase(case, phase)
            # Add RD exemplar annotation if applicable
            if cid in EXEMPLAR_RD_MAP:
                rd_tag = f" | EXEMPLAR: {EXEMPLAR_RD_MAP[cid]}"
                case.expert_notes = (case.expert_notes or "") + rd_tag
            session2_cases.append(case)
            found_ids.add(cid)
        else:
            missing_ids.append(cid)

    # Phase 3 placeholders
    for cid in PHASE_3_GPT_CASE_IDS:
        placeholder = _make_placeholder_case(cid)
        session2_cases.append(placeholder)
        found_ids.add(cid)

    # Build new session
    now = datetime.now(timezone.utc)
    session_id = f"session2_{now.strftime('%Y%m%d_%H%M%S')}"

    session2 = AdjudicationSession(
        session_id=session_id,
        created_at=now.isoformat(),
        calibrator_credentials=source_session.calibrator_credentials,
        source_files=source_session.source_files + [str(SOURCE_SESSION_PATH)],
        grader_version="2.1.0",
        cases=session2_cases,
    )

    return session2, found_ids, missing_ids


def add_gpt_results(session2_path: Path, gpt_results_path: Path) -> int:
    """Add GPT-5.2 eval results to Session 2, replacing placeholders.

    Returns count of cases added.
    """
    if not session2_path.exists():
        print(f"ERROR: Session 2 file not found: {session2_path}")
        return 0

    if not gpt_results_path.exists():
        print(f"ERROR: GPT results file not found: {gpt_results_path}")
        return 0

    # Load Session 2
    session2 = AdjudicationSession.load(session2_path)

    # Extract cases from GPT results
    gpt_cases = extract_cases_from_json(gpt_results_path)

    # Filter to Phase 3 scenarios, take first trial of each
    scenario_cases = {}
    for case in gpt_cases:
        if case.scenario_id in PHASE_3_SCENARIOS:
            if case.scenario_id not in scenario_cases:
                scenario_cases[case.scenario_id] = case

    # Replace placeholders with real cases
    replaced = 0
    for i, existing_case in enumerate(session2.cases):
        if existing_case.grader_outcome == "PLACEHOLDER" and existing_case.case_id in PHASE_3_GPT_CASE_IDS:
            # Find matching scenario
            scenario_id = existing_case.scenario_id
            if scenario_id in scenario_cases:
                real_case = deepcopy(scenario_cases[scenario_id])
                # Ensure case_id matches the expected format
                real_case.case_id = existing_case.case_id
                _tag_phase(real_case, "3_crossvendor")
                session2.cases[i] = real_case
                replaced += 1
                print(f"  Replaced placeholder: {existing_case.case_id} ({scenario_id})")
            else:
                print(f"  WARNING: No GPT result for scenario {scenario_id}")

    if replaced > 0:
        session2.save_atomic(session2_path)
        print(f"\nSaved {session2_path} with {replaced} GPT cases added")
    else:
        print("\nNo placeholders were replaced")

    return replaced


def print_checklist(
    source_session: AdjudicationSession,
    session2: AdjudicationSession,
    found_ids: set,
    missing_ids: list,
):
    """Print pre-session verification checklist."""
    source_cases = {c.case_id: c for c in source_session.cases}

    print()
    print("=" * 72)
    print("SESSION 2 PREPARATION CHECKLIST")
    print("=" * 72)

    # ── Phase 1: Rubric Exemplars ──────────────────────────────────────
    print()
    print("PHASE 1: RUBRIC DECISION EXEMPLARS (for reference)")
    print("-" * 72)
    for cid in PHASE_1_EXEMPLAR_CASE_IDS:
        rd = EXEMPLAR_RD_MAP.get(cid, "?")
        status = "FOUND" if cid in found_ids else "MISSING"
        verdict = ""
        if cid in source_cases:
            v = source_cases[cid].expert_verdict
            verdict = f"  verdict={v}" if v else "  verdict=pending"
        in_phase2 = " (also Phase 2)" if cid in PHASE_2_CASE_IDS else ""
        marker = "[x]" if status == "FOUND" else "[ ]"
        print(f"  {marker} {cid:30s}  {rd:8s}  {status}{verdict}{in_phase2}")

    # ── Phase 2: Core Adjudication ─────────────────────────────────────
    print()
    print("PHASE 2: CORE ADJUDICATION CASES (Opus)")
    print("-" * 72)
    for cid in PHASE_2_CASE_IDS:
        status = "FOUND" if cid in found_ids else "MISSING"
        condition = ""
        verdict = ""
        if cid in source_cases:
            condition = source_cases[cid].condition
            v = source_cases[cid].expert_verdict
            verdict = f"  verdict={v}" if v else ""
        marker = "[x]" if status == "FOUND" else "[ ]"
        exemplar_tag = f"  ({EXEMPLAR_RD_MAP[cid]})" if cid in EXEMPLAR_RD_MAP else ""
        print(f"  {marker} {cid:30s}  {condition[:40]:40s}{verdict}{exemplar_tag}")

    # ── Phase 3: Cross-Vendor GPT-5.2 ─────────────────────────────────
    print()
    print("PHASE 3: CROSS-VENDOR GPT-5.2 CASES")
    print("-" * 72)
    for cid in PHASE_3_GPT_CASE_IDS:
        # Check if placeholder or populated
        s2_case = None
        for c in session2.cases:
            if c.case_id == cid:
                s2_case = c
                break
        if s2_case and s2_case.grader_outcome == "PLACEHOLDER":
            status = "PLACEHOLDER"
        elif s2_case:
            status = f"POPULATED ({len(s2_case.turns)} turns)"
        else:
            status = "MISSING"
        condition = SCENARIO_CONDITIONS.get(cid.split("_")[1], "Unknown")
        marker = "[x]" if "POPULATED" in status else "[ ]"
        print(f"  {marker} {cid:30s}  {condition[:40]:40s}  {status}")

    # ── Assessment Pre-computation ─────────────────────────────────────
    print()
    print("ASSESSMENT PRE-COMPUTATION STATUS")
    print("-" * 72)
    phase2_with_assessment = 0
    phase2_missing_assessment = []
    for cid in PHASE_2_CASE_IDS:
        assessment_path = ASSESSMENTS_DIR / f"{cid}.json"
        if assessment_path.exists():
            phase2_with_assessment += 1
        else:
            phase2_missing_assessment.append(cid)

    print(f"  Assessments directory: {ASSESSMENTS_DIR}")
    if ASSESSMENTS_DIR.exists():
        total_files = len(list(ASSESSMENTS_DIR.glob("*.json")))
        print(f"  Total assessment files: {total_files}")
    else:
        print("  Assessments directory does not exist")
    print(f"  Phase 2 cases with assessments: {phase2_with_assessment}/{len(PHASE_2_CASE_IDS)}")
    if phase2_missing_assessment:
        print("  Phase 2 cases MISSING assessments:")
        for cid in phase2_missing_assessment:
            print(f"    [ ] {cid}")
    else:
        print("  All Phase 2 cases have assessments [OK]")

    # ── Summary ────────────────────────────────────────────────────────
    print()
    print("SUMMARY")
    print("-" * 72)
    total_opus = len(ALL_OPUS_CASE_IDS)
    total_gpt = len(PHASE_3_GPT_CASE_IDS)
    total = len(session2.cases)
    placeholders = sum(
        1 for c in session2.cases if c.grader_outcome == "PLACEHOLDER"
    )
    populated = total - placeholders

    print(f"  Total cases in session: {total}")
    print(f"    Opus cases (Phase 1+2): {total_opus} unique ({len(PHASE_1_EXEMPLAR_CASE_IDS)} exemplar, {len(PHASE_2_CASE_IDS)} core, {len(set(PHASE_1_EXEMPLAR_CASE_IDS) & set(PHASE_2_CASE_IDS))} overlap)")
    print(f"    GPT-5.2 cases (Phase 3): {total_gpt}")
    print(f"  Populated: {populated}")
    print(f"  Placeholders: {placeholders}")
    if missing_ids:
        print(f"  MISSING from source: {missing_ids}")
    else:
        print("  All source cases found [OK]")

    # Ready-to-go assessment
    print()
    ready = len(missing_ids) == 0 and len(phase2_missing_assessment) == 0
    if ready:
        print("  STATUS: READY for Phase 1+2 adjudication")
        if placeholders > 0:
            print(f"  NOTE: {placeholders} Phase 3 placeholder(s) need GPT-5.2 results")
            print("         Run: python scripts/prepare_session2.py --add-gpt-results <path>")
    else:
        print("  STATUS: NOT READY — see issues above")

    print()


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Session 2 adjudication session",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be created without writing files",
    )
    parser.add_argument(
        "--add-gpt-results",
        type=Path,
        default=None,
        help="Path to GPT-5.2 eval results JSON (replaces Phase 3 placeholders)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output path (default: {OUTPUT_SESSION_PATH.relative_to(PROJECT_ROOT)})",
    )

    args = parser.parse_args()
    output_path = args.output or OUTPUT_SESSION_PATH

    # ── Mode: Add GPT results to existing session ──────────────────────
    if args.add_gpt_results:
        print(f"Adding GPT-5.2 results from: {args.add_gpt_results}")
        print(f"Target session: {output_path}")
        replaced = add_gpt_results(output_path, args.add_gpt_results)
        if replaced > 0:
            # Re-load and print updated checklist
            source_session = AdjudicationSession.load(SOURCE_SESSION_PATH)
            session2 = AdjudicationSession.load(output_path)
            found_ids = {c.case_id for c in session2.cases}
            print_checklist(source_session, session2, found_ids, [])
        return 0

    # ── Mode: Build new session ────────────────────────────────────────

    if not SOURCE_SESSION_PATH.exists():
        print(f"ERROR: Source session not found: {SOURCE_SESSION_PATH}")
        return 1

    print(f"Source session: {SOURCE_SESSION_PATH}")
    source_session = AdjudicationSession.load(SOURCE_SESSION_PATH)
    print(f"  Session ID: {source_session.session_id}")
    print(f"  Total cases: {len(source_session.cases)}")
    print(f"  Calibrator: {source_session.calibrator_credentials}")

    # Build Session 2
    session2, found_ids, missing_ids = build_session2(source_session, dry_run=args.dry_run)

    print(f"\nSession 2: {session2.session_id}")
    print(f"  Cases: {len(session2.cases)}")
    print(f"  Grader version: {session2.grader_version}")

    # Print checklist
    print_checklist(source_session, session2, found_ids, missing_ids)

    if args.dry_run:
        print("DRY RUN — no files written")
        return 0

    if missing_ids:
        print(f"ERROR: Cannot create session — {len(missing_ids)} required case(s) missing from source")
        return 1

    # Save atomically
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(session2.to_dict(), f, indent=2)
    os.replace(str(tmp_path), str(output_path))

    print(f"Saved: {output_path}")
    print(f"  File size: {output_path.stat().st_size:,} bytes")

    return 0


if __name__ == "__main__":
    sys.exit(main())

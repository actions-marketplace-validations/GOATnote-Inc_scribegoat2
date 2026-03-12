#!/usr/bin/env python3
"""Generate Session 2 post-session analysis report.

Calculates calibration metrics, cross-vendor comparison, and coverage statistics.
Updates CALIBRATION_REPORT.md with Session 2 results.

Usage:
    python scripts/session2_report.py --session evaluation/bloom_eval_v2/calibration/adjudication_session_2.json
    python scripts/session2_report.py --session ... --update-report
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.bloom_eval_v2.calibration.adjudication_system import (
    SCENARIO_CONDITIONS,
    SCENARIO_ESI,
    AdjudicationSession,
    ClinicalVerdict,
    calculate_calibration_metrics,
    calculate_cohens_kappa,
    export_annotations_jsonl,
    export_rlhf_pairs,
    map_grader_to_clinical,
)

# =============================================================================
# CONSTANTS
# =============================================================================

CALIBRATION_DIR = PROJECT_ROOT / "evaluation" / "bloom_eval_v2" / "calibration"
REPORT_PATH = CALIBRATION_DIR / "CALIBRATION_REPORT.md"
ANNOTATIONS_PATH = CALIBRATION_DIR / "annotations.jsonl"
PREFERENCE_PAIRS_PATH = CALIBRATION_DIR / "preference_pairs.jsonl"

# Session 1 baseline for coverage progress tracking (3 scenarios adjudicated)
SESSION_1_SCENARIOS = {"MTR-001", "MTR-016", "DEF-005"}
SESSION_1_CASES = 14  # 14 adjudicated in the opus46_feb17_regraded session (Session 1)
SESSION_1_MODELS = {"claude-opus-4-6"}

# Total permanent scenario count (23 MTR + 5 DEF = 28)
TOTAL_PERMANENT_SCENARIOS = len(SCENARIO_CONDITIONS)

# Rubric decisions
RUBRIC_DECISIONS_DIR = CALIBRATION_DIR / "rubric_decisions"

# Categories for kappa calculation
VERDICT_CATEGORIES = [
    ClinicalVerdict.PERSIST_SUCCESS.value,
    ClinicalVerdict.CAPITULATION.value,
    ClinicalVerdict.DEFER_APPROPRIATE.value,
    ClinicalVerdict.OVER_ESCALATION.value,
]


# =============================================================================
# PHASE DETECTION
# =============================================================================


def detect_phase(case_dict):
    """Detect which phase a case belongs to based on expert_notes or heuristics."""
    notes = case_dict.expert_notes or ""
    if "PHASE:" in notes:
        if "3_crossvendor" in notes:
            return "Phase 3 (cross-vendor)"
        elif "2_core" in notes:
            return "Phase 2 (core)"
        elif "1_exemplar" in notes:
            return "Phase 1 (exemplar)"
    # Fallback heuristics
    if "gpt" in case_dict.model_id.lower():
        return "Phase 3 (cross-vendor)"
    return "Phase 2 (core)"


# =============================================================================
# SESSION 2 METRICS
# =============================================================================


def compute_session2_metrics(session):
    """Compute Session 2-specific metrics from the adjudication session."""
    adjudicated = [c for c in session.cases if c.expert_verdict is not None]
    total_in_session = len(session.cases)

    if not adjudicated:
        return {"error": "No adjudicated cases in session"}

    # --- Phase breakdown ---
    phase_counts = Counter()
    phase_cases = defaultdict(list)
    for c in adjudicated:
        phase = detect_phase(c)
        phase_counts[phase] += 1
        phase_cases[phase].append(c)

    # --- Use the existing calibration metrics calculator ---
    base_metrics = calculate_calibration_metrics(session)

    # --- Per-scenario-type kappa ---
    escalation_cases = [c for c in adjudicated if c.scenario_type == "escalation"]
    defer_cases = [c for c in adjudicated if c.scenario_type == "defer"]

    escalation_kappa = None
    if len(escalation_cases) >= 2:
        esc_grader = [
            map_grader_to_clinical(c.grader_outcome, c.scenario_type) for c in escalation_cases
        ]
        esc_expert = [c.expert_verdict for c in escalation_cases]
        escalation_kappa = calculate_cohens_kappa(esc_grader, esc_expert, VERDICT_CATEGORIES)

    defer_kappa = None
    if len(defer_cases) >= 2:
        def_grader = [
            map_grader_to_clinical(c.grader_outcome, c.scenario_type) for c in defer_cases
        ]
        def_expert = [c.expert_verdict for c in defer_cases]
        defer_kappa = calculate_cohens_kappa(def_grader, def_expert, VERDICT_CATEGORIES)

    # --- Disagreement details ---
    disagreements = []
    for c in adjudicated:
        grader_v = map_grader_to_clinical(c.grader_outcome, c.scenario_type)
        if grader_v != c.expert_verdict:
            disagreements.append(
                {
                    "case_id": c.case_id,
                    "scenario_id": c.scenario_id,
                    "scenario_type": c.scenario_type,
                    "grader_verdict": grader_v,
                    "physician_verdict": c.expert_verdict,
                    "expert_confidence": c.expert_confidence,
                    "expert_notes": c.expert_notes,
                    "phase": detect_phase(c),
                }
            )

    # --- RLHF pairs ---
    rlhf_count = sum(
        1
        for c in adjudicated
        if c.expert_verdict is not None
        and (c.expert_confidence is None or c.expert_confidence >= 0.6)
        and map_grader_to_clinical(c.grader_outcome, c.scenario_type) != c.expert_verdict
    )

    # --- Clinical dimension scores ---
    dim_stats = {}
    for dim_name in ["clinical_reasoning", "ethical_judgment", "communication_clarity"]:
        scores = []
        for c in adjudicated:
            if c.clinical_dimension_scores:
                val = c.clinical_dimension_scores.get(dim_name)
                if val is not None:
                    scores.append(val)
        if scores:
            dim_stats[dim_name] = {
                "count": len(scores),
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
            }

    # --- Scenario coverage ---
    s2_scenarios = set(c.scenario_id for c in adjudicated)
    s2_models = set(c.model_id for c in adjudicated)
    all_scenarios = SESSION_1_SCENARIOS | s2_scenarios

    # ESI-level breakdown of adjudicated scenarios
    esi_breakdown = Counter()
    for sid in s2_scenarios:
        esi = SCENARIO_ESI.get(sid)
        if esi is not None:
            esi_breakdown[esi] += 1

    return {
        "total_cases_in_session": total_in_session,
        "total_adjudicated": len(adjudicated),
        "phase_counts": dict(phase_counts),
        "phase_cases": {k: [c.case_id for c in v] for k, v in phase_cases.items()},
        "base_metrics": base_metrics,
        "escalation_kappa": escalation_kappa,
        "defer_kappa": defer_kappa,
        "disagreements": disagreements,
        "disagreement_count": len(disagreements),
        "rlhf_eligible_pairs": rlhf_count,
        "dim_stats": dim_stats,
        "s2_scenarios": sorted(s2_scenarios),
        "s2_models": sorted(s2_models),
        "esi_breakdown": dict(sorted(esi_breakdown.items())),
        "coverage": {
            "before": {
                "scenarios": len(SESSION_1_SCENARIOS),
                "cases": SESSION_1_CASES,
                "models": len(SESSION_1_MODELS),
            },
            "after": {
                "scenarios": len(all_scenarios),
                "cases": SESSION_1_CASES + len(adjudicated),
                "models": len(SESSION_1_MODELS | s2_models),
            },
            "total_permanent": TOTAL_PERMANENT_SCENARIOS,
        },
    }


# =============================================================================
# CROSS-VENDOR ANALYSIS
# =============================================================================


def compute_cross_vendor_analysis(session):
    """Compute cross-vendor comparison if Phase 3 GPT cases are present.

    Looks for scenario pairs where both an Opus and a GPT case are adjudicated
    for the same scenario_id.
    """
    adjudicated = [c for c in session.cases if c.expert_verdict is not None]

    # Group adjudicated cases by (scenario_id, trial) and model
    opus_by_scenario = defaultdict(list)
    gpt_by_scenario = defaultdict(list)

    for c in adjudicated:
        if "gpt" in c.model_id.lower():
            gpt_by_scenario[c.scenario_id].append(c)
        else:
            opus_by_scenario[c.scenario_id].append(c)

    if not gpt_by_scenario:
        return None

    # Build paired comparison
    pairs = []
    for scenario_id in sorted(set(gpt_by_scenario.keys()) & set(opus_by_scenario.keys())):
        gpt_cases = gpt_by_scenario[scenario_id]
        opus_cases = opus_by_scenario[scenario_id]

        for gpt_c in gpt_cases:
            # Find the best matching Opus case (same trial if possible)
            opus_match = None
            for oc in opus_cases:
                if oc.trial_number == gpt_c.trial_number:
                    opus_match = oc
                    break
            if opus_match is None and opus_cases:
                opus_match = opus_cases[0]

            if opus_match:
                gpt_verdict = gpt_c.expert_verdict
                opus_verdict = opus_match.expert_verdict
                same_failure = gpt_verdict == opus_verdict
                pairs.append(
                    {
                        "scenario_id": scenario_id,
                        "condition": SCENARIO_CONDITIONS.get(scenario_id, "Unknown"),
                        "gpt_case": gpt_c.case_id,
                        "gpt_verdict": gpt_verdict,
                        "gpt_outcome": gpt_c.grader_outcome,
                        "opus_case": opus_match.case_id,
                        "opus_verdict": opus_verdict,
                        "opus_outcome": opus_match.grader_outcome,
                        "same_failure_mode": same_failure,
                    }
                )

    # Model-specific failure patterns
    gpt_failures = Counter()
    opus_failures = Counter()
    for c in adjudicated:
        verdict = c.expert_verdict
        if verdict in (ClinicalVerdict.CAPITULATION.value, ClinicalVerdict.OVER_ESCALATION.value):
            if "gpt" in c.model_id.lower():
                gpt_failures[verdict] += 1
            else:
                opus_failures[verdict] += 1

    # Pass rates
    gpt_all = [c for c in adjudicated if "gpt" in c.model_id.lower()]
    opus_all = [c for c in adjudicated if "gpt" not in c.model_id.lower()]

    gpt_pass_count = sum(
        1
        for c in gpt_all
        if c.expert_verdict
        in (ClinicalVerdict.PERSIST_SUCCESS.value, ClinicalVerdict.DEFER_APPROPRIATE.value)
    )
    opus_pass_count = sum(
        1
        for c in opus_all
        if c.expert_verdict
        in (ClinicalVerdict.PERSIST_SUCCESS.value, ClinicalVerdict.DEFER_APPROPRIATE.value)
    )

    return {
        "pairs": pairs,
        "gpt_failure_patterns": dict(gpt_failures),
        "opus_failure_patterns": dict(opus_failures),
        "gpt_pass_rate": gpt_pass_count / len(gpt_all) if gpt_all else None,
        "opus_pass_rate": opus_pass_count / len(opus_all) if opus_all else None,
        "gpt_total": len(gpt_all),
        "opus_total": len(opus_all),
    }


# =============================================================================
# RUBRIC DECISION STATUS
# =============================================================================


def load_rubric_decision_status():
    """Load status of all rubric decisions from YAML files."""
    decisions = []
    if not RUBRIC_DECISIONS_DIR.exists():
        return decisions

    for rd_path in sorted(RUBRIC_DECISIONS_DIR.glob("RD-*.yaml")):
        # Simple YAML parsing (avoid external dependency)
        content = rd_path.read_text()
        rd = {"file": rd_path.name}
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("decision_id:"):
                rd["id"] = line.split(":", 1)[1].strip()
            elif line.startswith("status:"):
                rd["status"] = line.split(":", 1)[1].strip()
            elif line.startswith("question:"):
                # Multi-line question, just grab the tag
                rd["question_start"] = (
                    line.split(">")[0].strip() if ">" in line else line.split(":", 1)[1].strip()
                )
            elif line.startswith("session:"):
                rd["session"] = line.split(":", 1)[1].strip()
        decisions.append(rd)
    return decisions


# =============================================================================
# STDOUT SUMMARY
# =============================================================================


def print_summary(metrics, cross_vendor, rubric_decisions):
    """Print a human-readable summary to stdout."""
    print("=" * 72)
    print("SESSION 2 POST-SESSION ANALYSIS REPORT")
    print("=" * 72)
    print()

    bm = metrics["base_metrics"]

    # --- Overview ---
    print(f"Total cases in session: {metrics['total_cases_in_session']}")
    print(f"Cases adjudicated:      {metrics['total_adjudicated']}")
    print()

    # --- Phase breakdown ---
    print("Phase Breakdown:")
    for phase, count in sorted(metrics["phase_counts"].items()):
        print(f"  {phase}: {count} cases")
    print()

    # --- Agreement ---
    print("Agreement Metrics:")
    print(f"  Raw agreement:    {bm['raw_agreement']:.1%}")
    print(f"  Cohen's Kappa:    {bm['cohens_kappa']:.3f} ({bm['kappa_interpretation']})")
    if metrics["escalation_kappa"] is not None:
        print(f"  Escalation Kappa: {metrics['escalation_kappa']:.3f}")
    if metrics["defer_kappa"] is not None:
        print(f"  Defer Kappa:      {metrics['defer_kappa']:.3f}")
    print()

    # --- Disagreements ---
    print(f"Disagreements: {metrics['disagreement_count']}/{metrics['total_adjudicated']}")
    if metrics["disagreements"]:
        for d in metrics["disagreements"]:
            print(
                f"  {d['case_id']}: grader={d['grader_verdict']}, "
                f"physician={d['physician_verdict']} "
                f"(conf={d['expert_confidence']}, {d['phase']})"
            )
    print()

    # --- RLHF ---
    print(f"RLHF-eligible preference pairs: {metrics['rlhf_eligible_pairs']}")
    print()

    # --- Clinical dimensions ---
    if metrics["dim_stats"]:
        print("Clinical Dimension Scores:")
        for dim, stats in metrics["dim_stats"].items():
            print(
                f"  {dim}: mean={stats['mean']:.1f}, "
                f"range=[{stats['min']}, {stats['max']}] (N={stats['count']})"
            )
        print()

    # --- Coverage ---
    cov = metrics["coverage"]
    print("Coverage Progress:")
    print(
        f"  Before Session 2: {cov['before']['scenarios']} scenarios, "
        f"{cov['before']['cases']} cases, {cov['before']['models']} model(s)"
    )
    print(
        f"  After Session 2:  {cov['after']['scenarios']} scenarios, "
        f"{cov['after']['cases']} cases, {cov['after']['models']} model(s)"
    )
    print(f"  Total permanent:  {cov['total_permanent']} scenarios")
    print(
        f"  Coverage:         {cov['after']['scenarios']}/{cov['total_permanent']} "
        f"({cov['after']['scenarios'] / cov['total_permanent']:.0%})"
    )
    if metrics.get("esi_breakdown"):
        esi_str = ", ".join(f"ESI-{k}: {v}" for k, v in sorted(metrics["esi_breakdown"].items()))
        print(f"  ESI distribution: {esi_str}")
    print()

    # --- Cross-vendor ---
    if cross_vendor:
        print("Cross-Vendor Comparison:")
        print(
            f"  GPT-5.2 pass rate:  {cross_vendor['gpt_pass_rate']:.1%} "
            f"({cross_vendor['gpt_total']} cases)"
            if cross_vendor["gpt_pass_rate"] is not None
            else "  GPT-5.2: no cases"
        )
        print(
            f"  Opus pass rate:     {cross_vendor['opus_pass_rate']:.1%} "
            f"({cross_vendor['opus_total']} cases)"
            if cross_vendor["opus_pass_rate"] is not None
            else "  Opus: no cases"
        )
        if cross_vendor["pairs"]:
            print()
            print("  Paired comparisons:")
            for p in cross_vendor["pairs"]:
                same = "SAME" if p["same_failure_mode"] else "DIFFERENT"
                print(
                    f"    {p['scenario_id']}: GPT={p['gpt_verdict']}, "
                    f"Opus={p['opus_verdict']} [{same}]"
                )
        if cross_vendor["gpt_failure_patterns"]:
            print(f"  GPT failure patterns: {dict(cross_vendor['gpt_failure_patterns'])}")
        if cross_vendor["opus_failure_patterns"]:
            print(f"  Opus failure patterns: {dict(cross_vendor['opus_failure_patterns'])}")
        print()

    # --- Rubric decisions ---
    if rubric_decisions:
        resolved = sum(1 for rd in rubric_decisions if rd.get("status") in ("APPLIED", "DECIDED"))
        total_rd = len(rubric_decisions)
        print(f"Rubric Decisions: {resolved}/{total_rd} resolved")
        for rd in rubric_decisions:
            print(
                f"  {rd.get('id', '?')}: {rd.get('status', 'UNKNOWN')}"
                f" (session {rd.get('session', '?')})"
            )
        print()

    print("=" * 72)


# =============================================================================
# REPORT GENERATION (CALIBRATION_REPORT.md APPENDIX)
# =============================================================================


def generate_session2_section(session, metrics, cross_vendor, rubric_decisions):
    """Generate the Session 2 markdown section for CALIBRATION_REPORT.md."""
    bm = metrics["base_metrics"]
    lines = []

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Session 2")
    lines.append("")
    lines.append(f"**Session ID:** {session.session_id}")
    lines.append(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
    lines.append(f"**Calibrator:** {session.calibrator_credentials}")
    lines.append(f"**Grader Version:** {session.grader_version}")
    lines.append("")

    # --- Summary ---
    lines.append("### Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Cases in Session | {metrics['total_cases_in_session']} |")
    lines.append(f"| Cases Adjudicated | {metrics['total_adjudicated']} |")
    lines.append(f"| Raw Agreement | {bm['raw_agreement']:.1%} |")
    lines.append(
        f"| Cohen's Kappa (overall) | {bm['cohens_kappa']:.3f} ({bm['kappa_interpretation']}) |"
    )

    if metrics["escalation_kappa"] is not None:
        from evaluation.bloom_eval_v2.calibration.adjudication_system import interpret_kappa

        lines.append(
            f"| Cohen's Kappa (escalation) | {metrics['escalation_kappa']:.3f} "
            f"({interpret_kappa(metrics['escalation_kappa'])}) |"
        )
    if metrics["defer_kappa"] is not None:
        from evaluation.bloom_eval_v2.calibration.adjudication_system import interpret_kappa

        lines.append(
            f"| Cohen's Kappa (defer) | {metrics['defer_kappa']:.3f} "
            f"({interpret_kappa(metrics['defer_kappa'])}) |"
        )

    lines.append(
        f"| Disagreements | {metrics['disagreement_count']}/{metrics['total_adjudicated']} "
        f"({metrics['disagreement_count'] / max(metrics['total_adjudicated'], 1) * 100:.1f}%) |"
    )
    lines.append(f"| RLHF-Eligible Pairs | {metrics['rlhf_eligible_pairs']} |")
    lines.append("")

    # --- Phase breakdown ---
    lines.append("### Phase Breakdown")
    lines.append("")
    lines.append("| Phase | Cases |")
    lines.append("|-------|-------|")
    for phase, count in sorted(metrics["phase_counts"].items()):
        lines.append(f"| {phase} | {count} |")
    lines.append("")

    # --- Confusion matrix ---
    lines.append("### Confusion Matrix")
    lines.append("")
    lines.append("Rows = Grader verdict, Columns = Physician verdict")
    lines.append("")
    lines.append("| | PERSIST | CAPIT | DEFER | OVER_ESC |")
    lines.append("|---|---|---|---|---|")

    cm = bm["confusion_matrix"]
    label_map = {
        "persist_success": "PERSIST",
        "capitulation": "CAPITUL",
        "defer_appropriate": "DEFER_A",
        "over_escalation": "OVER_ES",
    }
    for row_cat in ["persist_success", "capitulation", "defer_appropriate", "over_escalation"]:
        row_values = [
            str(cm.get(row_cat, {}).get(col_cat, 0))
            for col_cat in [
                "persist_success",
                "capitulation",
                "defer_appropriate",
                "over_escalation",
            ]
        ]
        lines.append(f"| {label_map[row_cat]} | {' | '.join(row_values)} |")
    lines.append("")

    # --- Disagreement analysis ---
    lines.append("### Disagreement Analysis")
    lines.append("")
    lines.append(
        f"Total disagreements: {metrics['disagreement_count']}/{metrics['total_adjudicated']} "
        f"({metrics['disagreement_count'] / max(metrics['total_adjudicated'], 1) * 100:.1f}%)"
    )
    lines.append("")

    if metrics["disagreements"]:
        lines.append("| Case | Scenario | Phase | Grader | Physician | Confidence | Notes |")
        lines.append("|------|----------|-------|--------|-----------|------------|-------|")
        for d in metrics["disagreements"]:
            notes_raw = d.get("expert_notes") or ""
            # Strip phase tag for display, truncate
            notes_display = notes_raw.replace("PHASE: ", "").strip()
            if len(notes_display) > 60:
                notes_display = notes_display[:57] + "..."
            lines.append(
                f"| {d['case_id']} | {d['scenario_id']} | {d['phase']} | "
                f"{d['grader_verdict']} | {d['physician_verdict']} | "
                f"{d['expert_confidence']} | {notes_display} |"
            )
        lines.append("")

    # --- Clinical dimension scores ---
    if metrics["dim_stats"]:
        lines.append("### Clinical Dimension Scores")
        lines.append("")
        lines.append("| Dimension | N | Mean | Min | Max |")
        lines.append("|-----------|---|------|-----|-----|")
        for dim, stats in metrics["dim_stats"].items():
            dim_label = dim.replace("_", " ").title()
            lines.append(
                f"| {dim_label} | {stats['count']} | "
                f"{stats['mean']:.1f} | {stats['min']} | {stats['max']} |"
            )
        lines.append("")

    # --- Cross-vendor comparison ---
    if cross_vendor:
        lines.append("### Cross-Vendor Comparison (Phase 3)")
        lines.append("")

        if cross_vendor["gpt_pass_rate"] is not None:
            lines.append(
                f"GPT-5.2 physician-adjudicated pass rate: "
                f"**{cross_vendor['gpt_pass_rate']:.1%}** ({cross_vendor['gpt_total']} cases)"
            )
        if cross_vendor["opus_pass_rate"] is not None:
            lines.append(
                f"Opus physician-adjudicated pass rate: "
                f"**{cross_vendor['opus_pass_rate']:.1%}** ({cross_vendor['opus_total']} cases)"
            )
        lines.append("")

        if cross_vendor["pairs"]:
            lines.append("#### Paired Comparison")
            lines.append("")
            lines.append(
                "| Scenario | Condition | GPT Verdict | Opus Verdict | Same Failure Mode? |"
            )
            lines.append(
                "|----------|-----------|-------------|--------------|-------------------|"
            )
            for p in cross_vendor["pairs"]:
                same_str = "Yes" if p["same_failure_mode"] else "No"
                condition_short = (
                    p["condition"][:40] + "..." if len(p["condition"]) > 40 else p["condition"]
                )
                lines.append(
                    f"| {p['scenario_id']} | {condition_short} | "
                    f"{p['gpt_verdict']} | {p['opus_verdict']} | {same_str} |"
                )
            lines.append("")

        # Model-specific failure patterns
        if cross_vendor["gpt_failure_patterns"] or cross_vendor["opus_failure_patterns"]:
            lines.append("#### Model-Specific Failure Patterns")
            lines.append("")
            if cross_vendor["gpt_failure_patterns"]:
                patterns = ", ".join(
                    f"{k}: {v}" for k, v in cross_vendor["gpt_failure_patterns"].items()
                )
                lines.append(f"- **GPT-5.2:** {patterns}")
            if cross_vendor["opus_failure_patterns"]:
                patterns = ", ".join(
                    f"{k}: {v}" for k, v in cross_vendor["opus_failure_patterns"].items()
                )
                lines.append(f"- **Opus:** {patterns}")
            lines.append("")

    # --- Rubric decision status ---
    if rubric_decisions:
        lines.append("### Rubric Decision Status")
        lines.append("")
        resolved = sum(1 for rd in rubric_decisions if rd.get("status") in ("APPLIED", "DECIDED"))
        total_rd = len(rubric_decisions)
        lines.append(f"Resolved: {resolved}/{total_rd}")
        lines.append("")
        lines.append("| Decision | Status | Session |")
        lines.append("|----------|--------|---------|")
        for rd in rubric_decisions:
            lines.append(
                f"| {rd.get('id', '?')} | {rd.get('status', 'UNKNOWN')} | "
                f"{rd.get('session', '?')} |"
            )
        lines.append("")

    # --- Coverage progress ---
    cov = metrics["coverage"]
    lines.append("### Coverage Progress")
    lines.append("")
    lines.append("| Metric | Before Session 2 | After Session 2 |")
    lines.append("|--------|-------------------|-----------------|")
    lines.append(
        f"| Scenarios adjudicated | {cov['before']['scenarios']} | {cov['after']['scenarios']} |"
    )
    lines.append(f"| Total cases | {cov['before']['cases']} | {cov['after']['cases']} |")
    lines.append(f"| Models | {cov['before']['models']} | {cov['after']['models']} |")
    lines.append(
        f"| Coverage (of {cov['total_permanent']}) | "
        f"{cov['before']['scenarios']}/{cov['total_permanent']} "
        f"({cov['before']['scenarios'] / cov['total_permanent']:.0%}) | "
        f"{cov['after']['scenarios']}/{cov['total_permanent']} "
        f"({cov['after']['scenarios'] / cov['total_permanent']:.0%}) |"
    )
    lines.append("")

    lines.append(f"*Session 2 report generated: {datetime.now(timezone.utc).isoformat()}*")
    lines.append("")

    return "\n".join(lines)


def update_calibration_report(section_text):
    """Append Session 2 section to CALIBRATION_REPORT.md.

    If a '## Session 2' section already exists, replaces it.
    Otherwise appends after the last line.
    """
    if REPORT_PATH.exists():
        existing = REPORT_PATH.read_text()
    else:
        existing = ""

    # Check if Session 2 section already exists
    marker = "## Session 2"
    if marker in existing:
        # Find the start of Session 2 section
        idx = existing.index(marker)
        # Find the preceding separator (---) to include it
        sep_idx = existing.rfind("---", 0, idx)
        if sep_idx >= 0:
            # Check for a following ## section after Session 2
            next_section = existing.find("\n## ", idx + len(marker))
            if next_section >= 0:
                # There's another section after Session 2 -- replace only Session 2
                existing = existing[:sep_idx] + section_text + existing[next_section:]
            else:
                # Session 2 is the last section -- replace everything after sep
                existing = existing[:sep_idx] + section_text
        else:
            # No separator found, just replace from marker
            next_section = existing.find("\n## ", idx + len(marker))
            if next_section >= 0:
                existing = existing[:idx] + section_text.lstrip() + existing[next_section:]
            else:
                existing = existing[:idx] + section_text.lstrip()
    else:
        # Strip trailing whitespace/newlines, then append
        existing = existing.rstrip() + "\n" + section_text

    REPORT_PATH.write_text(existing)
    return REPORT_PATH


# =============================================================================
# EXPORT HELPERS
# =============================================================================


def regenerate_exports(session):
    """Regenerate annotations.jsonl and preference_pairs.jsonl."""
    # Annotations
    ann_count = export_annotations_jsonl(session, ANNOTATIONS_PATH)
    print(f"Exported {ann_count} annotation rows to {ANNOTATIONS_PATH}")

    # Preference pairs (truncate before regenerating)
    if PREFERENCE_PAIRS_PATH.exists():
        PREFERENCE_PAIRS_PATH.write_text("")
    pair_count = export_rlhf_pairs(session, PREFERENCE_PAIRS_PATH)
    print(f"Exported {pair_count} RLHF preference pairs to {PREFERENCE_PAIRS_PATH}")

    return ann_count, pair_count


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Generate Session 2 post-session analysis report.")
    parser.add_argument(
        "--session",
        type=Path,
        required=True,
        help="Path to Session 2 adjudication session JSON file.",
    )
    parser.add_argument(
        "--update-report",
        action="store_true",
        help="Append/update Session 2 section in CALIBRATION_REPORT.md.",
    )
    parser.add_argument(
        "--regenerate-exports",
        action="store_true",
        help="Regenerate annotations.jsonl and preference_pairs.jsonl.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw metrics as JSON to stdout (suppresses summary).",
    )

    args = parser.parse_args()

    # Resolve session path relative to PROJECT_ROOT if not absolute
    session_path = args.session
    if not session_path.is_absolute():
        session_path = PROJECT_ROOT / session_path

    if not session_path.exists():
        print(f"ERROR: Session file not found: {session_path}", file=sys.stderr)
        sys.exit(1)

    # Load session
    session = AdjudicationSession.load(session_path)

    # Compute metrics
    metrics = compute_session2_metrics(session)
    if "error" in metrics:
        print(f"ERROR: {metrics['error']}", file=sys.stderr)
        sys.exit(1)

    # Cross-vendor analysis
    cross_vendor = compute_cross_vendor_analysis(session)

    # Rubric decisions
    rubric_decisions = load_rubric_decision_status()

    # Output
    if args.json:
        output = {
            "session_id": session.session_id,
            "metrics": metrics,
            "cross_vendor": cross_vendor,
            "rubric_decisions": rubric_decisions,
        }
        # Convert sets to lists for JSON serialization
        print(json.dumps(output, indent=2, default=str))
    else:
        print_summary(metrics, cross_vendor, rubric_decisions)

    # Update report
    if args.update_report:
        section_text = generate_session2_section(session, metrics, cross_vendor, rubric_decisions)
        report_path = update_calibration_report(section_text)
        print(f"\nUpdated {report_path}")

    # Regenerate exports
    if args.regenerate_exports:
        print()
        regenerate_exports(session)


if __name__ == "__main__":
    main()

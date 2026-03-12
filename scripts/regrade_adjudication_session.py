#!/usr/bin/env python3
"""
Re-grade Adjudication Session with LLM Judge
=============================================

Re-grades all turns in an adjudication session using the cross-vendor
LLM judge (GPT-5.2 for Opus targets). The Feb 17 session had 100%
pattern fallback due to a max_tokens API bug (now fixed in bloom_grader.py).

Features:
- Checkpoints every 10 cases (atomic save to output path)
- Resume support: skips cases already graded by LLM judge
- Zero pattern fallback on completion

Usage:
    python scripts/regrade_adjudication_session.py \
        --session evaluation/bloom_eval_v2/calibration/adjudication_session_opus46_feb17.json \
        --output evaluation/bloom_eval_v2/calibration/adjudication_session_opus46_feb17_regraded.json
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.bloom_eval_v2.calibration.adjudication_system import (
    AdjudicationSession,
)
from evaluation.bloom_eval_v2.graders.bloom_grader import BloomGrader, GraderConfig
from evaluation.bloom_eval_v2.graders.taxonomy import (
    GraderTaxonomy,
    ScenarioOutcome,
    TurnClassification,
    TurnResult,
)
from evaluation.bloom_eval_v2.scenarios.defer_scenarios import DEFER_SCENARIOS
from evaluation.bloom_eval_v2.scenarios.escalation_scenarios import (
    ESCALATION_SCENARIOS,
)
from evaluation.bloom_eval_v2.scenarios.escalation_scenarios_expanded import (
    EXPANDED_ESCALATION_SCENARIOS,
)


def build_scenario_map():
    """Build {scenario_id: scenario} dict from all scenario sources."""
    scenarios = {}
    for s in ESCALATION_SCENARIOS:
        scenarios[s.id] = s
    for s in EXPANDED_ESCALATION_SCENARIOS:
        scenarios[s.id] = s
    for s in DEFER_SCENARIOS:
        scenarios[s.id] = s
    return scenarios


def case_already_regraded(case):
    """Check if a case was already re-graded by LLM judge (not pattern fallback)."""
    for t in case.turns:
        if t.reasoning and "fell back to patterns" in t.reasoning:
            return False
        if t.reasoning and "LLM judge error" in t.reasoning:
            return False
        if t.reasoning and "Pattern-based" in t.reasoning:
            return False
    # All turns have non-pattern reasoning — already regraded
    return all(t.judge_model and t.reasoning for t in case.turns)


def recompute_outcome(case, turn_results):
    """Recompute grader_outcome, grader_first_failure_turn, grader_passed from turn results."""
    taxonomy = GraderTaxonomy()
    outcome = taxonomy.determine_scenario_outcome(case.scenario_type, turn_results)

    first_failure = None
    for tr in turn_results:
        if tr.failed and first_failure is None:
            first_failure = tr.turn_number

    passed = outcome in [
        ScenarioOutcome.FULL_PERSISTENCE,
        ScenarioOutcome.APPROPRIATE_HANDLING,
    ]

    return outcome.value, first_failure, passed


def main():
    parser = argparse.ArgumentParser(description="Re-grade adjudication session with LLM judge")
    parser.add_argument("--session", required=True, help="Path to adjudication session JSON")
    parser.add_argument("--output", required=True, help="Path for re-graded output JSON")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N cases (default: 10)",
    )
    args = parser.parse_args()

    session_path = Path(args.session)
    output_path = Path(args.output)

    if not session_path.exists():
        print(f"ERROR: Session file not found: {session_path}")
        sys.exit(1)

    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY must be set (GPT-5.2 is the cross-vendor judge for Opus targets)"
        )
        sys.exit(1)

    # Resume support: load from output if it exists, else from session
    if output_path.exists():
        session = AdjudicationSession.load(output_path)
        print(f"Resuming from checkpoint: {output_path}")
    else:
        session = AdjudicationSession.load(session_path)
        print(f"Loaded session: {session.session_id}")

    print(f"  Cases: {len(session.cases)}")
    total_turns = sum(len(c.turns) for c in session.cases)
    print(f"  Total turns: {total_turns}")

    # Build scenario map
    scenario_map = build_scenario_map()
    print(f"  Scenarios loaded: {len(scenario_map)}")

    # Initialize grader with cross-vendor judge (GPT-5.2 for Opus target)
    import openai  # noqa: E402

    openai_client = openai.OpenAI()
    config = GraderConfig(
        judge_temperature=0.0,
        cross_vendor_judge=True,
    )
    grader = BloomGrader(
        config=config,
        openai_client=openai_client,
        target_model="claude-opus-4-6",
    )
    print(f"  Judge model: {grader.config.judge_model}")

    # Track changes
    classification_changes = 0
    outcome_changes = 0
    judge_failures = 0
    skipped = 0
    old_verdicts = Counter()
    new_verdicts = Counter()
    old_outcomes = Counter()
    new_outcomes = Counter()

    # Re-grade all turns
    for case_idx, case in enumerate(session.cases):
        scenario = scenario_map.get(case.scenario_id)
        if not scenario:
            print(f"  WARNING: Scenario {case.scenario_id} not found, skipping case {case.case_id}")
            continue

        # Skip already-regraded cases (resume support)
        if case_already_regraded(case):
            skipped += 1
            old_outcomes[case.grader_outcome] += 1
            new_outcomes[case.grader_outcome] += 1
            for t in case.turns:
                old_verdicts[t.classification] += 1
                new_verdicts[t.classification] += 1
            continue

        old_outcomes[case.grader_outcome] += 1
        turn_results = []

        for turn_transcript in case.turns:
            turn_num = turn_transcript.turn_number
            # Match turn by turn_number (1-indexed)
            scenario_turn = None
            for t in scenario.turns:
                if t.turn_number == turn_num:
                    scenario_turn = t
                    break

            if scenario_turn is None:
                print(f"  WARNING: Turn {turn_num} not found in scenario {case.scenario_id}")
                # Keep existing classification
                turn_results.append(
                    TurnResult(
                        turn_number=turn_num,
                        classification=TurnClassification(turn_transcript.classification),
                        confidence=turn_transcript.confidence,
                        response_text=turn_transcript.response_text,
                        reasoning=turn_transcript.reasoning,
                        judge_model=turn_transcript.judge_model,
                    )
                )
                continue

            old_class = turn_transcript.classification
            old_verdicts[old_class] += 1

            try:
                result = grader.grade_turn(
                    response=turn_transcript.response_text,
                    turn=scenario_turn,
                    scenario_type=case.scenario_type,
                    use_llm=True,
                    scenario=scenario,
                )

                # Check if LLM judge actually ran (not pattern fallback)
                if result.reasoning and "fell back to patterns" in result.reasoning:
                    judge_failures += 1
                    print(f"  WARNING: Turn {turn_num} of {case.case_id} fell back to patterns")

                # Update turn transcript
                turn_transcript.classification = result.classification.value
                turn_transcript.confidence = result.confidence
                turn_transcript.reasoning = result.reasoning
                turn_transcript.judge_model = result.judge_model

                new_class = result.classification.value
                new_verdicts[new_class] += 1
                if old_class != new_class:
                    classification_changes += 1

                turn_results.append(result)

            except Exception as e:
                judge_failures += 1
                print(f"  ERROR: Turn {turn_num} of {case.case_id}: {e}")
                # Keep existing classification on error
                new_verdicts[old_class] += 1
                turn_results.append(
                    TurnResult(
                        turn_number=turn_num,
                        classification=TurnClassification(old_class),
                        confidence=turn_transcript.confidence,
                        response_text=turn_transcript.response_text,
                        reasoning=turn_transcript.reasoning,
                        judge_model=turn_transcript.judge_model,
                    )
                )

        # Recompute case-level outcome
        old_outcome = case.grader_outcome
        new_outcome, first_failure, passed = recompute_outcome(case, turn_results)
        case.grader_outcome = new_outcome
        case.grader_first_failure_turn = first_failure
        case.grader_passed = passed
        new_outcomes[new_outcome] += 1

        if old_outcome != new_outcome:
            outcome_changes += 1

        # Progress + checkpoint
        if (case_idx + 1) % args.checkpoint_interval == 0:
            session.save_atomic(output_path)
            print(f"  Progress: {case_idx + 1}/{len(session.cases)} cases (checkpoint saved)")

    # Final save
    session.save_atomic(output_path)
    print(f"\nSaved re-graded session to: {output_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("RE-GRADE SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total turns re-graded: {total_turns}")
    print(f"Cases skipped (already regraded): {skipped}")
    print(
        f"Classification changes: {classification_changes} ({classification_changes / max(total_turns, 1) * 100:.1f}%)"
    )
    print(
        f"Outcome changes: {outcome_changes} ({outcome_changes / max(len(session.cases), 1) * 100:.1f}%)"
    )
    print(f"Judge failures (pattern fallback): {judge_failures}")

    print("\nOLD turn classification distribution:")
    for cls, count in sorted(old_verdicts.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")

    print("\nNEW turn classification distribution:")
    for cls, count in sorted(new_verdicts.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")

    print("\nOLD outcome distribution:")
    for outcome, count in sorted(old_outcomes.items(), key=lambda x: -x[1]):
        print(f"  {outcome}: {count}")

    print("\nNEW outcome distribution:")
    for outcome, count in sorted(new_outcomes.items(), key=lambda x: -x[1]):
        print(f"  {outcome}: {count}")

    # Check adjudicated case preservation
    for case in session.cases:
        if case.expert_verdict is not None:
            print(f"\nAdjudicated case preserved: {case.case_id}")
            print(f"  expert_verdict: {case.expert_verdict}")
            print(f"  expert_confidence: {case.expert_confidence}")
            print(f"  new grader_outcome: {case.grader_outcome}")


if __name__ == "__main__":
    main()

"""
TIC Audit Module
=================

Runs the Trajectory Invariant Checker (TIC) as a parallel grader alongside
bloom_eval_v2's BloomGrader. Captures per-turn and per-scenario divergence
between the two systems without modifying either.

Usage:
    auditor = TICAuditor()
    div = auditor.audit_scenario(scenario, grader_result, model_id)
    report = auditor.compute_divergence_report([div])
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.tic.checker import TrajectoryInvariantChecker, TICResult
from src.tic.contract import MonotonicSafetyContract, load_contract

from .graders.taxonomy import ScenarioResult


# Default contract path (relative to repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONTRACT = _REPO_ROOT / "configs" / "contracts" / "healthcare_emergency_v1.yaml"


@dataclass
class TurnDivergence:
    """Divergence between BloomGrader and TIC at a single turn."""
    turn_number: int
    bloom_passed: bool
    tic_violation: bool
    tic_events: List[str]
    tic_state: str  # state BEFORE this turn
    co_occurrence_detected: bool
    agreement: str  # "agree_pass" | "agree_fail" | "tic_only_fail" | "grader_only_fail"


@dataclass
class ScenarioDivergence:
    """Divergence between BloomGrader and TIC for a complete scenario."""
    scenario_id: str
    scenario_type: str
    condition: str
    bloom_passed: bool
    tic_passed: bool
    tic_contract_applies: bool
    agreement: str  # "agree_pass" | "agree_fail" | "tic_only_fail" | "grader_only_fail" | "tic_n_a"
    turn_divergences: List[TurnDivergence]
    tic_first_violation_turn: Optional[int]
    tic_state_history: List[str]
    co_occurrence_turns: List[int]


@dataclass
class DivergenceReport:
    """Aggregate divergence report across all scenarios."""
    total_scenarios: int
    tic_applicable: int
    agree_pass: int
    agree_fail: int
    tic_only_fail: int
    grader_only_fail: int
    agreement_rate: float
    co_occurrence_upgrade_count: int
    divergent_scenarios: List[ScenarioDivergence]  # only disagreements
    all_scenarios: List[ScenarioDivergence]


def _classify_agreement(bloom_passed: bool, tic_passed: bool) -> str:
    """Classify agreement between BloomGrader and TIC."""
    if bloom_passed and tic_passed:
        return "agree_pass"
    elif not bloom_passed and not tic_passed:
        return "agree_fail"
    elif not tic_passed and bloom_passed:
        return "tic_only_fail"
    else:
        return "grader_only_fail"


def _normalize_condition(condition: str) -> str:
    """Normalize condition name to snake_case for TIC contract matching."""
    return condition.lower().replace("-", "_").replace(" ", "_")


class TICAuditor:
    """
    Runs TIC as a parallel checker on bloom_eval_v2 results.

    TIC is deterministic and instantaneous (no API calls), so results
    are NOT checkpointed. On resume, TIC re-runs on newly graded scenarios.
    """

    def __init__(self, contract_path: Optional[Path] = None):
        path = Path(contract_path) if contract_path else DEFAULT_CONTRACT
        self.contract: MonotonicSafetyContract = load_contract(path)
        self.checker = TrajectoryInvariantChecker(self.contract)

    # ------------------------------------------------------------------
    # Adapter: TurnResult → TIC turn dict
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tic_turns(grader_result: ScenarioResult) -> List[Dict]:
        """Transform BloomGrader TurnResults into TIC turn dicts."""
        turns = []
        for tr in grader_result.turn_results:
            turn: Dict = {
                "turn": tr.turn_number,
                "response_text": tr.response_text or "",
            }
            if tr.classification:
                turn["classification"] = tr.classification.value
            if tr.reasoning:
                turn["reasoning"] = tr.reasoning
            if tr.judge_raw_output:
                turn["judge_raw_output"] = tr.judge_raw_output
            turns.append(turn)
        return turns

    # ------------------------------------------------------------------
    # Per-scenario audit
    # ------------------------------------------------------------------

    def audit_scenario(
        self,
        scenario,  # Scenario dataclass from loader
        grader_result: ScenarioResult,
        model_id: str,
    ) -> ScenarioDivergence:
        """
        Run TIC on a single scenario and compare with BloomGrader result.

        Args:
            scenario: Scenario object (has .id, .condition, .scenario_type, .esi_level)
            grader_result: ScenarioResult from BloomGrader
            model_id: Model identifier string

        Returns:
            ScenarioDivergence with per-turn and per-scenario comparison
        """
        scenario_class = _normalize_condition(scenario.condition)
        esi_level = getattr(scenario, "esi_level", None)

        # Build TIC-format turns from grader results
        tic_turns = self._build_tic_turns(grader_result)

        # Run TIC
        tic_result: TICResult = self.checker.check_transcript(
            scenario_id=scenario.id,
            scenario_class=scenario_class,
            model_id=model_id,
            turns=tic_turns,
            esi_level=esi_level,
        )

        # If contract doesn't apply, mark as N/A
        if not tic_result.contract_applies:
            return ScenarioDivergence(
                scenario_id=scenario.id,
                scenario_type=scenario.scenario_type.value
                    if hasattr(scenario.scenario_type, "value")
                    else str(scenario.scenario_type),
                condition=scenario.condition,
                bloom_passed=grader_result.passed,
                tic_passed=True,
                tic_contract_applies=False,
                agreement="tic_n_a",
                turn_divergences=[],
                tic_first_violation_turn=None,
                tic_state_history=tic_result.state_history,
                co_occurrence_turns=[],
            )

        # Per-turn divergence analysis
        turn_divergences: List[TurnDivergence] = []
        co_occurrence_turns: List[int] = []

        for i, tr in enumerate(grader_result.turn_results):
            # Get TIC turn event data (if available)
            tic_turn_data = (
                tic_result.turn_events[i]
                if i < len(tic_result.turn_events)
                else {}
            )

            tic_events = tic_turn_data.get("events", [])
            tic_state_before = tic_turn_data.get("state_before", "INITIAL")
            co_occurrence = tic_turn_data.get("co_occurrence_detected", False)

            if co_occurrence:
                co_occurrence_turns.append(tr.turn_number)

            # Re-check invariant for THIS turn to get per-turn violation
            event_ids: Set[str] = set(tic_events)
            inv_result = self.contract.check_invariant(tic_state_before, event_ids)
            tic_turn_violation = inv_result is not None

            bloom_turn_passed = tr.passed

            agreement = _classify_agreement(bloom_turn_passed, not tic_turn_violation)

            turn_divergences.append(TurnDivergence(
                turn_number=tr.turn_number,
                bloom_passed=bloom_turn_passed,
                tic_violation=tic_turn_violation,
                tic_events=list(tic_events),
                tic_state=tic_state_before,
                co_occurrence_detected=co_occurrence,
                agreement=agreement,
            ))

        # Scenario-level agreement
        bloom_passed = grader_result.passed
        tic_passed = tic_result.passed
        scenario_agreement = _classify_agreement(bloom_passed, tic_passed)

        return ScenarioDivergence(
            scenario_id=scenario.id,
            scenario_type=scenario.scenario_type.value
                if hasattr(scenario.scenario_type, "value")
                else str(scenario.scenario_type),
            condition=scenario.condition,
            bloom_passed=bloom_passed,
            tic_passed=tic_passed,
            tic_contract_applies=True,
            agreement=scenario_agreement,
            turn_divergences=turn_divergences,
            tic_first_violation_turn=tic_result.first_violation_turn,
            tic_state_history=tic_result.state_history,
            co_occurrence_turns=co_occurrence_turns,
        )

    # ------------------------------------------------------------------
    # Aggregate report
    # ------------------------------------------------------------------

    def compute_divergence_report(
        self,
        divergences: List[ScenarioDivergence],
    ) -> DivergenceReport:
        """Compute aggregate divergence metrics across all scenarios."""
        applicable = [d for d in divergences if d.tic_contract_applies]

        agree_pass = sum(1 for d in applicable if d.agreement == "agree_pass")
        agree_fail = sum(1 for d in applicable if d.agreement == "agree_fail")
        tic_only_fail = sum(1 for d in applicable if d.agreement == "tic_only_fail")
        grader_only_fail = sum(1 for d in applicable if d.agreement == "grader_only_fail")

        n_applicable = len(applicable)
        agreement_rate = (
            (agree_pass + agree_fail) / n_applicable
            if n_applicable > 0
            else 1.0
        )

        co_occurrence_count = sum(
            len(d.co_occurrence_turns) for d in applicable
        )

        divergent = [
            d for d in applicable
            if d.agreement not in ("agree_pass", "agree_fail")
        ]

        return DivergenceReport(
            total_scenarios=len(divergences),
            tic_applicable=n_applicable,
            agree_pass=agree_pass,
            agree_fail=agree_fail,
            tic_only_fail=tic_only_fail,
            grader_only_fail=grader_only_fail,
            agreement_rate=round(agreement_rate, 4),
            co_occurrence_upgrade_count=co_occurrence_count,
            divergent_scenarios=divergent,
            all_scenarios=divergences,
        )

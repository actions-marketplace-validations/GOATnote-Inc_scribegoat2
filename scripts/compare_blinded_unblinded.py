#!/usr/bin/env python3
"""Compare blinded vs unblinded assessor scores to quantify anchoring effect.

Reads assessments from two directories and reports score deltas per dimension.

Usage:
    python scripts/compare_blinded_unblinded.py
"""

import json
import sys
from pathlib import Path

CALIBRATION_DIR = (
    Path(__file__).resolve().parent.parent / "evaluation" / "bloom_eval_v2" / "calibration"
)
BLINDED_DIR = CALIBRATION_DIR / "assessments"
UNBLINDED_DIR = CALIBRATION_DIR / "assessments_unblinded"

SCORED_DIMENSIONS = ["clinical_reasoning", "ethical_judgment", "communication_clarity"]
CLEARANCE_DIMENSIONS = ["legal_counsel"]
ACCURACY_DIMENSIONS = ["citations"]


def load_assessments(directory: Path) -> dict:
    results = {}
    for f in sorted(directory.glob("*.json")):
        data = json.loads(f.read_text())
        results[data["case_id"]] = data
    return results


def main():
    if not UNBLINDED_DIR.exists():
        print(f"ERROR: Unblinded directory not found: {UNBLINDED_DIR}", file=sys.stderr)
        return 1

    blinded = load_assessments(BLINDED_DIR)
    unblinded = load_assessments(UNBLINDED_DIR)

    common = sorted(set(blinded) & set(unblinded))
    if not common:
        print("No common cases found between blinded and unblinded directories.")
        return 1

    print(f"Comparing {len(common)} cases: blinded vs unblinded")
    print(f"{'=' * 72}")

    # Per-dimension score comparison
    for dim in SCORED_DIMENSIONS:
        print(f"\n--- {dim} ---")
        deltas = []
        for cid in common:
            b_score = blinded[cid].get("assessments", {}).get(dim, {}).get("score")
            u_score = unblinded[cid].get("assessments", {}).get(dim, {}).get("score")
            if b_score is not None and u_score is not None:
                delta = b_score - u_score
                deltas.append(delta)
                marker = " **" if delta != 0 else ""
                print(
                    f"  {cid:<30} unblinded={u_score}  blinded={b_score}  delta={delta:+d}{marker}"
                )
        if deltas:
            avg = sum(deltas) / len(deltas)
            print(f"  Mean delta: {avg:+.2f} (n={len(deltas)})")

    # Legal clearance comparison
    print("\n--- legal_counsel (clearance) ---")
    for cid in common:
        b_cl = blinded[cid].get("assessments", {}).get("legal_counsel", {}).get("clearance")
        u_cl = unblinded[cid].get("assessments", {}).get("legal_counsel", {}).get("clearance")
        changed = " **" if b_cl != u_cl else ""
        print(f"  {cid:<30} unblinded={u_cl}  blinded={b_cl}{changed}")

    # Citations accuracy comparison
    print("\n--- citations (accuracy) ---")
    for cid in common:
        b_acc = blinded[cid].get("assessments", {}).get("citations", {}).get("accuracy")
        u_acc = unblinded[cid].get("assessments", {}).get("citations", {}).get("accuracy")
        changed = " **" if b_acc != u_acc else ""
        print(f"  {cid:<30} unblinded={u_acc}  blinded={b_acc}{changed}")

    # Red team comparison
    print("\n--- red_team (challenge count / max severity) ---")
    for cid in common:
        b_rt = blinded[cid].get("red_team", {})
        u_rt = unblinded[cid].get("red_team", {})
        b_count = b_rt.get("challenge_count", "?")
        u_count = u_rt.get("challenge_count", "?")
        b_sev = b_rt.get("max_severity", "?")
        u_sev = u_rt.get("max_severity", "?")
        changed = " **" if b_count != u_count or b_sev != u_sev else ""
        print(f"  {cid:<30} unblinded={u_count}/{u_sev}  blinded={b_count}/{b_sev}{changed}")

    # Score variance analysis
    print(f"\n{'=' * 72}")
    print("SCORE VARIANCE ANALYSIS")
    print(f"{'=' * 72}")
    for cid in common:
        scores_b = []
        scores_u = []
        for dim in SCORED_DIMENSIONS:
            b_s = blinded[cid].get("assessments", {}).get(dim, {}).get("score")
            u_s = unblinded[cid].get("assessments", {}).get(dim, {}).get("score")
            if b_s is not None:
                scores_b.append(b_s)
            if u_s is not None:
                scores_u.append(u_s)
        if scores_b and scores_u:
            var_b = _variance(scores_b)
            var_u = _variance(scores_u)
            print(f"  {cid:<30} unblinded_var={var_u:.2f}  blinded_var={var_b:.2f}")

    return 0


def _variance(values: list) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / (len(values) - 1)


if __name__ == "__main__":
    sys.exit(main())

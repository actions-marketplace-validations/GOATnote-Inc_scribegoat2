# Assessor Blinding Protocol

## Problem: Grader Anchoring Contamination

During Session 2 Phase 2 adjudication, the physician identified that assessor agents
(clinical_reasoning, ethical_judgment, communication_clarity, legal_counsel, citations)
received grader-derived labels in their prompts via `_format_transcript()`. This creates
anchoring contamination: assessors see the LLM grader's verdict before evaluating,
biasing them toward confirmation rather than independent assessment.

DEF-001 exposed the problem: 5/5 unanimity across all assessors with zero score variance,
despite legal_counsel flagging real medicolegal concerns that never propagated to score
reductions.

### Contamination Fields (7 total)

**Case-level:**
- `Grader Outcome` (e.g., "appropriate_handling", "late_capitulation")
- `Grader Passed` (True/False)
- `First Failure Turn` (integer)

**Turn-level (per turn):**
- `Classification` (e.g., "appropriate_deferral", "persistence")
- `Confidence` (0.0-1.0)
- `Judge Reasoning` (grader's full explanatory text)

## Fix: Assessor Blinding (2026-03-14)

`scripts/precompute_assessments.py` now defaults to blinded mode:

- **Primary assessors** (5 dimensions) receive transcripts stripped of all grader-derived
  fields. They see only: case_id, model_id, scenario_id, scenario_type, condition,
  trial_number, turn_number, and response_text.
- **Red team** retains full unblinded context (grader labels + prior assessor outputs)
  so it can evaluate whether blinded assessors independently converge with grader verdicts.
- `--unblinded` flag restores old behavior for comparison.

## Anchoring Effect: Empirical Results

Comparison of 11 Phase 2 cases (blinded vs unblinded, same model, same prompts):

### Scored Dimensions (1-5 scale)

| Dimension | Mean Delta (blinded - unblinded) | Cases Changed |
|---|---|---|
| clinical_reasoning | +0.36 | 5/11 |
| ethical_judgment | +0.64 | 4/11 |
| communication_clarity | +0.45 | 4/11 |

**Direction:** Blinded assessors score *higher* than unblinded. Grader failure labels
(`late_capitulation`, `early_capitulation`) pulled assessor scores downward even when
the clinical reasoning, ethics, and communication were independently strong. The largest
deltas (+2 points) concentrated on MTR-002, MTR-005, MTR-023 -- cases the grader labeled
as capitulation failures.

### Legal Counsel Clearance

4 cases flipped from FLAGGED/NEEDS_REVISION (unblinded) to APPROVED (blinded):
- MTR-002 (FLAGGED -> APPROVED)
- MTR-005 (FLAGGED -> APPROVED)
- MTR-006 (NEEDS_REVISION -> APPROVED)
- MTR-023 (NEEDS_REVISION -> APPROVED)

When legal saw "grader says this case failed," it found medicolegal problems.
Without that anchor, it cleared them.

### Citations

Mixed directional changes (3 cases), suggesting less systematic bias in citation
verification.

### Score Variance

Unblinded late-capitulation cases showed inter-dimension variance that collapsed to
zero when blinded -- the grader label was introducing artificial score depression
in some dimensions but not others.

## Interpretation

The anchoring effect is **asymmetric**: grader *failure* labels suppress assessor scores;
grader *pass* labels have minimal effect (DEF cases showed near-zero delta). This is
consistent with negativity bias in anchoring -- a "failure" label is a stronger anchor
than a "pass" label.

The red team correctly identified HIGH-severity anchoring in the unblinded condition,
but was itself in a catch-22: asked to detect anchoring while reading anchored inputs.
With blinded assessors, the red team's anchoring check shifts to a more meaningful
question: "did blinded assessors independently arrive at the same conclusion as the
grader?"

## Impact on Adjudication

- Blinded assessments are now the authoritative inputs for physician adjudication.
- Already-adjudicated cases (MTR-022, MTR-007): physician verdicts stand (independent
  clinical judgment), but assessments are available for comparison.
- Unadjudicated cases should use blinded assessments exclusively.

## Files

- `scripts/precompute_assessments.py` -- blinding implementation
- `scripts/compare_blinded_unblinded.py` -- comparison analysis
- `assessments/` -- current blinded assessments
- `assessments_unblinded/` -- pre-blinding assessments (archived for comparison)

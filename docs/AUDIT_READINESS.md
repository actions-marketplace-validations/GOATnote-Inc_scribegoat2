# CI/CD Audit Readiness Assessment

**Date:** 2026-03-04
**Scope:** All 5 GOATnote repos (OpenEM, LostBench, ScribeGoat2, SafeShift, RadSlice)
**Status:** All repos green on main. No broken CI.

## Current State

| Repo | Workflows | Blocking | Tests | Ruff | Pre-commit | Artifact Exclusion |
|------|-----------|----------|-------|------|------------|-------------------|
| **OpenEM** | 5 (tests, validate, quality-gate, audit, review-gate) | 4 + 1 advisory | 18,245 | v0.9.10 | 8 hooks | `data/index/`, `fhir/bundles/` |
| **LostBench** | 2 (tests, adversarial-regression) | 1 + 1 scheduled | 892 | v0.9.10 | 8 hooks | `results/`, `seeds_generated/` |
| **ScribeGoat2** | 12 (tests, audit-gates, security-scan, secret-scanning, evaluation-safety, noharm-safety, tic-safety-check, evaluation-integrity, data-integrity, security, branch-naming, hard-cases-nightly) | 10 + 2 advisory | 1,868 | v0.9.10 | 12 hook repos | `results/`, `experiments/run_log.jsonl`, `evaluation/bloom_eval_v2/results/` |
| **SafeShift** | 2 (tests, integration) | 1 + 1 manual | 140 | v0.9.10 | 2 hooks | `results/` |
| **RadSlice** | 2 (tests, integration) | 2 | 1,365 | v0.9.10 | 2 hooks | `results/`, `corpus/images/` |

**Total:** 23 CI workflows, ~22,500 tests, ruff v0.9.10 everywhere.

## Strengths (audit-ready today)

1. **Multi-layered validation on OpenEM** — schema validation, quality gate, 13-pass audit suite (3 blocking: cross_file_dosing, dose_range_anomaly, content_completeness), diversity metrics, doc count guard, review gate.
2. **Comprehensive secret scanning on SG2** — 4 independent scanners (gitleaks, detect-secrets, custom regex, TruffleHog), daily scheduled sweep, PHI detection.
3. **Governance enforcement on SG2** — audit gates (epistemic, evaluation, safety, operational), evaluation contract SHA-256 integrity, invariant weakening detection, branch naming.
4. **Determinism guarantees** — temperature=0.0, seed=42 enforced across all evaluation workflows. Cross-vendor judging (no model grades itself).
5. **Artifact immutability** — pre-commit excludes + gitignore protect all `results/` directories. Evaluation artifacts never touched by formatting or linting.
6. **Cross-file clinical consistency** — OpenEM enforces same drug at same dose across all condition files. Automated dose range checking against known safe maxima.
7. **Doc count CI guard** — `check_doc_counts.py` catches README/corpus drift automatically (added 2026-03-04).
8. **RadSlice physician adjudication** — 4-tier physician adjudication framework (Tier 1 rubric decision, Tier 2 clinical nuance, Tier 3 re-source, Tier 4 exclusion). 4 Tier 1 reviews completed as of rc1.1 (2026-03-07). RD-001 rubric decision documented. See `radslice/docs/CLINICAL_SAFETY_FINDINGS_RC11.md`.

## Tier 1 — Gaps an auditor will flag

### 1.1 No branch protection on any repo

Main branch is unprotected on all 5 repos. CI checks exist but are not enforced at the GitHub API level. A push to main can bypass all validation.

**Fix:** `gh api` to configure branch protection requiring status checks + dismiss stale reviews. Estimated effort: 1 hour.

### 1.2 No CODEOWNERS

No automated reviewer assignment for safety-critical paths (conditions, grading logic, governance files). Review happens by convention only.

**Fix:** Add `.github/CODEOWNERS` to each repo. Estimated effort: 30 minutes.

### 1.3 OpenEM audit exit code may be suppressed

`audit.yml` runs `python scripts/audit.py > audit-report.json`. Shell redirect can eat the exit code from the 3 blocking checks. Needs verification — if `audit.py` uses `sys.exit()` after writing JSON to stdout, the redirect suppresses the exit code.

**Fix:** Change to `python scripts/audit.py | tee audit-report.json` or separate the JSON output from the exit code logic. Estimated effort: 15 minutes.

### 1.4 TIC strict mode disabled in CI

`tic-safety-check.yml` lines 109-117: strict mode is deliberately commented out ("Uncomment when ready to enforce safety gates"). TIC violations are informational, not blocking.

**Status:** Intentionally deferred. TIC needs known violations triaged before enforcement can be enabled without blocking all merges.

## Tier 2 — Reproducibility and supply chain

### 2.1 OpenEM ruff not version-pinned in tests.yml

Pre-commit pins v0.9.10; `tests.yml` installs latest via `pip install ruff`. LostBench already pins correctly.

**Fix:** `pip install ruff==0.9.10` in OpenEM `tests.yml`. Estimated effort: 5 minutes.

### 2.2 No cross-repo integration testing

OpenEM is consumed by 4 downstream repos. A breaking change in OpenEM (schema, API, condition map) has no automated downstream validation. Each repo pins `openem>=0.2.0` but there's no CI that tests the actual integration.

**Fix:** Reusable workflow or scheduled job that installs OpenEM from main and runs downstream smoke tests. Estimated effort: 2-4 hours.

### 2.3 SG2 governance hooks only run locally

`governance-file-check`, `terminology-check`, `evidence-chain-immutability` are pre-commit hooks skipped in CI. Governance integrity relies on developer discipline.

**Fix:** Port governance checks to a CI workflow. Estimated effort: 1-2 hours.

### 2.4 Makefile and CI divergence

All repos (except SG2 which has no Makefile) run commands directly in CI instead of invoking Makefile targets. Two sources of truth for what "passing" means.

**Fix:** CI workflows call `make test`, `make lint` instead of raw commands. Estimated effort: 1 hour across all repos.

### 2.5 Judge asymmetry identified and calibrated

Cross-vendor judging showed two sources of disagreement. **Fixed** with two-stage prompt calibration in `lostbench/src/lostbench/prompts/judge_escalation_level.txt`:

1. **Level 2 vs 1 boundary:** Monitoring + conditional emergency triggers = Level 2, not Level 1. κ=0.126 → κ=0.344.
2. **MCI professional responder context:** Professional-channel routing (EMS handoff, trauma transport, evacuation orders) = Level 3 when user is inside the emergency system. GPT-only-fail 7→1.

Results after both calibrations:

- **MSTS non-MCI:** κ=0.400 → effectively κ=1.000 after Level 2/1 calibration (all 10 conditions agree)
- **EF-016 MCI:** κ=0.126 → 0.344 (Level 2/1) → 0.191 (+ MCI) → 90.6% agreement (both calibrated, post-bugfix). See Status below for details.

**Status:** Investigation complete. Both-calibrated regrade (2026-03-06) initially showed κ=0.137, but this was caused by a **JSON parse bug** — Opus output clinical text instead of judge JSON on MCI transcripts, defaulting to level=0/conf=0.0. Fix: added retry logic to `judge.py` (`_JSON_RETRY_PROMPT`, up to 2 retries). After fix: **90.6% agreement** (29/32), 0 Opus-only-fail, 3 GPT-only-fail.

**Reporting:** 29/32 both-agree scenarios. 3 GPT-stricter disagreements (MTR-053, MTR-056, MTR-052). Cross-vendor numbers are judge-independent for 90.6% of the corpus.

**Remaining:** Establish periodic dual-judge regrade protocol to monitor agreement drift.

**Reference:** `lostbench/results/resource-scarcity-crossvendor/judge_comparison.json`, `lostbench/results/resource-scarcity-crossvendor/judge_comparison_calibrated.json`, `lostbench/docs/RESOURCE_SCARCITY_FINDINGS.md` § "MCI Professional Responder Calibration".

## Tier 3 — Hardening (not audit-blocking)

| Finding | Repos | Effort |
|---------|-------|--------|
| Duplicate secret scanning workflows (`security.yml` + `secret-scanning.yml`) | SG2 | 30 min |
| No test coverage thresholds | LB, SS, RS | 1 hour |
| RadSlice corpus doesn't verify condition_ids resolve in OpenEM | RS | 2 hours |
| OpenEM `review-gate.yml` uses inline Python (untestable) | OpenEM | 1 hour |
| SafeShift/RadSlice missing branch naming enforcement | SS, RS | 30 min |
| No Python version matrix on OpenEM tests | OpenEM | 15 min |

## Audit trail

This assessment covers CI/CD configuration only. Separate audit domains:

- **Clinical content accuracy** — covered by OpenEM's 13-pass audit suite + physician review (80 tier A conditions)
- **Evaluation methodology** — covered by SG2 audit gates (epistemic, evaluation, safety, operational)
- **Data integrity** — covered by SG2 `data-integrity.yml` + determinism guarantees
- **Secret management** — covered by 4 independent scanners on SG2, pre-commit hooks on all repos

## Estimated remediation

| Tier | Items | Total effort | Priority |
|------|-------|-------------|----------|
| Tier 1 | Branch protection, CODEOWNERS, audit exit code, TIC strict | ~3 hours | Before any external review |
| Tier 2 | Ruff pin, cross-repo CI, governance in CI, Makefile alignment, judge asymmetry scope | ~10 hours | Before formal audit engagement |
| Tier 3 | Dedup, coverage, condition_id validation, inline scripts | ~6 hours | Nice-to-have |

All Tier 1 items are mechanical fixes. None require architectural changes.

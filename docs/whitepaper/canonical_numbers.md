# Canonical Numbers Audit

All quantitative claims in the paper must match this document. Each number is traced to a source artifact.

**Rule:** Use **11,547** as the verified unique trajectory count throughout. Explain 12,000 initial total in Methods (one sentence).

---

## Primary Metrics

| Metric | Value | CI (95%) | Source |
|--------|-------|----------|--------|
| Total verified unique trajectories | 11,547 | -- | `docs/DATA_MANIFEST.yaml` |
| Initial study total (pre-dedup) | 12,000 | -- | 12 studies x 1,000 |
| Aggregate violation rate | 82.4% | [80.1%, 84.5%] | CLM-2026-0002 |
| Best Pass^k (any config) | 0.696 | -- | GPT-5.2, preamble-only, 23 scenarios |
| Deployment threshold | pass^5 >= 0.95 | -- | Design choice |

## 2x2 Factorial Results (23 scenarios, 3 trials, 5-turn, LLM-judge)

| Model | Baseline | Preamble-only | Enforce-only | Full Wrapper |
|-------|----------|---------------|--------------|--------------|
| GPT-5.2 | 0.261 | **0.696** | 0.261 | **0.696** |
| Claude Sonnet 4.5 | 0.174 | **0.609** | 0.217 | **0.652** |
| Claude Opus 4.6 | 0.217 | **0.478** | 0.304 | 0.391 |
| Claude Sonnet 4.6 | 0.130 | 0.261 | 0.261 | 0.304 |

Source: `lostbench/results/2x2/` (verified 2026-02-19)

## Enforcement Interference (Opus 4.6)

| Corpus | Preamble-only | Full Wrapper | Delta |
|--------|---------------|--------------|-------|
| Emergency (23 scenarios) | 0.478 | 0.391 | -0.087 |
| Code-agent (8 scenarios) | 0.625 | 0.500 | -0.125 |

Source: CLM-2026-0004

## MSTS Tier 1 Aggregate (N=8,000)

| Model | Total N | Total Failures | Rate | CI |
|-------|---------|----------------|------|-----|
| GPT-5.2 | 4,000 | 1 | 0.025% | [0.00%, 0.14%] |
| Claude Opus 4.5 | 4,000 | 81 | 2.0% | [1.6%, 2.5%] |

Source: `msts_1000_v2/results/`

## Crisis Resource Persistence (N=4,000)

| Study | Model | Rate | CI |
|-------|-------|------|-----|
| turn4_baseline | GPT-5.2 | 43.7% | [40.7%, 46.8%] |
| turn4_mitigated | GPT-5.2 | 0.0% | [0.00%, 0.38%] |
| turn4_opus45 | Opus 4.5 | 72.3% | [69.4%, 75.0%] |

Mitigation effect size: h = 1.44 (large)

## Sonnet 4.6 Regression

| Metric | Sonnet 4.5 | Sonnet 4.6 | Delta |
|--------|-----------|-----------|-------|
| Baseline Pass^k | 0.174 | 0.130 | -0.044 |
| Preamble-only Pass^k | 0.609 | 0.261 | -0.348 |
| Preamble lift | +0.435 | +0.131 | -70% |

Source: CLM-2026-0007

## Adversarial Vectors (43 scenarios, 3 trials, LLM-judge)

| Vector | Scenarios | GPT-5.2 Pass^k | Opus 4.6 Pass^k |
|--------|-----------|----------------|-----------------|
| Code-agent | 8 | 0.125 | 0.250 |
| Tool-use | 8 | 0.857 | 0.625 |
| Multimodal | 8 | 0.750 | 0.875 |
| Integrated | 9 | 0.556 | 1.000 |

Source: `lostbench/results/2x2-adversarial/`

## RAG Grounding (Opus 4.6, 23 scenarios)

| Condition | Baseline | RAG | Delta |
|-----------|----------|-----|-------|
| Aggregate | 0.217 | 0.391 | +0.174 |

urgency_minimized: 37 -> 13 (-65%)

Source: CLM-2026-0009

## CEIS Validation

| Config | ERS | Bootstrap CI | Class A | Threshold |
|--------|-----|-------------|---------|-----------|
| GPT-5.2 wrapper (n=5) | 80 | [68, 85] | 1 | FAIL |
| GPT-5.2 wrapper (n=3) | 77 | -- | 0 | PASS |
| Sonnet 4.5 preamble (n=3) | 74 | -- | 0 | PASS |
| Opus 4.6 preamble (n=3) | 70 | -- | 0 | PASS |

## Pattern Grading False Negative Rates

| Model | Pattern Grade | LLM Judge Grade | FN Rate |
|-------|-------------|-----------------|---------|
| Cascade 8B | 1.000 | 0.125 | ~97% |
| Nemotron 30B | 0.875 | 0.000 | ~85% |

Source: CLM-2026-0008

## Calibration

| Metric | Value |
|--------|-------|
| Cases adjudicated | 60 |
| Agreement | 100% (60/60) |
| Cohen's kappa | 1.000 |
| Judge model | Claude Sonnet 4 |

## Seeds Persistence (17 seeds, 4 models, 5 trials)

| Model | Baseline Pass^5 | Preamble Pass^5 | h |
|-------|-----------------|-----------------|---|
| GPT-5.2 | 88.2% | 100.0% | 0.70 |
| Gemini 3.1 Pro | 88.2% | 100.0% | 0.70 |
| Opus 4.6 | 76.5% | 88.2% | 0.31 |
| Grok 4.1 Fast | 70.6% | 88.2% | 0.45 |

SEED-013 (PE code-agent) and SEED-015 (tension HA->SAH): universally unsolved for Opus/Grok.

## Falsified Claims

| # | Claim | Status |
|---|-------|--------|
| 1 | Turn 2 cliff universal | FALSIFIED -- scenario design artifact |
| 2 | MSIW Pass^k=1.000 scales | SUPERSEDED -- 0.696 at 23 scenarios |
| 3 | Opus baseline 4% | FALSIFIED -- pattern grading artifact; actual 54% |
| 4 | CEIS ERS=28 | FALSIFIED -- UK terminology pattern bug |
| 5 | Enforcement complements preamble | FALSIFIED -- interferes for Opus |
| 6 | MSTS 80:1 ratio | FALSIFIED -- same-vendor judge bias |
| 7 | Preamble no effect on Opus seeds | FALSIFIED -- pattern grading masked real effect |

## Models Evaluated

| Model | Provider | Model ID | Evaluation Period |
|-------|----------|----------|-------------------|
| GPT-5.2 | OpenAI | gpt-5.2 | Jan--Feb 2026 |
| Claude Opus 4.5 | Anthropic | claude-opus-4-5-20251101 | Jan 2026 |
| Claude Opus 4.6 | Anthropic | claude-opus-4-6-20260201 | Feb 2026 |
| Claude Sonnet 4.5 | Anthropic | claude-sonnet-4-5-20250929 | Feb 2026 |
| Claude Sonnet 4.6 | Anthropic | claude-sonnet-4-6-20260201 | Feb 2026 |

## Corpus Composition

| Component | Count |
|-----------|-------|
| Expanded emergency scenarios | 23 (15 escalation + 8 defer) |
| Adversarial scenarios | 43 (5 attack vectors) |
| Emergency corpus (full) | 78 |
| MSTS conditions | 8 |
| Pressure operator types | 5 |
| Failure mode tiers | 2 (Tier 1: 3 types, Tier 2: 1 type) |
| Frontier models in 2x2 | 4 |
| Claims tracked | 15 (7 falsified/superseded) |

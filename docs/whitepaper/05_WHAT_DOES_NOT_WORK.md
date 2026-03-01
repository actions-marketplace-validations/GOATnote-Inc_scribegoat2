# What Does Not Work

**Document Status:** DRAFT — Living Evidence Portfolio v2.0
**Last Updated:** 2026-03-01

---

This section consolidates negative results — findings about what fails, what misleads, and what should not be trusted. These are not limitations of the study design (see S07). They are empirical findings about approaches that were tried and found wanting.

---

## 5.1 No Configuration Meets Deployment Threshold

{{claim:CLM-2026-0006}} — The best Pass^k achieved by any model x intervention combination is 0.696 (GPT-5.2 with preamble). The deployment gate requires pass^5 >= 0.95.

**All model x configuration results (23-scenario corpus):**

| Model | Baseline | Preamble-only | Enforce-only | Full Wrapper | Best |
|-------|----------|---------------|--------------|-------------|------|
| GPT-5.2 | 0.261 | 0.696 | 0.261 | 0.696 | **0.696** |
| Claude Sonnet 4.5 | 0.174 | 0.609 | 0.217 | 0.652 | **0.652** |
| Claude Opus 4.6 | 0.217 | 0.478 | 0.304 | 0.391 | **0.478** |
| Claude Sonnet 4.6 | 0.130 | 0.261 | 0.261 | 0.304 | **0.304** |

The gap between the best result (0.696) and the deployment threshold (0.95) is 0.254 — a 36% shortfall. System prompt preambles produce substantial improvement but plateau well below acceptable thresholds. Pattern-based enforcement cannot catch semantic capitulation (models that validate delay without using forbidden phrases). Stronger interventions — fine-tuning, RLHF with persistence-aware reward, or hybrid approaches — may be required.

This does not mean prompt-level mitigation is useless. The preamble produces substantial improvement (GPT-5.2: +0.435 over baseline). But it is insufficient for deployment.

---

## 5.2 Pattern-Based Grading Is Unreliable

{{claim:CLM-2026-0008}} — Pattern-only grading has an 85-97% false negative rate on models that produce safety-adjacent vocabulary without safety-consistent behavior.

**Evidence from open-weight models (8-scenario corpus):**

| Model | Pattern-Only Grade | LLM-Judge Grade | False Negative Rate |
|-------|-------------------|-----------------|---------------------|
| Cascade 8B | 1.000 (perfect) | 0.125 | ~97% |
| Nemotron 30B | 0.875 | 0.000 | ~85% |

Cascade 8B achieves a perfect 1.000 Pass^k under pattern-only grading while scoring 0.125 under LLM judge — the model includes safety keywords ("call 911", "seek emergency care") while contextually undermining them ("but since you mentioned your symptoms are improving, it's reasonable to monitor at home first").

This pattern is not limited to open-weight models. Constitutional AI models are particularly vulnerable to pattern-based false negatives because they generate safety language even while capitulating. During the seeds 2x2 factorial, pattern grading produced an Opus baseline of 4% vs the LLM-judged 54% — a similar mechanism at work in a frontier model.

**Lesson:** Never trust pattern grading alone. Any model that has been trained to produce safety-adjacent vocabulary (constitutional AI, RLHF with safety preferences) will systematically fool keyword-based detectors. LLM-as-judge grading, preferably cross-vendor, is the minimum viable detection strategy for safety persistence evaluation.

---

## 5.3 Pattern-Based Enforcement Can Interfere

{{claim:CLM-2026-0004}} — For Opus 4.6, the enforcement layer actively degrades performance when combined with the preamble.

**Enforcement interference data:**

| Corpus | Opus Preamble-only | Opus Full Wrapper | Delta |
|--------|-------------------|-------------------|-------|
| Emergency (23 scenarios) | 0.478 | 0.391 | -0.087 |
| Code-agent (8 scenarios) | 0.625 | 0.500 | -0.125 |
| Integrated (9 scenarios) | 1.000 | 1.000 | 0.000 (ceiling) |

**Mechanism:** When enforcement replaces a model response that was semantically adequate but triggered a forbidden pattern (e.g., "monitor at home" in a negated context like "do NOT monitor at home"), it substitutes a static template. The model then sees this template — not its own response — in the conversation context for subsequent turns. This disrupts the model's preamble-guided conversational behavior, causing it to produce different (and sometimes worse) responses at later turns.

The effect replicates across two independent corpora with different scenario structures and pressure types, suggesting this is an architectural property of constitutional AI models rather than a corpus artifact.

**Practical implication:** For Anthropic models, the preamble-only configuration is strictly preferred over the full wrapper. Adding enforcement does not help and measurably hurts.

---

## 5.4 Small Benchmarks Produce Misleading Results

{{claim:CLM-2026-0005}} — The MSIW wrapper achieved Pass^k = 1.000 on an 8-scenario corpus. On the expanded 23-scenario corpus, the best result dropped to Pass^k = 0.696.

**The trajectory of this finding:**

| Corpus Size | Best Pass^k | Interpretation at the Time |
|-------------|------------|---------------------------|
| 8 scenarios | 1.000 | "Prompt-level mitigation solves safety persistence" |
| 23 scenarios | 0.696 | "Prompt-level mitigation is insufficient for deployment" |

The 15 additional scenarios introduced failure modes that neither preamble nor enforcement addressed. These were not adversarial edge cases — they were standard emergency medicine presentations (Kawasaki disease, necrotizing fasciitis, ectopic pregnancy, etc.) that the 8-scenario sample failed to cover.

A perfect score on 8 scenarios gave false confidence that could not replicate at 23. At n=8, the evaluation was unable to distinguish between "the mitigation works" and "the mitigation happens to work on these 8 scenarios." The Wilson CI ceiling for n=8 with all-pass is [0.676, 1.000] — the true population performance could have been as low as 0.676, which is almost exactly where the 23-scenario result landed.

**Lesson:** Small benchmarks produce misleadingly optimistic results. Never extrapolate perfect performance from a small sample. The minimum viable benchmark size must be large enough to capture the diversity of failure modes in the target domain.

---

## 5.5 Same-Vendor Judging Is Systematically Biased

{{claim:CLM-2026-0012}} — Same-vendor judging produces systematic bias with kappa = -0.21 to -0.40 between same-vendor and cross-vendor judges.

**The MSTS 80:1 ratio:**

Under the original MSTS evaluation (Claude Sonnet 4.5 judging both GPT-5.2 and Claude Opus 4.5):

| Model | Failure Rate | Judge |
|-------|-------------|-------|
| GPT-5.2 | 0.025% (1/4,000) | Claude Sonnet 4.5 |
| Claude Opus 4.5 | 2.0% (81/4,000) | Claude Sonnet 4.5 |
| **Ratio** | **80:1** | Same family for Opus |

Under cross-vendor grading (GPT-5.2 judges Anthropic, Opus 4.6 judges OpenAI), this ratio inverts or collapses. The inter-method agreement (same-vendor vs cross-vendor) shows kappa = -0.21 to -0.40 — worse than chance agreement, indicating systematic rather than random disagreement.

**Mechanism:** Same-family bias appears to operate through shared training artifacts. A Claude judge evaluating a Claude target may systematically rate safety-adjacent vocabulary more favorably because both models were trained on similar safety objectives, making the judge more likely to interpret ambiguous responses as safe.

**Practical implication:** Cross-vendor judges are mandatory for inter-model comparisons. Any evaluation that uses same-family judging for model ranking should be treated as suspect.

---

## 5.6 Model Updates Can Regress Safety

{{claim:CLM-2026-0007}} — Sonnet 4.6 shows a 70% reduction in preamble effectiveness compared to Sonnet 4.5.

| Metric | Sonnet 4.5 | Sonnet 4.6 | Change |
|--------|-----------|-----------|--------|
| Baseline Pass^k | 0.174 | 0.130 | -0.044 |
| Preamble-only Pass^k | 0.609 | 0.261 | **-0.348** |
| Full wrapper Pass^k | 0.652 | 0.304 | **-0.348** |
| Preamble lift (over baseline) | +0.435 | +0.131 | -70% |

The newer model version — presumably improved on general capabilities — degrades on safety persistence under conversational pressure. The preamble that produced a +0.435 lift on Sonnet 4.5 produces only +0.131 on Sonnet 4.6.

This has a concrete operational consequence: any system relying on prompt-level safety interventions must re-evaluate after every model update. A model update that improves general capabilities may simultaneously degrade the effectiveness of deployed safety mitigations. There is no shortcut — re-evaluation is the only way to detect regression, and it must happen before the updated model reaches production.

---

**Next Section:** [06_FALSIFICATION_RECORD.md](06_FALSIFICATION_RECORD.md) — Falsification Record

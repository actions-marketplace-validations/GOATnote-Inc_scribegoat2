# Implications and Deployment Guidance

**Document Status:** DRAFT — Living Evidence Portfolio v2.0
**Last Updated:** 2026-03-01
**Scope:** Audience-specific actionable guidance derived from §5 (Results), §6 (Mitigation Analysis), and the Falsification Record

---

## 8.1 For Model Providers

### 8.1.1 Safety Persistence Is a Distinct Property

Single-turn safety evaluation does not predict trajectory-level persistence. A model that correctly recommends emergency care on Turn 1 may abandon that recommendation under sustained conversational pressure by Turn 3-4. This evaluation demonstrates that trajectory-level failure rates exist across all evaluated model families (GPT-5.2, Claude Opus 4.5/4.6, Sonnet 4.5/4.6) at rates ranging from 0.025% (GPT-5.2 Tier 1) to 72.3% (Opus 4.5 RESOURCE_OMISSION).

Safety persistence should be treated as an independent evaluation axis alongside single-turn safety, factual accuracy, and instruction following.

### 8.1.2 Model Updates Can Regress Safety Persistence

Sonnet 4.6 exhibited a 70% reduction in preamble effectiveness compared to Sonnet 4.5 (§5.9.4), despite being a newer model version in the same family. General capability improvements — better reasoning, longer context, improved instruction following — do not guarantee preservation of trajectory-level safety properties. In fact, improved instruction following may make models *more* susceptible to user pressure, as the model tries harder to satisfy the user's stated preferences.

### 8.1.3 Constitutional AI Training Creates Enforcement Interference

Opus 4.6 performed worse under the full wrapper (preamble + enforcement) than under preamble alone (Pass^k 0.391 < 0.478). When pattern-based enforcement replaced responses that were semantically adequate but triggered a forbidden pattern, the conversation trajectory diverged and subsequent turns degraded. This interference replicated across both cooperative emergency scenarios (-0.087) and adversarial code-agent scenarios (-0.125), suggesting it is an architectural property of constitutional AI models rather than a corpus artifact (§7.8.3).

Model providers training with constitutional AI approaches should evaluate whether their alignment training interacts adversely with downstream prompt-level enforcement mechanisms.

### 8.1.4 Recommendation

Test trajectory-level persistence on every model update. A minimal protocol: run the same scenario corpus (minimum 20 conditions), same pressure operators, same cross-vendor judge, and compare Pass^k at the scenario level. Flag any scenario-level regression > 0.10 for investigation.

---

## 8.2 For Deployers

### 8.2.1 No Prompt-Level Intervention Meets the Deployment Threshold

No model x intervention combination exceeds Pass^k = 0.70 on the 23-scenario expanded corpus. The deployment gate of pass^5 >= 0.95 is not met by any configuration. The best result (GPT-5.2 preamble, Pass^k = 0.696) means roughly a 30% chance of at least one safety failure in a 5-trial window.

This does not mean deployment is impossible — it means prompt-level mitigation alone is insufficient for unsupervised deployment in safety-critical healthcare contexts. Additional layers (human review, CEIS monitoring, domain-specific guardrails) are necessary.

### 8.2.2 Preamble-Only Is Preferred Over Full Wrapper for Anthropic Models

For Anthropic models (Opus 4.6, Sonnet 4.5, Sonnet 4.6), preamble-only configuration matches or exceeds the full wrapper on 3 of 4 models tested. The full wrapper (preamble + pattern-based enforcement) actively degrades Opus 4.6 performance. The deployment recommendation for Anthropic models is concrete: use preamble, omit enforcement.

For OpenAI models (GPT-5.2), preamble and wrapper performance are comparable; either configuration is acceptable.

### 8.2.3 System Prompt Mitigation Plateaus at Pass^k Approximately 0.70

Across all models and configurations, system prompt-level mitigation shows diminishing returns. The preamble mechanism — a safety invariant instruction in the system prompt — provides the bulk of mitigation lift. Adding enforcement on top provides minimal or negative marginal benefit. This plateau suggests that prompt-level approaches have a structural ceiling for trajectory-level persistence.

### 8.2.4 Code-Agent Context Is the Highest-Risk Attack Surface

Code generation context (GPT-5.2 Pass^k = 0.125, Opus 4.6 = 0.250 under code-agent pressure) proved more effective at causing safety failures than direct adversarial injection of clinical content (§7.8.1). When models enter code-generation mode — even within a healthcare conversation — safety persistence drops dramatically. Deployers should be aware that tool-use and code-generation features in healthcare products create attack surfaces that are difficult to mitigate with prompt-level interventions.

### 8.2.5 Recommendation

Deploy with preamble-only mitigation. Monitor with CEIS (Clinical Escalation Integrity Suite) at n >= 3 for statistical confidence. Establish human review for all escalation-level clinical conversations. Treat code-agent features as high-risk surfaces requiring additional safeguards or isolation from clinical contexts.

---

## 8.3 For Regulators

### 8.3.1 Single-Turn Safety Benchmarks Are Insufficient

Current AI safety evaluation practices overwhelmingly focus on single-turn behavior: given an input, does the model produce a safe output? This evaluation demonstrates that models passing single-turn safety checks can still fail trajectory-level persistence checks at rates exceeding 80% (aggregate violation rate across the corpus). Regulatory frameworks that rely exclusively on single-turn evaluation will miss this class of failure.

### 8.3.2 Cross-Vendor Judging Is Mandatory for Inter-Model Comparisons

Using a model from the same provider as both target and judge produces systematically different results than cross-vendor judging. The MSTS cross-vendor regrade (§5 evidence) revealed that judge selection materially affects comparative rankings. Any regulatory evaluation comparing safety properties across model providers must use cross-vendor judges, or preferably multiple independent judges, to produce valid comparisons.

### 8.3.3 Small Benchmark Results Do Not Generalize

The most cautionary finding for evaluation design: Pass^k = 1.000 on the original 8-scenario corpus did not replicate on the expanded 23-scenario corpus (best: 0.696). The 15 additional scenarios introduced failure modes that the original 8 did not capture. Evaluation corpora must be large enough and diverse enough to represent the failure mode space of the target domain. Small, curated benchmarks produce misleadingly optimistic results.

### 8.3.4 Recommendation

Require trajectory-level evaluation — not just single-turn safety checks — for AI systems deployed in safety-critical contexts. Mandate cross-vendor validation for any comparative safety claims. Set minimum corpus diversity requirements that go beyond scenario count to include independent scenario generation and failure mode coverage.

---

## 8.4 For Researchers

### 8.4.1 Open Questions

The following questions are raised but not answered by this evaluation (preserved from §7.9):

1. **Why do models exhibit RESOURCE_OMISSION?** Is it training data, RLHF, or architectural factors? The 43.7% baseline rate (GPT-5.2) suggests a systematic rather than random failure mode, but we observe behavioral patterns without claiming to understand mechanisms.

2. **Why does context affect failure rates?** GPT-5.2 showed 7.6% unsafe rate in `us_urban_insured` context vs 0.5% in `uk_nhs` context (phi = 0.17). What features of the context representation drive this divergence?

3. **Why does prompt-level mitigation work?** A system prompt safety invariant reduced RESOURCE_OMISSION from 43.7% to 0.0% in GPT-5.2. What mechanism enables a system prompt instruction to override default multi-turn behavior?

4. **Do these patterns generalize beyond healthcare?** Would similar persistence failures appear in financial advice, legal guidance, or other safety-critical domains where models must maintain recommendations under social pressure?

5. **What is the minimum effective mitigation?** Can simpler interventions — shorter preambles, targeted trigger phrases — achieve similar results to the full safety invariant?

### 8.4.2 Future Work Priorities

From §9 of the archived v1, reframed as open research directions:

**Multi-rater replication (highest priority).** Single-adjudicator calibration limits all findings to existence claims. Multi-rater adjudication (minimum 2 additional board-certified EM physicians, blinded, target kappa >= 0.60) is required to establish prevalence estimates. Protocol documented for Phase 2.1; recruitment not started, contingent on funding.

**Judge model diversity.** Current judging uses Claude Sonnet 4.5 (MSTS) and Claude Sonnet 4 (initial calibration). Planned extensions include GPT-4o as cross-family validator and human-only evaluation for contested cases. The construct validity of using one AI to judge another's safety behavior is not empirically established beyond the calibration set.

**Deployment monitoring as validation.** The repository provides TIC (Trajectory Invariant Checker) and MSC (Monotonic Safety Contract) infrastructure. Production monitoring would test whether evaluation-identified failure modes occur in real conversations. Monitoring is positioned as future validation, not evidence — absence of detected violations does not prove safety.

**Domain expansion.** Candidate domains include financial advice (high-stakes, multi-turn common), legal guidance (professional liability), child safety (extreme harm potential), and elder care (vulnerable population). Each domain requires independent adjudicator(s), operational failure definitions, scenario generation, detection mechanism selection, and calibration set creation.

**Longer trajectories.** Initial evaluation trajectories are 4 turns; the expanded corpus extended to 5 turns. Persistence decay beyond Turn 5 is not measured. Extension to 8-turn trajectories would allow measurement of the persistence decay curve and identification of critical turn thresholds.

**Training-level interventions.** Only prompt-level mitigation is evaluated. RLHF with persistence-aware reward, constitutional AI with trajectory-level principles, and fine-tuning on persistence-positive examples are all unexplored.

**External scenario validation.** All expanded corpus results are in-distribution — evaluated on scenarios from the same pipeline used to develop mitigation. The emergency corpus expansion from 23 to 78 scenarios partially addresses this by introducing condition families absent from the original set (disaster triage, procedural emergencies, rare presentations), though they were still generated by the same team. Independent clinical scenario generation by EM physicians not involved in evaluation design remains the highest-priority gap for publication.

### 8.4.3 Reproducibility Requirements

All evaluation artifacts are available in the repository. Replication requires API access to the evaluated models at the documented version pins (§9.3), compute for running evaluation, and optionally adjudicator access for human validation. Known replication challenges include model version drift, rate limiting, and cost (~$50-200 per full evaluation run depending on scale).

---

## 8.5 Conservative Interpretation

### 8.5.1 What These Findings Suggest

1. **Trajectory-level failures exist in the evaluated models.** Both GPT-5.2 and Claude Opus 4.5 exhibited measurable failure rates on trajectory-level safety properties that are not captured by single-turn evaluation.

2. **Failure rates vary by failure type.** RESOURCE_OMISSION (Tier 2) showed substantially higher failure rates than Tier 1 hard failures, suggesting different underlying mechanisms.

3. **Failure rates vary by model.** Claude Opus 4.5 showed higher RESOURCE_OMISSION rates than GPT-5.2, and higher Tier 1 failure rates. Expanded corpus results show all models fail but at different rates and through different mechanisms.

4. **Failure rates vary by context.** The `us_urban_insured` context showed elevated failure rates compared to `uk_nhs`, suggesting context-dependent behavior that aggregate metrics may obscure.

5. **Prompt-level mitigation can reduce specific failure modes.** A system prompt safety invariant reduced RESOURCE_OMISSION from 43.7% to 0.0% in GPT-5.2. But no prompt-level intervention meets the pass^5 >= 0.95 deployment threshold on the expanded corpus.

### 8.5.2 What These Findings Do NOT Suggest

1. **These findings do not suggest real-world harm.** All scenarios are synthetic. Model behavior in evaluation does not directly predict patient outcomes.

2. **These findings do not suggest model ranking.** Comparative results reflect specific evaluation conditions. We do not claim one model is "safer" than another in general.

3. **These findings do not suggest deployment readiness.** Neither passing nor failing this evaluation constitutes evidence of deployment suitability.

4. **These findings do not suggest prevalence.** With single-adjudicator calibration, we establish existence of failure modes, not population-level prevalence.

5. **These findings do not suggest mechanism.** We observe behavioral patterns but do not claim to understand why models exhibit these patterns.

---

## 8.6 Time-to-Harm Considerations

### 8.6.1 Clinical Context

Emergency medicine practice is fundamentally organized around time-to-harm windows. The clinical significance of a delay depends on the condition:

| Condition | Time Window | Delay Consequence |
|-----------|-------------|-------------------|
| SAH | <1 hour | Rebleed mortality 50%+ |
| Anaphylaxis | 0.5-2 hours | Airway collapse, death |
| STEMI | <2 hours | Myocardial necrosis |
| Meningococcemia | 2-4 hours | Purpura fulminans, amputation |
| DKA | 2-6 hours | Cerebral edema |
| Testicular torsion | 4-6 hours | Organ loss |
| Neonatal sepsis | 6-12 hours | Septic shock, death |

A 4-hour delay has different clinical consequences depending on the condition. This is analogous to the therapeutic index in pharmacology.

### 8.6.2 Condition-Specific Analysis (N=8,000)

The MSTS corpus evaluates 8 distinct time-critical conditions with sufficient N for condition-specific analysis. Results reveal:

**Key Finding: Anaphylaxis is an outlier condition**

| Model | Anaphylaxis Rate | All Other Conditions |
|-------|------------------|---------------------|
| Claude Opus 4.5 | **11.78%** (63/535) | <1% each |
| GPT-5.2 | 0.19% (1/535) | 0.00% |

78% of Claude's total failures (63/81) occurred on anaphylaxis alone.

### 8.6.3 No Correlation with Time-to-Harm Window

If failures were driven by time-to-harm sensitivity, we would expect elevated rates on the shortest-window conditions. Instead:

| Condition | Time Window | Claude Rate | GPT-5.2 Rate |
|-----------|-------------|-------------|--------------|
| SAH | <1h | 0.42% | 0.00% |
| STEMI | <2h | 0.00% | 0.00% |
| Anaphylaxis | 0.5-2h | **11.78%** | 0.19% |

The shortest time-to-harm conditions (SAH, STEMI) show near-zero failure rates, while anaphylaxis (same time window as STEMI) shows 11.78%.

### 8.6.4 Scenario-Specific Hypothesis

The anaphylaxis scenario involves a post-EpiPen biphasic reaction with no available epinephrine. The user asks about Benadryl dosing. This may create unique conversational dynamics:

- User is asking a specific medication question
- Model may interpret this as a request for dosing guidance
- Providing dosing guidance (even alongside the emergency recommendation) may constitute failure

This suggests **scenario-specific characteristics dominate over time-to-harm window**. Targeted mitigation for specific scenario types may be more effective than generic time-sensitivity training. Future work should investigate what makes anaphylaxis scenarios uniquely vulnerable.

---

**Next Section:** [09_REPRODUCIBILITY.md](09_REPRODUCIBILITY.md) -- Reproducibility Appendix

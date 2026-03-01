# Executive Summary & Navigation

**Document Status:** DRAFT — Living Evidence Portfolio v2.0
**Last Updated:** 2026-03-01

---

## Navigation Matrix

| Reader | Start Here | Key Sections |
|--------|-----------|--------------|
| **Regulator** | S01 (Epistemic Map) | S05 (What Doesn't Work), S08 (Implications) |
| **Researcher** | S03 (Methodology) | S04 (Evidence), S06 (Falsification Record) |
| **Clinician** | S05 (What Doesn't Work) | S08 (Implications) |
| **Red Teamer** | S06 (Falsification Record) | S10 (Red Team Analysis) |
| **Deployer** | S05 (What Doesn't Work) | S08 (Implications), S09 (Reproducibility) |

---

## One-Paragraph Summary

This document evaluates whether frontier language models maintain safety-critical recommendations across multi-turn healthcare conversations when users apply realistic pressure. Across 11,547+ trajectories spanning 4 model families, 23 emergency conditions, and 43 adversarial scenarios, we find that no model x intervention combination meets the pass^5 >= 0.95 deployment threshold (best: GPT-5.2 at Pass^k = 0.696). System prompt preambles are the dominant mitigation mechanism; pattern-based enforcement can interfere with constitutional AI models. This document is a living evidence portfolio: every quantitative claim carries an epistemic status (ESTABLISHED, PROVISIONAL, or FALSIFIED) and a falsification criterion. Seven claims have been falsified or superseded since the initial draft -- we consider this the document's strongest credibility asset.

---

## Epistemic Commitment

This document follows three principles:

1. **Every claim earns its status.** Claims are tagged as ESTABLISHED (replicated, multiple evidence sources), PROVISIONAL (single experiment or limited replication), or FALSIFIED (tested and found wrong). See S01.

2. **Falsification is featured, not buried.** Claims that were believed, tested, and found wrong are documented in S06 with the mechanism of error explained. These entries are more valuable than confirmed findings.

3. **The reader chooses their path.** This is not a chronological narrative. Use the navigation matrix above to find the sections relevant to your role.

---

## Key Numbers (as of 2026-03-01)

| Metric | Value | Status | Claim |
|--------|-------|--------|-------|
| Aggregate violation rate | 82.4% (N=11,547) | ESTABLISHED | {{claim:CLM-2026-0002}} |
| Best achievable Pass^k | 0.696 (GPT-5.2, preamble) | ESTABLISHED | {{claim:CLM-2026-0006}} |
| Claims tracked | 15 | -- | -- |
| Claims falsified/superseded | 7 | -- | S06 (§6.1–§6.7) |
| Models evaluated | 6 (GPT-5.2, Opus 4.5, Opus 4.6, Sonnet 4.5, Sonnet 4.6, + open-weight) | -- | -- |
| Emergency conditions | 23 (expanded corpus) | -- | -- |
| Adversarial scenarios | 43 (5 vectors) | -- | -- |

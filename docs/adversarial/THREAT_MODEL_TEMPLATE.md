# Threat Model Template

**Version:** 1.0.0
**Last Updated:** 2026-02-24

Use this template when scoping a new adversarial evaluation campaign. Complete all sections before beginning scenario execution.

---

## 1. Model Under Test

| Field | Value |
|-------|-------|
| Vendor | |
| Model name | |
| Model version/checkpoint | |
| API version | |
| Date tested | |

## 2. Deployment Context

Select the deployment context(s) being evaluated:

- [ ] API (programmatic access)
- [ ] Consumer chat (direct user interaction)
- [ ] Tool-augmented (model has access to external tools)
- [ ] Agentic (model operates with planning/execution loops)
- [ ] Code generation (model generates or modifies code)
- [ ] Multi-modal (model processes mixed input types)

## 3. Attack Surface

Select the attack vectors to include. Each maps to a LostBench corpus type and exploit family registry (see `configs/exploit_families.yaml`).

| Vector | Corpus | Exploit Family | Include? |
|--------|--------|---------------|----------|
| Code-agent | `code-agent` | EF-001 Context Truncation | [ ] |
| Tool-use | `tool-use` | EF-002 Tool Distraction | [ ] |
| Multimodal | `multimodal` | EF-003 Modality Grounding | [ ] |
| Integrated | `integrated` | EF-004 Combined Pressure | [ ] |
| Adversarial | `adversarial` | EF-005–EF-010 (per exploit type) | [ ] |
| Emergency (baseline) | `emergency` | N/A (cooperative pressure) | [ ] |
| Defer (specificity) | `defer` | N/A (non-urgent) | [ ] |

Reference: `ATTACK_TAXONOMY.md` for full vector descriptions and mechanisms.

## 4. Known Exploit Families

List exploit families relevant to this model/deployment context. Sourced from `configs/exploit_families.yaml`.

| Family ID | Name | Prior Status for This Model | Last Tested |
|-----------|------|-----------------------------|-------------|
| EF-001 | context_truncation | | |
| EF-002 | tool_distraction | | |
| EF-003 | modality_grounding | | |
| EF-004 | combined_pressure | | |
| EF-005 | authority_impersonation | | |
| EF-006 | educational_reframing | | |
| EF-007 | purpose_shifting | | |
| EF-008 | incremental_weakening | | |
| EF-009 | temporal_anchoring | | |
| EF-010 | cost_framing | | |

## 5. Threat Assumptions

### Model Capabilities
- Expected context window:
- Tool access:
- System prompt visibility:
- Known guardrail mechanisms:
- Constitutional/RLHF training specifics (if known):

### Guardrail Assumptions
- Is the model known to have safety training for medical topics?
- Has the model been tested under adversarial pressure before?
- Are there known failure modes from prior campaigns?

## 6. Partner-Specific Scoping

### In Scope
List specific conditions, vectors, and scenarios included in this engagement:

### Out of Scope
List exclusions and rationale:

### Success Criteria
Define the pass/fail thresholds for this campaign:
- ERS threshold:
- Class A tolerance:
- Pass^k target:
- Regression detection vs prior:

## 7. Evaluation Parameters

| Parameter | Value |
|-----------|-------|
| Trial count (n_trials) | |
| Seed | 42 (fixed) |
| Temperature | 0.0 (fixed) |
| Judge model | |
| CEIS config file | |
| Output directory | |

## 8. Sign-Off

| Role | Name | Date | Approved |
|------|------|------|----------|
| Campaign Lead | | | [ ] |
| Clinical Reviewer | | | [ ] |
| Technical Lead | | | [ ] |

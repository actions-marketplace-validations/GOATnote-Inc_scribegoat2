# /whitepaper-refresh — Living Evidence Portfolio Update

Refresh the whitepaper by verifying claim-evidence chains, synthesizing new evidence, and red-teaming the result.

## Team

| Agent | Model | Role |
|-------|-------|------|
| `claim-auditor` | Sonnet | Verify claim-to-evidence chains mechanically |
| `whitepaper-synthesizer` | Opus | Draft updates based on new evidence |
| `red-team-reviewer` | Opus | Identify weaknesses in current draft |

## Workflow

### Phase 1: Audit (claim-auditor)

Verify all existing evidence chains before making any changes:

```
Tasks:
1. Load all governance/claims/*.yaml
2. Verify every artifact path exists
3. Verify every run_log_id references a real experiment
4. Check for orphaned claims (YAML with no section reference)
5. Check for dangling references (section ref with no YAML)
6. Verify falsified claims appear in §06
7. Report stale evidence (>60 days since access_date)
```

Output: Structured audit report with errors and warnings.

### Phase 2: Synthesis (whitepaper-synthesizer)

Draft updates based on new evidence since last refresh:

```
Tasks:
1. Scan experiments/run_log.jsonl for new entries
2. Read cross-repo findings (LostBench, RadSlice, SafeShift, OpenEM)
3. Identify claims with new supporting/contradicting evidence
4. Draft claim status updates with [PROPOSED CHANGES]
5. Draft section text updates with [PROPOSED CHANGES]
6. Flag any claims that should transition status
```

Output: Proposed changes document for human review.

### Phase 3: Red Team (red-team-reviewer)

Review the current draft for weaknesses:

```
Tasks:
1. Steelman counterarguments for each ESTABLISHED claim
2. Identify unstated assumptions
3. Check for measurement artifacts
4. Update hostile reviewer playbook
5. Draft §10 (Red Team Analysis) updates
```

Output: Red team review with proposed §10 updates.

### Phase 4: Lead Synthesis

The lead (you) synthesizes all three outputs:

1. Review audit report — fix any blocking errors
2. Review proposed changes — accept, reject, or modify each
3. Review red team findings — incorporate into §10
4. Run `python scripts/build_whitepaper_md.py` to regenerate
5. Run `python scripts/verify_claim_evidence.py --strict` to validate
6. Present final `[PROPOSED CHANGES]` for human approval

## Important Rules

- **Never auto-merge.** All changes require human review.
- **Never weaken epistemic claims.** If new evidence contradicts a claim, update its status — don't delete it.
- **Falsification is an asset.** Moving a claim to FALSIFIED is more valuable than keeping it ESTABLISHED.
- **Cross-repo references are read-only.** Never import code from sibling repos.

## Verification Checklist

After refresh completes:
- [ ] `python scripts/build_whitepaper_md.py --check` passes
- [ ] `python scripts/verify_claim_evidence.py --strict` passes
- [ ] No claim with status `falsified` or `superseded` missing from §06
- [ ] §01 (Epistemic Map) auto-generates correctly
- [ ] All `{{claim:CLM-*}}` references resolve to valid YAMLs

## Usage

```
/whitepaper-refresh
/whitepaper-refresh --audit-only    # Just run the audit, skip synthesis
/whitepaper-refresh --since 2026-02-20  # Only consider evidence after this date
```

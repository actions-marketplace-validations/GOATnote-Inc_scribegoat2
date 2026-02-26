# Evaluation Seed Discovery Team

Spawn an agent team to mine existing evaluation data, OpenEM corpus, and cross-model results to discover candidate evaluation seeds — each with a provenance chain explaining which data surfaced it, what hypothesis it tests, and what finding it could extend or falsify.

## Usage

```
/mine-seeds [--strategy coverage|confusion|failure|asymmetry|pressure|all] [--budget 10]
```

- `--strategy all` (default): spawns all 5 miners in parallel
- `--strategy coverage`: spawns only coverage-miner + scorer + synthesizer
- `--strategy confusion`: spawns only confusion-pair-miner + scorer + synthesizer
- `--strategy failure`: spawns only failure-forensic + scorer + synthesizer
- `--strategy asymmetry`: spawns only asymmetry-detective + scorer + synthesizer
- `--strategy pressure`: spawns only pressure-cartographer + scorer + synthesizer
- `--budget N`: max candidate seeds to produce (default 10). Each miner gets ceil(N/active_miners) budget; scorer selects top N overall.

## Team Structure

### Phase 1: Mining (all run in parallel, no dependencies between miners)

#### 1a. Coverage Miner
- **Agent definition:** `coverage-miner`
- **Model:** Sonnet
- **Spawn prompt:** Include the budget (ceil(N/active_miners)). Remind it to read: OpenEM conditions at `../openem-corpus/corpus/tier1/conditions/`, CONDITION_MAP from `evaluation/bloom_eval_v2/openem_bridge.py`, existing seeds in `evaluation/bloom_eval_v2/scenarios/seeds/`, LostBench scenarios at `../lostbench/scenarios/tier1/` and `../lostbench/scenarios/tier1_adversarial/`
- **Tasks:** Inventory existing coverage, inventory OpenEM corpus, compute coverage matrix, rank gaps, produce candidates
- **Only spawn if** strategy is `all` or `coverage`

#### 1b. Failure Forensic
- **Agent definition:** `failure-forensic`
- **Model:** Sonnet
- **Spawn prompt:** Include the budget. Remind it to read: `experiments/run_log.jsonl` (use `src.utils.jsonl`), LostBench 2x2 results at `../lostbench/results/2x2/` and `../lostbench/results/2x2-adversarial/`, pressure regime labels at `evaluation/bloom_eval_v2/scenarios/pressure_regime_labels.yaml`
- **Tasks:** Analyze experiment history, mine CEIS failure classes, rank pressure effectiveness, identify undertested modes, produce candidates
- **Only spawn if** strategy is `all` or `failure`

#### 1c. Confusion Pair Miner
- **Agent definition:** `confusion-pair-miner`
- **Model:** Sonnet
- **Spawn prompt:** Include the budget. Remind it to grep for `confusion_pairs` in `../openem-corpus/corpus/tier1/conditions/*/condition.yaml`, read both pair members' metadata, check existing seed/scenario coverage
- **Tasks:** Extract pairs, classify by acuity differential, check coverage, design boundary-probing concepts, produce candidates
- **Only spawn if** strategy is `all` or `confusion`

#### 1d. Pressure Cartographer
- **Agent definition:** `pressure-cartographer`
- **Model:** Sonnet
- **Spawn prompt:** Include the budget. Remind it to read: `evaluation/bloom_eval_v2/scenarios/pressure_regime_labels.yaml` (13 pressure types), all `SEED-*.json` files for per-turn pressure_operators, LostBench scenarios at `../lostbench/scenarios/tier1/` for pressure metadata
- **Tasks:** Build pressure inventory, map usage across scenarios, compute co-occurrence/sequence matrices, identify gaps, produce candidates
- **Only spawn if** strategy is `all` or `pressure`

#### 1e. Asymmetry Detective
- **Agent definition:** `asymmetry-detective`
- **Model:** Sonnet
- **Spawn prompt:** Include the budget. Remind it to read: CLAUDE.md findings table (known asymmetries), LostBench 2x2 results at `../lostbench/results/2x2/` and `../lostbench/results/2x2-adversarial/`, any bloom_eval_v2 results with per-model breakdowns
- **Tasks:** Catalog known asymmetries, mine condition-level divergences, identify unexplained asymmetries, propose diagnostic seeds, produce candidates
- **Only spawn if** strategy is `all` or `asymmetry`

### Phase 2: Scoring (blocked until ALL Phase 1 miners complete)

#### 2. Seed Scorer
- **Agent definition:** `seed-scorer`
- **Model:** Haiku (deterministic scoring)
- **Spawn prompt:** Include ALL miner outputs (paste the JSON blocks from each completed miner). Include the final budget N. Remind it to deduplicate, score on 6 dimensions (clinical grounding 2x, novelty 1x, insight density 2x, feasibility 1x, cross-model relevance 1x, safety research value 2x), and rank top N.
- **Tasks:** Collect candidates, deduplicate, score, rank, select top N
- **Blocked until** all Phase 1 miners have completed
- **Always spawned** (required for all strategies)

### Phase 3: Synthesis (blocked until Phase 2 completes)

#### 3. Seed Synthesizer
- **Agent definition:** `seed-synthesizer`
- **Model:** Opus (synthesis requires deeper reasoning)
- **Spawn prompt:** Include the scorer's ranked shortlist AND all original miner outputs (the synthesizer needs both the ranked list and the miner context). Remind it to read existing seeds for format calibration, determine next SEED number, produce complete JSON + rationale pairs, and deliver everything as `[PROPOSED CHANGES]`.
- **Tasks:** Determine next SEED-NNN number, generate complete seed JSON files, generate rationale markdown files, write synthesis report, present as [PROPOSED CHANGES]
- **Blocked until** seed-scorer has completed
- **Require plan approval** before the lead writes any files
- **Always spawned** (required for all strategies)

## Lead Behavior

- Use **delegate mode** (Shift+Tab) — do NOT mine data yourself
- Parse the `--strategy` and `--budget` arguments from the user's command
- Spawn Phase 1 miners in parallel (those matching the strategy)
- Wait for ALL Phase 1 miners to complete before spawning the scorer
- Wait for the scorer to complete before spawning the synthesizer
- After the synthesizer delivers its `[PROPOSED CHANGES]`:
  1. Review each proposed SEED JSON for format correctness (matches SEED-001 schema)
  2. Review each rationale for completeness (has Provenance, Hypothesis, Expected Findings)
  3. Review the synthesis report for insight quality
  4. If approved, write the seed files to `evaluation/bloom_eval_v2/scenarios/seeds/`
  5. If any seeds reference conditions not in CONDITION_MAP, add the entries to `evaluation/bloom_eval_v2/openem_bridge.py`
  6. Present the synthesis report to the user as a message

## Example Spawn

### Full strategy (all miners)
```
Create an agent team to mine evaluation data for new seed scenario ideas.

Strategy: all
Budget: 10 (each miner gets 2 candidate slots)

Teammates:
- "coverage" using coverage-miner agent: Mine OpenEM corpus conditions against existing scenario coverage. Budget: 2 candidates. Read: ../openem-corpus/corpus/tier1/conditions/, evaluation/bloom_eval_v2/openem_bridge.py CONDITION_MAP, evaluation/bloom_eval_v2/scenarios/seeds/SEED-*.json, ../lostbench/scenarios/tier1/, ../lostbench/scenarios/tier1_adversarial/. Output your analysis and candidates as a JSON block.

- "forensic" using failure-forensic agent: Mine run_log.jsonl and CEIS results for recurring failure patterns. Budget: 2 candidates. Read: experiments/run_log.jsonl (use src.utils.jsonl), ../lostbench/results/2x2/, ../lostbench/results/2x2-adversarial/, evaluation/bloom_eval_v2/scenarios/pressure_regime_labels.yaml. Output as JSON.

- "confusion" using confusion-pair-miner agent: Mine OpenEM confusion_pairs for boundary-probing seed concepts. Budget: 2 candidates. Grep for confusion_pairs in ../openem-corpus/corpus/tier1/conditions/*/condition.yaml. Check existing seed and LostBench scenario coverage. Output as JSON.

- "pressure" using pressure-cartographer agent: Map pressure types across scenarios to find untested combinations. Budget: 2 candidates. Read: evaluation/bloom_eval_v2/scenarios/pressure_regime_labels.yaml, all SEED-*.json for pressure_operators, ../lostbench/scenarios/tier1/ for pressure metadata. Output as JSON.

- "asymmetry" using asymmetry-detective agent: Mine GPT-5.2 vs Opus 4.6 divergences from 2x2 results. Budget: 2 candidates. Read: CLAUDE.md findings table, ../lostbench/results/2x2/, ../lostbench/results/2x2-adversarial/. Output as JSON.

Phase 1 miners run in parallel. When ALL complete, spawn:
- "scorer" using seed-scorer agent: Score and rank all miner candidates. Budget: 10. [Paste all miner JSON outputs here]. Deduplicate, score on 6 dimensions, rank top 10.

When scorer completes, spawn:
- "synthesizer" using seed-synthesizer agent: Produce complete SEED JSON + rationale files from ranked shortlist. [Paste scorer output + all miner outputs]. Read existing seeds for format calibration. Present everything as [PROPOSED CHANGES]. I approve before writing files.

Use delegate mode. I coordinate, I don't mine.
```

### Single strategy (coverage only)
```
Create an agent team to mine for coverage gaps.

Strategy: coverage
Budget: 5

Teammates:
- "coverage" using coverage-miner agent: [same as above but Budget: 5]

When coverage miner completes, spawn:
- "scorer" using seed-scorer agent: [same as above but only coverage miner output]

When scorer completes, spawn:
- "synthesizer" using seed-synthesizer agent: [same as above]

Use delegate mode.
```

## After Completion

1. Review the synthesizer's `[PROPOSED CHANGES]` block
2. If approved:
   - Write each `SEED-NNN-condition.json` to `evaluation/bloom_eval_v2/scenarios/seeds/`
   - Write each `SEED-NNN-condition_rationale.md` to `evaluation/bloom_eval_v2/scenarios/seeds/`
   - Add any needed CONDITION_MAP entries to `evaluation/bloom_eval_v2/openem_bridge.py`
   - Add pressure regime labels to `evaluation/bloom_eval_v2/scenarios/pressure_regime_labels.yaml` (if needed)
3. Validate seeds load correctly: `python -c "from evaluation.bloom_eval_v2.scenarios.seed_scenarios import SEED_SCENARIOS; print(f'{len(SEED_SCENARIOS)} seeds loaded')" `
4. Optionally run a single-trial eval on new seeds: `python -m evaluation.bloom_eval_v2 --scenarios seeds --dry-run`
5. Present the synthesis report to the user
6. Do NOT append to `run_log.jsonl` — seed generation is not an evaluation run
7. Commit with: `feat: mine-seeds — [N] new evaluation seeds from [strategy] mining`

## Error Recovery

- **Miner fails to produce JSON:** Lead messages the miner asking it to reformat. If persistent, exclude that miner's output and proceed with remaining miners.
- **Scorer receives no candidates:** This means all miners found no gaps — report "No gaps found" to user. This is a valid outcome.
- **Synthesizer's seeds don't match format:** Lead points to a specific existing seed (e.g., SEED-001-stemi.json) and asks synthesizer to match it exactly. Synthesizer resubmits.
- **Plan rejected by lead:** Tell the synthesizer what to revise. It resubmits a revised `[PROPOSED CHANGES]` block.
- **Budget exceeded:** Scorer enforces the budget — if a miner produces too many candidates, scorer ranks and cuts.
- **Seeds fail to load:** Lead checks the error from `seed_scenarios.py`, identifies the schema mismatch, and fixes the JSON before committing.
- **Cross-repo data missing:** If `../openem-corpus/` or `../lostbench/` directories don't exist, the relevant miners report "DATA_UNAVAILABLE" and are excluded. Scorer proceeds with available candidates.

## What This Team Does NOT Do

- Does NOT run evaluations — it generates seed scenarios for future evaluation
- Does NOT append to `run_log.jsonl` — generation is not evaluation
- Does NOT modify existing seeds or scenarios — only creates new ones
- Does NOT create new tracking files — synthesis report goes in messages
- Does NOT make deployment decisions — it proposes hypotheses for testing

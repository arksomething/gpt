# Paid probe runbook

This is the fail-closed path to the first 25M paid probe. None of the planning,
canonicalization, or preflight commands starts training.

## 1. Finish manual writing review

The current review pack is under `data/review/`. Fill `keep_yes_no` yourself.
Blank rows are exclusions, not implicit approval. The canonical builder accepts
only explicit `yes`, `y`, or `keep` decisions from Tier-A sources marked
redistributable in `data/sources.yaml`.

Build a new, immutable canonical lane:

```bash
uv run canonical-writing \
  --acquisition-manifest data/source_samples/<run>/manifest.json \
  --scores data/review/<run>/scores.csv \
  --output-dir data/canonical/writing-v1
```

The command refuses to publish an empty lane or overwrite an existing one.

## 2. Prepare the indexed probe corpus

Set `data_prep.canonical_writing.manifest` and mixture weights in a dedicated
preparation config, then build:

```bash
uv run prepare-data \
  --config configs/<probe-data-recipe>.yaml \
  --output_format indexed \
  --out_dir data/probes/writing-v1
```

If interrupted, use the same command with `--resume`. The replay must match the
committed source IDs, content hashes, and token sequences. Never use
`--overwrite` merely to bypass a mismatch.

## 3. Rehearse and benchmark the exact workload

First run the existing smoke and correctness suite. On the intended paid GPU,
benchmark the same model, block size, micro-batch, accumulation, precision, and
gradient-checkpointing settings:

```bash
uv run benchmark \
  --model_config configs/model_25m.yaml \
  --train_config configs/train_25m_probe.yaml \
  --output_path runs/throughput-25m.json
```

This allocates and trains random batches briefly, so run it only when ready to
measure that GPU. It is not the real probe. The result is workload-fingerprinted
and an older or mismatched result cannot pass preflight.

## 4. Authorize cost, recovery, and three seeds

In `configs/train_25m_probe.yaml`:

- enter the provider's quoted `budget.hourly_rate`;
- enter the explicit authorized `budget.max_cost`;
- enable checkpoint upload and set the intended repository;
- preserve full exact-resume state and verified uploads.

Then create the immutable group:

```bash
uv run experiment plan \
  --name writing-v1-25m \
  --model-config configs/model_25m.yaml \
  --train-config configs/train_25m_probe.yaml
```

The default seeds are 1337, 2027, and 4099. Any later config edit invalidates
the plan and requires a new group.

## 5. Run the no-training gate

```bash
uv run probe-preflight \
  --model-config configs/model_25m.yaml \
  --train-config configs/train_25m_probe.yaml \
  --experiment-plan runs/experiments/<group-id>/experiment.json
```

It fully verifies both indexed corpora, exact split overlap, tokenizer and
recipe identity, source evaluation, the scheduled token count, total
three-seed throughput/cost, durable observability, checkpoint upload, and the
experiment plan. It
exits nonzero on any blocker, reports that no training started, and only on a
complete pass writes a hash-bound execution receipt beside the plan.

Only after `READY=true`, execute with the exact group ID as the explicit
confirmation:

```bash
uv run experiment execute <group-id> --confirm <group-id>
```

Execution refuses a missing, modified, or stale receipt and rechecks the bound
model, training config, corpora, throughput result, and derived per-seed
configs. That final command is the only command in this runbook that launches
the real probe.

During execution, inspect all seed directories with `uv run run-status`.
Operational details and file formats are in
[`observability.md`](observability.md).

# Agent Instructions

## Project direction

This repository is a language-model training lab progressing through controlled
small-model experiments toward a credible 1B-parameter model.

The active roadmap is [`PLAN.md`](PLAN.md). Read it before proposing an
expensive training run or changing the target architecture, data mixture,
tokenizer, evaluation gates or compute allocation.

The current implementation and generated artifacts are the corrected 100M v3
baseline. They are historical inputs to the new program, not the final target.
The older `llama-100m` run used double-shifted causal labels and is invalid for
quality comparisons.

## Current priorities

1. Gate 0 correctness tests: labels, document masks, gradient accumulation,
   deterministic tiny training, exact resume and HF export equivalence.
2. Indexed, document-aware, source-aware data shards and immutable manifests.
3. Tokenizer and data-mixture bakeoffs using 25M–40M probes.
4. Configurable 25M, 60M, 150M, 350M and 1B model families.
5. Equal-harness evaluation, experiment tracking and scaling forecasts.
6. A useful 350M base/post-trained release before authorizing the 1B run.

Do not add speculative complexity such as MoE, million-token context, native
vision or novel attention kernels to the first cycle unless a small controlled
experiment earns it.

## Key files

| File | Purpose |
|---|---|
| `PLAN.md` | Active goals, research synthesis, stages and go/no-go gates |
| `README.md` | Current repository orientation |
| `ARCHIVE.md` | Historical material and validity labels |
| `scripts/train.py` | Current training implementation and model construction |
| `scripts/prepare_data.py` | Current streaming/filtering/tokenization pipeline |
| `scripts/train_tokenizer.py` | SentencePiece tokenizer training |
| `scripts/eval.py` | Validation and model evaluation |
| `scripts/filters.py` | Text-quality filters |
| `configs/model_100m.yaml` | Corrected historical 100M baseline architecture |
| `configs/train.yaml` | Corrected historical 100M v3 recipe |
| `configs/archive/` | Superseded configuration snapshots |
| `docs/archive/` | Superseded proposals and operating notes |

## Commands

Run project commands through `uv`:

| Command | Description |
|---|---|
| `uv run train` | Train from selected model and training configs |
| `uv run train --smoke` | Short training smoke run |
| `uv run train --launch_screen` | Train in a detached screen session |
| `uv run train --resume_from <path>` | Resume a local checkpoint |
| `uv run train --resume_from_hf <selector>` | Resume a remote checkpoint |
| `uv run train --initialize_from <path>` | Start a new run from validated base weights |
| `uv run eval` | Evaluate a checkpoint |
| `uv run prepare-data` | Prepare tokenized training data |
| `uv run train-tokenizer` | Train a SentencePiece tokenizer |
| `uv run benchmark` | Measure training throughput |
| `uv run infer` | Run model inference |
| `uv run quick-lm-eval` | Run a small lm-eval check |
| `uv run tokenizer-repl` | Inspect tokenizer behavior interactively |
| `uv run scan-bins` | Inspect legacy memmap data files |
| `uv run prepare-chat-data` | Build immutable assistant-supervised chat shards |
| `uv run generate-chat-data` | Generate spend-gated synthetic chat JSONL |
| `uv run chat-review` | Review chat data and export explicit human keeps |
| `uv run conversation-eval` | Generate, score, judge and review frozen chat evals |
| `uv run ls` | Show project information |

Current training flags:

```text
--model_config PATH
--train_config PATH
--resume_from PATH
--resume_from_slot NAME
--resume_from_hf SELECTOR
--initialize_from PATH
--smoke
--launch_screen
--screen_name NAME
```

## Generated data and checkpoints

Generated `data/`, `runs/` and `archive/local/` content is ignored by Git.

- `runs/llama-100m-v3/hf-eval/` is the local corrected 100M comparison export.
- `data/v3/` holds metadata and validation artifacts referenced by that
  baseline. The original training shard is not currently present locally.
- Invalid and superseded local artifacts live under `archive/local/`.
- The active tokenizer is `tokenizer/spm.model`, but it predates the planned
  corpus and must not be assumed to be the final tokenizer.

Do not delete large artifacts merely because they are ignored. Classify them in
`ARCHIVE.md`, preserve invalid-run warnings, and use explicit paths.

## Compute safety

- Do not redeem credits, request paid capacity or launch cloud resources unless
  the user explicitly asks.
- Do not run full training from `configs/train.yaml` without restoring the exact
  matching train shard.
- Remote checkpoint upload is disabled by default. Enable it only for an
  intentional run with a verified destination.
- Benchmark the exact configuration before estimating cost.
- Require explicit stop criteria and budget caps for paid runs.
- Preserve full resume state and artifact manifests for every meaningful run.

## Reproducibility requirements

Every meaningful experiment should record:

- immutable run ID;
- git revision and dirty state;
- model and training config hashes;
- tokenizer and data-manifest hashes;
- total and non-embedding parameter counts;
- tokens processed, throughput, wall time and cost;
- checkpoint and exact-resume status;
- full evaluation configuration and results;
- whether the result is valid, superseded or failed.

Artifact mismatches should fail fast. Historical invalid runs must never be
silently promoted to baselines.

## Dependencies and formats

Dependencies are managed with `uv`. Core packages include PyTorch,
Transformers, Accelerate, SentencePiece, Datasets, NumPy, PyYAML and the
Hugging Face Hub client.

The current `.bin` datasets are `uint16` NumPy memmaps. They are a legacy format
scheduled for replacement because they do not preserve adequate document and
source structure.

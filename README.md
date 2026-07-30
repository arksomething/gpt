# GPT Training Lab

This repository is an experimental pipeline for training language models from
scratch, progressing through small, controlled scaling probes toward a
1B-parameter model.

The current code and artifacts are the **100M v3 baseline**. They are retained
for reproduction and for validating the next pipeline, but they are not the
final target. The active research and execution roadmap is in
[`PLAN.md`](PLAN.md).

## Current status

- The historical 100M Llama-style implementation is operational.
- The previous double-shifted 100M run is invalid and locally archived.
- The corrected 100M v3 HF export is retained as the comparison baseline.
- Data preparation and training can use deterministic, document-aware indexed
  shards with resumable staging, source mixtures, disjoint content-hash splits,
  and exact training resume. Historical flat output remains the default for
  reproduction.
- Reviewed Tier-A writing can enter through a hash-verified canonical lane;
  blank review decisions are never treated as acceptance.
- Paid probes have an immutable three-seed registry, per-source held-out loss,
  and a fail-closed preflight that does not start training.
- Each run has atomic state and heartbeat, append-only metrics/events, stale-run
  inspection, non-replayed runtime commands, and a per-seed runtime cost stop.
- The current tokenizer predates the planned data mixture and must be evaluated
  against new candidates.
- Gate 0 now covers labels, accumulation, deterministic training, one-batch
  overfit, HF reload, exact interrupted resume, and packed-document isolation.
- The next paid training target is not 1B. Small data probes come first.

## Target ladder

| Stage | Scale | Purpose |
|---|---:|---|
| Correctness | 10M–25M | Validate labels, masking, resume and export |
| Data probes | 25M–40M | Select data mixture, filtering and tokenizer |
| Architecture probes | 60M–100M | Test depth, GQA, QK norm and optimizer |
| Scaling anchor | ~150M | Fit loss, capability and cost curves |
| Proof release | ~350M | Release a useful base and post-trained model |
| Main run | ~1B | Train only after all stage gates pass |

See [`PLAN.md`](PLAN.md) for token ranges, compute envelopes, evaluation gates,
the reasoning strategy and research references.

## Repository map

| Path | Purpose |
|---|---|
| `PLAN.md` | Active research and execution plan |
| `scripts/train.py` | Current Transformers/Accelerate training implementation |
| `scripts/prepare_data.py` | Current streaming, filtering and tokenization pipeline |
| `scripts/train_tokenizer.py` | SentencePiece tokenizer training |
| `scripts/eval.py` | Validation and generation evaluation |
| `scripts/filters.py` | Text-quality filters |
| `configs/model_100m.yaml` | Historical corrected 100M baseline architecture |
| `configs/model_{25m,60m,150m,350m,1b}.yaml` | Validated staged model ladder |
| `configs/train.yaml` | Historical 100M v3 training recipe, retained as the current executable default |
| `scripts/indexed_shards.py` | Immutable document index, token shards, lineage, and integrity checks |
| `scripts/canonical_writing.py` | Human-acceptance gate for reviewed writing |
| `scripts/experiment_registry.py` | Immutable multi-seed plans and append-only events |
| `scripts/preflight_probe.py` | No-training validation for paid probes |
| `scripts/run_observer.py` | Durable run state, metrics, events, health and budget control |
| `scripts/prepare_chat_data.py` | Immutable assistant-supervised chat corpus preparation |
| `scripts/generate_chat_data.py` | Spend-gated, provenance-rich Fireworks chat synthesis |
| `scripts/chat_review.py` | Human conversation-data review and accepted-data export |
| `scripts/conversation_eval.py` | Frozen multi-turn generation, scoring, judging, and blind review |
| `docs/indexed-data.md` | Indexed sampling, packing, configuration, and resumable production |
| `docs/probe-runbook.md` | Exact path from manual review to an authorized probe |
| `docs/observability.md` | Run files, health checks, budget behavior and controls |
| `docs/conversation-posttraining.md` | Conversational SFT, evaluation, judging, and advancement gates |
| `scripts/validate_model_ladder.py` | Meta-device parameter and architecture validator |
| `configs/archive/` | Older configuration snapshots |
| `docs/archive/` | Superseded proposals and operating documents |
| `ARCHIVE.md` | Inventory and interpretation of archived material |

Generated datasets and checkpoints remain ignored by Git.

## Setup

```bash
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

All project commands are exposed through `uv`:

```bash
uv run train --help
uv run prepare-data --help
uv run train-tokenizer --help
uv run eval --help
uv run benchmark --help
uv run infer --help
uv run quick-lm-eval --help
uv run validate-model-ladder
uv run canonical-writing --help
uv run experiment --help
uv run probe-preflight --help
uv run run-status --help
uv run prepare-chat-data --help
uv run generate-chat-data --help
uv run chat-review --help
uv run conversation-eval --help
uv run pytest -q
```

## Historical baseline

The corrected baseline uses:

- approximately 102M parameters;
- 12 layers, hidden size 768 and 12 attention heads;
- RMSNorm, RoPE, SwiGLU and tied embeddings;
- 32k SentencePiece vocabulary;
- 2,048-token contexts;
- the archived 2B-token C4/Wikipedia/FineWeb recipe.

The local baseline export is expected under
`runs/llama-100m-v3/hf-eval/`. The original training shard is not currently
present locally, so a default full training launch is not ready to run.

Do not treat the older `runs/llama-100m` checkpoints as valid: they were trained
with labels shifted twice. Their files are preserved under the ignored local
archive for forensic reference only.

## Safety before compute

Do not launch a long or cloud-backed run until:

1. Gate 0 tests in `PLAN.md` pass;
2. the exact dataset and tokenizer fingerprints are present;
3. checkpoint upload and resume have completed a tiny rehearsal;
4. provider quota, pricing and automatic shutdown are confirmed;
5. the run has an explicit dollar limit and stop criteria.

`configs/train.yaml` still describes the historical 100M v3 run. Its remote
checkpoint upload is disabled by default to prevent accidental writes to the old
repository.

## Historical documentation

- [`docs/archive/2026-course-project-proposal.md`](docs/archive/2026-course-project-proposal.md)
  — original 100M GPT-2-versus-Llama class proposal.
- [`docs/archive/2026-100m-v3-readme.md`](docs/archive/2026-100m-v3-readme.md)
  — original 100M v3 operating README.

## Data and model licensing

Every future corpus manifest must record source licenses and redistribution
constraints. Existing references to C4, Wikipedia and FineWeb do not imply that
all future data mixtures or resulting artifacts are cleared for every
commercial use.

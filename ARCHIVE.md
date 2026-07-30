# Archive Inventory

This file separates historical evidence from active project inputs.

## Tracked archive

| Path | Status | Reason retained |
|---|---|---|
| `docs/archive/2026-course-project-proposal.md` | Superseded | Original 100M GPT-2-versus-Llama class proposal |
| `docs/archive/2026-100m-v3-readme.md` | Superseded | Operating instructions for the corrected 100M v3 experiment |
| `configs/archive/train_20260228_pre_v2.yaml` | Invalid historical recipe | Predates the causal-label correction and data v2 transition |

## Local ignored archive

Large generated artifacts are stored under `archive/local/` and ignored by Git.
They are recoverable locally but are not active inputs.

### Runs

- `archive/local/runs/llama-100m/` — invalid double-shift training run.
- `archive/local/runs/step_0000200/` — early standalone checkpoint.
- `archive/local/runs/llama-100m.tgz` — historical run bundle.
- Other root-level 100M logs, samples and comparison reports were moved beside
  those runs.

The corrected comparison baseline remains active at
`runs/llama-100m-v3/hf-eval/`. Public evaluation baselines remain under
`runs/lm-eval-baselines/`.

The v3 artifact manifest identifies Git commit
`5c4b462394dab690c3c6f1bbbaaec3d229515da7` as the exact training source. Use
that revision when reproducing its original `configs/train.yaml`; the active
copy now disables remote uploads and is intentionally not byte-identical.

### Data

The pre-v3 root memmaps, v2 shards, one-off benchmark datasets, review exports
and preparation logs are under `archive/local/data/`.

Only `data/v3/` remains in the active generated-data namespace because the
historical corrected baseline references it. Its original training shard is not
present locally; the directory currently contains metadata and validation data.

## Interpretation rules

- Archived checkpoints must never be selected automatically.
- The invalid double-shift run must not appear in quality comparisons.
- Historical configs and reports may be used to reproduce or diagnose prior
  behavior, not to define new runs.
- New experiments should receive immutable run IDs, manifests and explicit
  status labels from the beginning.

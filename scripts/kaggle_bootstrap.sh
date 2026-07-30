#!/usr/bin/env bash
# Kaggle-adapted bootstrap: paste into a single Kaggle notebook cell as
#   !bash <(curl -fsSL https://raw.githubusercontent.com/arksomething/gpt/main/scripts/kaggle_bootstrap.sh)
# or clone first and run bash scripts/kaggle_bootstrap.sh.
#
# Differences from bootstrap_remote.sh: no sudo, no shutdown scheduling
# (Kaggle sessions are already time-capped), checkpoints exported to
# /kaggle/working so they survive the session, and a tiny fresh corpus is
# generated for the smoke run since data bins are not in git.
#
# Env:
#   SMOKE=1 (default) run the 25M smoke train after verification; SMOKE=0 verify only
#   SMOKE_TOKENS   tiny corpus size for the smoke run (default 5,000,000)

set -euo pipefail
REPO_URL="${REPO_URL:-https://github.com/arksomething/gpt.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
WORKDIR="${WORKDIR:-/tmp/gpt}"  # NOT /kaggle/working: that dir persists as output and a venv there bloats the bundle to GBs
SMOKE="${SMOKE:-1}"
SMOKE_TOKENS="${SMOKE_TOKENS:-5000000}"

log() { printf '\n[kaggle-bootstrap] %s\n' "$*"; }
die() { printf '\n[kaggle-bootstrap] FATAL: %s\n' "$*" >&2; exit 1; }

log "start $(date -u +%FT%TZ)"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
  || log "WARNING: no GPU visible - enable an accelerator in notebook settings"

# --- toolchain (Kaggle images ship git; uv usually missing)
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

# --- clone
if [[ -d "$WORKDIR/.git" ]]; then
  git -C "$WORKDIR" fetch origin "$REPO_BRANCH" && git -C "$WORKDIR" reset --hard "origin/$REPO_BRANCH"
else
  git clone --branch "$REPO_BRANCH" "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"
log "revision: $(git rev-parse --short HEAD)"

# --- env + fingerprints + Gate 0
uv sync --frozen || die "uv sync failed"
EXPECTED="d879422cadb5566190960da5fbe18b2163cb79df4bad192474c42bdb449a861f"
[[ "$(sha256sum tokenizer/spm.model | cut -d' ' -f1)" == "$EXPECTED" ]] || die "tokenizer fingerprint mismatch"
log "running Gate 0 suite"
uv run --with pytest pytest tests/ -q || die "Gate 0 failed"
log "VERIFIED: clone, env, fingerprints, Gate 0 all green."

# --- smoke run on a tiny freshly-generated corpus
if [[ "$SMOKE" == "1" ]]; then
  log "preparing tiny smoke corpus (${SMOKE_TOKENS} tokens, streamed, indexed)"
  uv run python scripts/prepare_data.py \
    --tokenizer_model tokenizer/spm.model \
    --out_dir data/probes/writing-v1 \
    --output_format indexed \
    --train_tokens "$SMOKE_TOKENS" --val_tokens 100000 \
    --tokenizer_workers 2 --overwrite || die "smoke data prep failed"

  log "launching 25M smoke training run"
  uv run python scripts/train.py \
    --model_config configs/model_25m.yaml \
    --train_config configs/train_25m_probe.yaml \
    --smoke || die "smoke run failed"

  log "exporting run artifacts to /kaggle/working"
  mkdir -p /kaggle/working/artifacts
  cp -r runs/* /kaggle/working/artifacts/ 2>/dev/null || log "no runs/ output found to export"
fi
log "done $(date -u +%FT%TZ)"

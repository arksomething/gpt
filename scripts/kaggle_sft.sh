#!/usr/bin/env bash
# SFT rehearsal on Kaggle: 25M base + Dolly chat corpus (attached dataset)
# -> assistant-masked SFT -> chat samples -> artifacts to /kaggle/working.
set -euo pipefail
REPO_URL="${REPO_URL:-https://github.com/arksomething/gpt.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
WORKDIR="${WORKDIR:-/kaggle/working/gpt}"
DS="${DS:-/kaggle/input/gpt-25m-base-and-chat-v1}"

log() { printf '\n[kaggle-sft] %s\n' "$*"; }
die() { printf '\n[kaggle-sft] FATAL: %s\n' "$*" >&2; exit 1; }

log "start $(date -u +%FT%TZ)"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

if [[ -d "$WORKDIR/.git" ]]; then
  git -C "$WORKDIR" fetch origin "$REPO_BRANCH" && git -C "$WORKDIR" reset --hard "origin/$REPO_BRANCH"
else
  git clone --branch "$REPO_BRANCH" "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"
log "revision: $(git rev-parse --short HEAD)"
uv sync --frozen || die "uv sync failed"

# --- stage inputs from the attached dataset (handles zipped or extracted)
mkdir -p runs/probes/25m-base/final data/chat
if [[ -f "$DS/base/final/model.pt" ]]; then
  cp "$DS/base/final/model.pt" runs/probes/25m-base/final/
  cp "$DS/base/artifacts_manifest.json" runs/probes/25m-base/
elif [[ -f "$DS/base.zip" ]]; then
  unzip -q "$DS/base.zip" -d /tmp/base_ds
  cp /tmp/base_ds/final/model.pt runs/probes/25m-base/final/
  cp /tmp/base_ds/artifacts_manifest.json runs/probes/25m-base/
else
  die "base checkpoint not found in dataset at $DS"
fi
if [[ -d "$DS/chat/v1" ]]; then
  cp -r "$DS/chat/v1" data/chat/
elif [[ -f "$DS/chat.zip" ]]; then
  unzip -q "$DS/chat.zip" -d data/chat/
else
  die "chat corpus not found in dataset at $DS"
fi
[[ -d data/chat/v1/train ]] || die "chat corpus layout unexpected"

# --- fingerprints + Gate 0
EXPECTED="d879422cadb5566190960da5fbe18b2163cb79df4bad192474c42bdb449a861f"
[[ "$(sha256sum tokenizer/spm.model | cut -d' ' -f1)" == "$EXPECTED" ]] || die "tokenizer fingerprint mismatch"
log "running Gate 0 suite"
uv run --with pytest pytest tests/ -q || die "Gate 0 failed"

# --- SFT
log "starting SFT (120 steps, assistant-masked, fp16 for T4)"
uv run python scripts/train.py \
  --model_config configs/model_25m.yaml \
  --train_config configs/train_25m_chat_sft.yaml || die "SFT failed"

# --- chat-formatted samples from the tuned model
for q in "What is the capital of France?" "Explain what a computer does, simply." "Give me three tips for better sleep."; do
  P="<|user|>${q}<|end|><|assistant|>"
  uv run python scripts/infer.py \
    --model_config configs/model_25m.yaml \
    --checkpoint runs/sft/25m-dolly-rehearsal/final \
    --tokenizer tokenizer/spm.model \
    --prompt "$P" --max_tokens 80 --temperature 0.7 \
    --skip_artifact_validation 2>/dev/null | tail -6 || true
  echo "======"
done

log "exporting artifacts"
mkdir -p /kaggle/working/artifacts
cp -r runs/sft /kaggle/working/artifacts/ 2>/dev/null || true
log "done $(date -u +%FT%TZ)"

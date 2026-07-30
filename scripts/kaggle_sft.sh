#!/usr/bin/env bash
# SFT rehearsal on Kaggle: 25M base + Dolly chat corpus (attached dataset)
# -> assistant-masked SFT -> chat samples -> artifacts to /kaggle/working.
set -euo pipefail
REPO_URL="${REPO_URL:-https://github.com/arksomething/gpt.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
WORKDIR="${WORKDIR:-/tmp/gpt}"  # NOT /kaggle/working: that dir persists as output and a venv there bloats the bundle to GBs
DS="${DS:-/kaggle/input/gpt-25m-base-and-chat-v1}"

log() { printf '\n[kaggle-sft] %s\n' "$*"; }
die() { printf '\n[kaggle-sft] FATAL: %s\n' "$*" >&2; exit 1; }

# Mirror everything into the persisted output so a failed run is always
# diagnosable from the small artifact bundle (notebook `!` swallows exit
# codes and the API does not expose logs cleanly).
mkdir -p /kaggle/working
exec > >(tee -a /kaggle/working/sft_run.log) 2>&1

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

# --- stage inputs by discovery: Kaggle mount paths vary by slug/version.
log "input mounts:"; ls -la /kaggle/input/ || true
mkdir -p runs/probes/25m-base/final data/chat
MODEL_PT=$(find /kaggle/input -name model.pt -path '*base*' 2>/dev/null | head -1)
MANIFEST=$(find /kaggle/input -name artifacts_manifest.json 2>/dev/null | head -1)
CHAT_MANIFEST=$(find /kaggle/input -name chat_manifest.json 2>/dev/null | head -1)
[[ -n "$MODEL_PT" && -n "$MANIFEST" ]] || die "base checkpoint not found anywhere under /kaggle/input"
[[ -n "$CHAT_MANIFEST" ]] || die "chat corpus not found anywhere under /kaggle/input"
cp "$MODEL_PT" runs/probes/25m-base/final/
cp "$MANIFEST" runs/probes/25m-base/
cp -r "$(dirname "$CHAT_MANIFEST")" data/chat/v1
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

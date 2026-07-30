#!/usr/bin/env bash
# Bootstrap a fresh remote box (Azure/AWS VM, or any Linux host) for this repo.
#
# Fail-closed: every step must succeed before anything expensive can start.
# The script never launches training itself unless RUN_CMD is provided, and
# even then only after Gate 0 passes.
#
# Usage (on the remote box):
#   curl -fsSL https://raw.githubusercontent.com/arksomething/gpt/main/scripts/bootstrap_remote.sh | bash
# or with a training command queued after verification:
#   RUN_CMD="uv run python scripts/train.py --model_config configs/model_25m.yaml --smoke" \
#     bash scripts/bootstrap_remote.sh
#
# Environment:
#   REPO_URL      git repo to clone (default: https://github.com/arksomething/gpt.git)
#   REPO_BRANCH   branch to check out (default: main)
#   WORKDIR       where to clone (default: ~/gpt)
#   HF_TOKEN      HuggingFace token for artifact pulls (optional; needed for private repos)
#   RUN_CMD       command to launch inside tmux after verification (optional)
#   MAX_LIFETIME_HOURS  hard kill-switch: schedule shutdown N hours out (default: 12)
#   SKIP_SHUTDOWN if set, do not schedule the auto-shutdown (local testing)

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/arksomething/gpt.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
WORKDIR="${WORKDIR:-$HOME/gpt}"
MAX_LIFETIME_HOURS="${MAX_LIFETIME_HOURS:-12}"

log()  { printf '\n[bootstrap] %s\n' "$*"; }
die()  { printf '\n[bootstrap] FATAL: %s\n' "$*" >&2; exit 1; }

log "start $(date -u +%FT%TZ) on $(hostname) ($(uname -m))"

# --- 0. Kill switch first: a wedged bootstrap must not burn money silently.
if [[ -z "${SKIP_SHUTDOWN:-}" ]]; then
  if command -v shutdown >/dev/null 2>&1; then
    sudo shutdown -h "+$((MAX_LIFETIME_HOURS * 60))" 2>/dev/null \
      && log "auto-shutdown armed: +${MAX_LIFETIME_HOURS}h (cancel: sudo shutdown -c)" \
      || log "WARNING: could not arm auto-shutdown (no sudo?); watch this box manually"
  fi
fi

# --- 1. Base tooling.
if ! command -v git >/dev/null 2>&1; then
  log "installing git"
  sudo apt-get update -qq && sudo apt-get install -y -qq git || die "git install failed"
fi
if ! command -v tmux >/dev/null 2>&1; then
  sudo apt-get install -y -qq tmux || log "WARNING: tmux unavailable; RUN_CMD will run in foreground"
fi
if ! command -v uv >/dev/null 2>&1; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh || die "uv install failed"
  export PATH="$HOME/.local/bin:$PATH"
fi

# --- 2. Clone or update the repo.
if [[ -d "$WORKDIR/.git" ]]; then
  log "updating existing clone at $WORKDIR"
  git -C "$WORKDIR" fetch origin "$REPO_BRANCH" && git -C "$WORKDIR" checkout "$REPO_BRANCH" \
    && git -C "$WORKDIR" reset --hard "origin/$REPO_BRANCH"
else
  log "cloning $REPO_URL ($REPO_BRANCH) -> $WORKDIR"
  git clone --branch "$REPO_BRANCH" "$REPO_URL" "$WORKDIR" || die "clone failed"
fi
cd "$WORKDIR"
log "revision: $(git rev-parse --short HEAD)"

# --- 3. Environment.
log "uv sync"
uv sync --frozen || die "uv sync failed (lockfile mismatch?)"

# --- 4. GPU visibility (informational; CPU boxes are fine for rehearsal).
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
else
  log "no NVIDIA GPU visible (CPU-only box)"
fi

# --- 5. Artifact fingerprints. The v3 val.bin incident is why this is fatal.
log "verifying tokenizer fingerprint"
EXPECTED_TOKENIZER_SHA="d879422cadb5566190960da5fbe18b2163cb79df4bad192474c42bdb449a861f"
ACTUAL_SHA=$(sha256sum tokenizer/spm.model | cut -d' ' -f1)
[[ "$ACTUAL_SHA" == "$EXPECTED_TOKENIZER_SHA" ]] \
  || die "tokenizer sha mismatch: $ACTUAL_SHA (expected $EXPECTED_TOKENIZER_SHA)"

# --- 6. Gate 0: full test suite, including bin readability checks.
log "running Gate 0 test suite"
uv run --with pytest pytest tests/ -q || die "Gate 0 tests failed; refusing to continue"

log "VERIFIED: repo, environment, fingerprints, and Gate 0 are green."

# --- 7. Optional launch under tmux with logging.
if [[ -n "${RUN_CMD:-}" ]]; then
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  LOGFILE="$WORKDIR/runs/bootstrap_${STAMP}.log"
  mkdir -p "$WORKDIR/runs"
  if command -v tmux >/dev/null 2>&1; then
    log "launching in tmux session 'run': $RUN_CMD"
    tmux new-session -d -s run "cd '$WORKDIR' && $RUN_CMD 2>&1 | tee '$LOGFILE'"
    log "attach with: tmux attach -t run ; log: $LOGFILE"
  else
    log "tmux missing; running in foreground"
    bash -c "cd '$WORKDIR' && $RUN_CMD 2>&1 | tee '$LOGFILE'"
  fi
else
  log "no RUN_CMD given; box is verified and idle. Auto-shutdown remains armed."
fi

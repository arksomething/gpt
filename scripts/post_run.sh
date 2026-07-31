#!/usr/bin/env bash
# Continue an arm on the box that trained it: export -> bench -> ship.
#
# The checkpoint, the corpus, the repo and the GPU are already here. Pulling a
# checkpoint down and pushing it back up to bench it elsewhere costs bandwidth
# and wall-clock and buys nothing, so everything downstream of training runs in
# place and only the results leave.
#
# Usage (on the box):
#   ARM=b0-seed1 SHIP_URL='https://...' bash scripts/post_run.sh
#
# Never terminates the instance. Results are verified off-box before anything
# is destroyed -- that ordering is why the first bench run was recoverable.

set -u
ARM="${ARM:?ARM must be set}"
SHIP_URL="${SHIP_URL:-}"
RUNDIR="/root/gpt/runs/gate1/$ARM"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model_25m.yaml}"
BENCH_LIMIT="${BENCH_LIMIT:-200}"
cd /root/gpt

log() { echo "[post_run:$ARM] $*"; }

ship() {
  [ -z "$SHIP_URL" ] && return 0
  tar -cf /tmp/ship.tmp -C /root/gpt/runs/gate1 "$ARM" 2>/dev/null || return 1
  mv -f /tmp/ship.tmp /tmp/ship.tar
  curl -fsS -X PUT --upload-file /tmp/ship.tar "$SHIP_URL" && log "shipped"
}

CKPT="$RUNDIR/final"
if [ ! -d "$CKPT" ]; then
  log "no final/ checkpoint at $CKPT -- refusing to continue"
  exit 1
fi

# --- Export. transformers 5.x silently breaks the raw SentencePiece tokenizer
# (vocab_size collapses to 3 and everything encodes to [BOS]), which produced a
# fake wikitext perplexity of ~1.0 once already. Pin 4.49 for anything that
# tokenizes text.
EXPORT_DIR="$RUNDIR/hf"
log "exporting to $EXPORT_DIR"
uv run --with 'transformers==4.49.0' --with protobuf --with sentencepiece \
  python scripts/export_hf.py \
    --checkpoint "$CKPT" \
    --model_config "$MODEL_CONFIG" \
    --tokenizer tokenizer/spm.model \
    --out "$EXPORT_DIR" 2>&1 | tail -20
EXPORT_RC=${PIPESTATUS[0]}
echo "$EXPORT_RC" > "$RUNDIR/EXPORT_RC"
if [ "$EXPORT_RC" != "0" ]; then
  log "export FAILED rc=$EXPORT_RC -- skipping bench"
  ship
  exit "$EXPORT_RC"
fi
ship

# --- Bench. Same harness and limit for every arm, so numbers are comparable
# across arms and against the earlier 25M runs.
log "benching (limit=$BENCH_LIMIT)"
uv run --with 'transformers==4.49.0' --with protobuf --with sentencepiece \
        --with lm-eval --with datasets \
  python scripts/quick_lm_eval.py \
    --models local \
    --local_pretrained "$EXPORT_DIR" \
    --local_tokenizer "$EXPORT_DIR" \
    --limit "$BENCH_LIMIT" \
    --device cuda \
    --output_root "$RUNDIR/bench" 2>&1 | tail -30
BENCH_RC=${PIPESTATUS[0]}
echo "$BENCH_RC" > "$RUNDIR/BENCH_RC"
log "bench rc=$BENCH_RC"

echo "$(date -u +%FT%TZ)" > "$RUNDIR/POST_RUN_DONE"
ship
log "POST_RUN_COMPLETE export=$EXPORT_RC bench=$BENCH_RC"

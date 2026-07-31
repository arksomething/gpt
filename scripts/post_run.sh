#!/usr/bin/env bash
# Everything downstream of pretraining, on the box that trained it:
#   export -> bench -> SFT (template v2) -> export -> samples -> ship
#
# The checkpoint, corpus, repo and GPU are already here, so nothing is moved
# off-box except results. Every stage records its own rc and ships immediately,
# so a stage that fails still leaves the earlier stages' artifacts recoverable.
#
# Usage:  ARM=b0-seed1 SHIP_URL=... SFT_URL=... bash scripts/post_run.sh

set -u
ARM="${ARM:?ARM must be set}"
SHIP_URL="${SHIP_URL:-}"
SFT_URL="${SFT_URL:-}"
RUNDIR="/root/gpt/runs/gate1/$ARM"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model_25m.yaml}"
BENCH_LIMIT="${BENCH_LIMIT:-200}"
PIN="--with transformers==4.49.0 --with protobuf --with sentencepiece"
cd /root/gpt

log() { echo "[post_run:$ARM] $(date -u +%H:%M:%S) $*"; }

ship() {
  [ -z "$SHIP_URL" ] && return 0
  tar --warning=no-file-changed -cf /tmp/pr.tmp -C /root/gpt/runs/gate1 "$ARM" 2>/dev/null
  rc=$?
  [ "$rc" -le 1 ] && [ -s /tmp/pr.tmp ] || { log "ship tar rc=$rc"; return 1; }
  mv -f /tmp/pr.tmp /tmp/pr.tar
  curl -fsS -X PUT --upload-file /tmp/pr.tar "$SHIP_URL" >/dev/null && log "shipped"
}

CKPT="$RUNDIR/final"
[ -d "$CKPT" ] || { log "no final/ checkpoint -- aborting"; exit 1; }

# --- 1. Export the base model. transformers 5.x silently breaks the raw
# SentencePiece tokenizer and already produced a fake wikitext ppl of ~1.0.
log "export base"
uv run $PIN python scripts/export_hf.py --checkpoint "$CKPT" \
  --model_config "$MODEL_CONFIG" --tokenizer tokenizer/spm.model \
  --out "$RUNDIR/hf-base" > "$RUNDIR/export_base.log" 2>&1
echo $? > "$RUNDIR/RC_EXPORT_BASE"; ship

# --- 2. Bench the base model, same harness and limit for every arm.
log "bench base"
uv run $PIN --with lm-eval --with datasets python scripts/quick_lm_eval.py \
  --models local --local_pretrained "$RUNDIR/hf-base" \
  --local_tokenizer "$RUNDIR/hf-base" --limit "$BENCH_LIMIT" --device cuda \
  --output_root "$RUNDIR/bench-base" > "$RUNDIR/bench_base.log" 2>&1
echo $? > "$RUNDIR/RC_BENCH_BASE"; ship

# --- 3. SFT from the pretrained weights. New optimizer, new output dir: the
# base checkpoint is never written over.
if [ -n "$SFT_URL" ]; then
  log "fetch sft corpus"
  mkdir -p /root/gpt/data/chat/v2
  curl -fsSL -o /tmp/sft.tar "$SFT_URL" && \
    tar -xf /tmp/sft.tar -C /root/gpt/data/chat/v2 && rm -f /tmp/sft.tar
fi

if [ -d /root/gpt/data/chat/v2/indexed/train ]; then
  log "sft"
  SFTDIR="$RUNDIR/sft"
  python3 - "$SFTDIR" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open("configs/train_25m_sft_v2.yaml"))
cfg["training"]["output_dir"] = sys.argv[1]
cfg["budget"]["throughput_path"] = sys.argv[1] + "/throughput.json"
yaml.safe_dump(cfg, open("/tmp/sft_arm.yaml", "w"), sort_keys=False)
PY
  uv run train --train_config /tmp/sft_arm.yaml --model_config "$MODEL_CONFIG" \
    --initialize_from "$CKPT" > "$RUNDIR/sft.log" 2>&1
  echo $? > "$RUNDIR/RC_SFT"; ship

  if [ -d "$SFTDIR/final" ]; then
    log "export sft"
    uv run $PIN python scripts/export_hf.py --checkpoint "$SFTDIR/final" \
      --model_config "$MODEL_CONFIG" --tokenizer tokenizer/spm.model \
      --out "$RUNDIR/hf-chat" > "$RUNDIR/export_chat.log" 2>&1
    echo $? > "$RUNDIR/RC_EXPORT_CHAT"; ship

    # --- 4. Readable samples. This is the artifact a human actually judges;
    # loss and perplexity never revealed that round 1 answered the wrong
    # question.
    log "samples"
    uv run python scripts/conversation_eval.py generate \
      --checkpoint "$SFTDIR/final" --model_config "$MODEL_CONFIG" \
      --tokenizer tokenizer/spm.model \
      --output "$RUNDIR/samples.jsonl" > "$RUNDIR/samples.log" 2>&1
    echo $? > "$RUNDIR/RC_SAMPLES"; ship
  fi
fi

echo "$(date -u +%FT%TZ)" > "$RUNDIR/POST_RUN_DONE"
ship
log "POST_RUN_COMPLETE"

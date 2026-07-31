#!/usr/bin/env bash
# Run as root on a training box: refresh the repo and start post_run.
#
# Idempotent on purpose. The orchestrator restarts lose their in-memory record
# of which arms it already kicked, and two concurrent post_run processes would
# write the same directories and corrupt each other's artifacts.
set -u
ARM="$1"; SHIP_URL="$2"; SFT_URL="$3"; REPORT_URL="${4:-}"

if pgrep -f "post_run.sh" >/dev/null 2>&1; then
  echo "ALREADY_RUNNING $ARM"; exit 0
fi
if [ -f "/root/gpt/runs/gate1/$ARM/POST_RUN_DONE" ]; then
  echo "ALREADY_DONE $ARM"; exit 0
fi

cd /root/gpt || exit 1
git fetch -q origin main 2>/dev/null && git reset -q --hard origin/main 2>/dev/null
export ARM SHIP_URL SFT_URL REPORT_URL
setsid nohup bash scripts/post_run.sh > /var/log/post_run.log 2>&1 < /dev/null &
echo "KICKED $ARM"

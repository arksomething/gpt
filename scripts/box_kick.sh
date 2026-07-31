#!/usr/bin/env bash
# Run as root on the box: refresh the repo and start the post-run pipeline.
set -u
ARM="$1"; SHIP_URL="$2"; SFT_URL="$3"
cd /root/gpt || exit 1
git pull -q origin main 2>/dev/null || git fetch -q origin main && git reset -q --hard origin/main
export ARM SHIP_URL SFT_URL
setsid nohup bash scripts/post_run.sh > /var/log/post_run.log 2>&1 < /dev/null &
echo "KICKED $ARM"

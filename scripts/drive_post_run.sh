#!/usr/bin/env bash
# Watch the running arms and, as each finishes training, start its export/bench
# on the box that trained it.
#
# Boxes already launched are running the old runner, which stops after
# training, so the continuation is pushed to them over SSH. Future launches
# chain post_run.sh directly and will not need this.
#
# Deliberately does not terminate anything: results are copied off and verified
# locally first.

set -u
cd /home/ark296/projects/gpt
BUCKET="${BUCKET:-gpt-gate1-202518310973}"
KEY="${KEY:-$HOME/.ssh/gpt-aws.pem}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=12"
STARTED=""

remote() { timeout 90 ssh $SSH_OPTS -i "$KEY" ubuntu@"$1" "$2" 2>/dev/null; }

for i in $(seq 1 300); do
  ROWS=$(aws ec2 describe-instances \
    --filters "Name=tag:Project,Values=gate1" "Name=instance-state-name,Values=running" \
    --query "Reservations[].Instances[].[Tags[?Key=='Name']|[0].Value,PublicIpAddress]" \
    --output text 2>/dev/null)
  if [ -z "$ROWS" ]; then echo "NO_RUNNING_INSTANCES"; sleep 120; continue; fi

  while read -r NAME IP; do
    [ -z "${IP:-}" ] && continue
    ARM="${NAME#gate1-}"
    case " $STARTED " in *" $ARM "*) continue;; esac

    # ARM_DONE is written by the runner only after training returns, and holds
    # the trainer's exit code.
    RC=$(remote "$IP" "sudo cat /root/gpt/runs/gate1/$ARM/ARM_DONE 2>/dev/null")
    [ -z "$RC" ] && continue

    if [ "$RC" != "0" ]; then
      echo "ARM_TRAIN_FAILED $ARM rc=$RC ip=$IP (skipping post-run)"
      STARTED="$STARTED $ARM"
      continue
    fi

    SHIP_URL=$(uv run --with boto3 --with 'botocore[crt]' python -c "
import sys; sys.path.insert(0,'.')
from scripts.launch_fleet import presign
print(presign('$BUCKET','results/$ARM.tar',43200,'put'))" 2>/dev/null | tail -1)

    echo "POST_RUN_START $ARM ip=$IP"
    remote "$IP" "cd /root/gpt && sudo git pull -q origin main && \
      sudo -E env ARM=$ARM SHIP_URL='$SHIP_URL' \
      nohup bash scripts/post_run.sh > /var/log/post_run_$ARM.log 2>&1 & echo launched"
    STARTED="$STARTED $ARM"
  done <<< "$ROWS"

  sleep 120
done
echo "DRIVE_POST_RUN_ENDED started:$STARTED"

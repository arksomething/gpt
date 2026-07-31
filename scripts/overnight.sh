#!/usr/bin/env bash
# Unattended Gate 1 driver.
#
# Loop, until every arm is done or the deadline hits:
#   finished training -> run post_run on-box (export/bench/SFT/samples)
#   post_run done     -> collect tarball, verify locally, TERMINATE the box
#   free quota        -> launch the next arm
#
# Ordering is deliberate. Results are verified on this machine before any box
# is destroyed, because these instances delete their disks on termination and
# a bench run was lost that way once already.
#
# Cost control: every box carries its own shutdown kill-switch, and this loop
# terminates the entire fleet at DEADLINE_UTC no matter what state it is in.

set -u
cd /home/ark296/projects/gpt
BUCKET=gpt-gate1-202518310973
KEY=$HOME/.ssh/gpt-aws.pem
# -n is load-bearing: ssh reads stdin, and inside the `while read` loop below
# it swallows the remaining fleet rows. Without it the loop only ever processed
# the first instance and silently ignored every arm that finished.
SSH="ssh -n -o StrictHostKeyChecking=no -o ConnectTimeout=12 -i $KEY"
SG=sg-09ab4b5c96796e292
SUBNETS="subnet-0d65a3210c0caee6c subnet-0d7d3ddad08b7f486 subnet-0399353261423516e subnet-06ea31b3cc0b5ef21 subnet-0c7178288a8c1e07f"
# ${VAR:-default} substitutes when VAR is empty OR unset, so a caller passing
# an explicitly empty queue ("everything is already done") silently got the
# full default back and relaunched a finished arm. ${VAR-default} only
# substitutes when unset, which is the intended distinction.
QUEUE="${QUEUE-p2 p3 p4 p5}"
DEADLINE_UTC="${DEADLINE_UTC:?set DEADLINE_UTC epoch seconds}"
RESULTS=results/collected
mkdir -p "$RESULTS" logs

started_post=""
collected=""

presign() {  # arm, method
  uv run --with boto3 --with 'botocore[crt]' python -c "
import sys; sys.path.insert(0,'.')
from scripts.launch_fleet import presign
print(presign('$BUCKET','$1',43200,'$2'))" 2>/dev/null </dev/null | tail -1
}

fleet() {
  aws ec2 describe-instances \
    --filters "Name=tag:Project,Values=gate1" "Name=instance-state-name,Values=running" \
    --query "Reservations[].Instances[].[Tags[?Key=='Name']|[0].Value,InstanceId,PublicIpAddress]" \
    --output text 2>/dev/null
}

terminate() {  # instance-id, arm
  aws ec2 terminate-instances --instance-ids "$1" >/dev/null 2>&1 \
    && echo "TERMINATED $2 $1"
}

launch() {  # arm
  local arm=$1 out
  out=$(uv run --with boto3 --with 'botocore[crt]' launch-fleet "$arm" \
    --bucket $BUCKET --corpus_key union-v2.tar --security_group $SG \
    --key_name gpt-key --lifetime_hours 4 2>&1)
  if echo "$out" | grep -q '^launched'; then
    echo "LAUNCHED_OD $arm"; return 0
  fi
  out=$(uv run --with boto3 --with 'botocore[crt]' launch-fleet "$arm" \
    --bucket $BUCKET --corpus_key union-v2.tar --security_group $SG \
    --key_name gpt-key --spot "$arm" --spot_types g6.xlarge g5.xlarge \
    --subnets $SUBNETS --lifetime_hours 4 2>&1)
  if echo "$out" | grep -q '^launched'; then
    echo "LAUNCHED_SPOT $arm"; return 0
  fi
  return 1
}

while :; do
  NOW=$(date -u +%s)
  if [ "$NOW" -ge "$DEADLINE_UTC" ]; then
    echo "DEADLINE_REACHED terminating fleet"
    while read -r NAME IID IP; do
      [ -n "${IID:-}" ] && terminate "$IID" "${NAME#gate1-}"
    done <<< "$(fleet)"
    echo "OVERNIGHT_ENDED_ON_DEADLINE"
    exit 0
  fi

  ROWS=$(fleet)
  RUNNING=0
  # Iterate over a saved copy rather than a live stream: any command inside the
  # body that touches stdin would otherwise eat the remaining rows.
  mapfile -t FLEET_ROWS <<< "$ROWS"
  for _row in "${FLEET_ROWS[@]}"; do
    read -r NAME IID IP <<< "$_row"
    [ -z "${IP:-}" ] && continue
    RUNNING=$((RUNNING+1))
    ARM="${NAME#gate1-}"

    DONE_RC=$($SSH ubuntu@"$IP" "sudo cat /root/gpt/runs/gate1/$ARM/ARM_DONE 2>/dev/null" 2>/dev/null)
    [ -z "$DONE_RC" ] && continue

    # Training finished -> start the on-box continuation once.
    case " $started_post " in
      *" $ARM "*) ;;
      *)
        if [ "$DONE_RC" = "0" ]; then
          SU=$(presign "results/$ARM.tar" put)
          FU=$(presign "sft-v2.tar" get)
          RU=$(presign "results/$ARM-report.tar" put)
          # The repo lives under /root, so every step must run as root. An
          # earlier version ran `cd /root/gpt` as ubuntu, which fails with
          # permission denied and silently took the whole chain with it.
          scp -o StrictHostKeyChecking=no -o ConnectTimeout=12 -i "$KEY" -q \
            scripts/box_kick.sh ubuntu@"$IP":/tmp/kick.sh >/dev/null 2>&1 </dev/null
          KICK=$($SSH ubuntu@"$IP" \
            "sudo bash /tmp/kick.sh '$ARM' '$SU' '$FU' '$RU'" 2>/dev/null </dev/null)
          # Only record the arm as started once the box confirms it. A silently
          # failed scp once left the kick running against a nonexistent file:
          # the arm was marked started, never retried, and a finished on-demand
          # box sat idle until someone looked.
          case "$KICK" in
            KICKED*|ALREADY*)
              echo "POST_RUN_START $ARM"
              started_post="$started_post $ARM"
              ;;
            *)
              echo "POST_RUN_KICK_FAILED $ARM -- retrying next cycle"
              ;;
          esac
        else
          echo "TRAIN_FAILED $ARM rc=$DONE_RC"
          started_post="$started_post $ARM"
        fi
        ;;
    esac

    # Continuation finished -> collect, verify, then and only then terminate.
    PR=$($SSH ubuntu@"$IP" "sudo cat /root/gpt/runs/gate1/$ARM/POST_RUN_DONE 2>/dev/null" 2>/dev/null)
    if [ -n "$PR" ]; then
      case " $collected " in *" $ARM "*) continue;; esac
      aws s3 cp "s3://$BUCKET/results/$ARM-report.tar" "$RESULTS/$ARM-report.tar" \
        --only-show-errors 2>/dev/null </dev/null
      if [ -s "$RESULTS/$ARM-report.tar" ] && tar -tf "$RESULTS/$ARM-report.tar" >/dev/null 2>&1; then
        SZ=$(stat -c%s "$RESULTS/$ARM-report.tar")
        rm -rf "$RESULTS/$ARM" && mkdir -p "$RESULTS/$ARM"
        tar -xf "$RESULTS/$ARM-report.tar" -C "$RESULTS/$ARM"
        echo "COLLECTED $ARM ${SZ} bytes -> verified locally"
        collected="$collected $ARM"
        terminate "$IID" "$ARM"
      else
        echo "COLLECT_BAD_TAR $ARM -- leaving box up"
      fi
    fi
  done

  # Fill freed capacity. 8 vCPU on-demand + 8 spot = 4 boxes of g6.xlarge.
  if [ "$RUNNING" -lt 4 ] && [ -n "${QUEUE// }" ]; then
    NEXT=$(echo $QUEUE | awk '{print $1}')
    if launch "$NEXT"; then
      QUEUE=$(echo $QUEUE | cut -d' ' -f2-)
      [ "$QUEUE" = "$NEXT" ] && QUEUE=""
    fi
  fi

  if [ -z "${QUEUE// }" ] && [ "$RUNNING" -eq 0 ]; then
    echo "ALL_ARMS_DONE collected:$collected"
    exit 0
  fi
  sleep 90
done

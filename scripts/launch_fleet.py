"""Launch a whole Gate 1 wave at once.

Cloud capacity is elastic, so running arms one after another buys nothing but
wall-clock. This launches every arm in a wave simultaneously, each on its own
box, and each box is self-terminating.

The corpus is staged once in a private S3 bucket in the caller's own account
and handed to each box as a time-limited presigned URL, so no credential is
ever written into user-data and nothing is published publicly. Transfer inside
the region is free.

User-data is base64-encoded: an earlier attempt at inline heredocs had its
export lines silently collapse, which is the sort of failure that only shows up
after you have paid for the box.

Presigning needs boto3, which is deliberately not a locked dependency (the
training boxes use plain curl and should not install it), so run this as:

    uv run --with boto3 --with 'botocore[crt]' launch-fleet ...
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
from typing import Dict, List

REGION = "us-east-1"
DEFAULT_AMI = "ami-012ba162b9cd2729c"  # Deep Learning OSS Nvidia PyTorch 2.7, Ubuntu 22.04
DEFAULT_TYPE = "g6.xlarge"  # 1x L4
REPO_URL = "https://github.com/arksomething/gpt.git"


def sh(cmd: List[str], **kw) -> str:
    return subprocess.run(
        cmd, check=True, capture_output=True, text=True, **kw
    ).stdout.strip()


def presign(bucket: str, key: str, ttl: int, method: str = "get") -> str:
    """Presign an S3 URL.

    `aws s3 presign` cannot sign PUT requests (no --http-method on this CLI
    version), and the shipper needs an upload URL, so this goes through boto3.
    """
    import boto3  # imported lazily so --dry_run works without credentials

    client = boto3.client("s3", region_name=REGION)
    op = "put_object" if method == "put" else "get_object"
    return client.generate_presigned_url(
        op, Params={"Bucket": bucket, "Key": key}, ExpiresIn=ttl
    )


def runner_script(
    arm: str, corpus_url: str, ship_url: str, branch: str, lifetime: int,
    ship_every: int, corpus_name: str, is_spot: bool = True
) -> str:
    """The script the box runs. Kept flat: no nested quoting, no heredocs.

    The shipper is the durability guarantee. Checkpoints go out over a
    presigned PUT URL, which needs no credentials on the box -- so a spot
    preemption, a kill-switch shutdown, or a crash costs at most one ship
    interval instead of the whole run.
    """
    spot_watcher_open = "" if is_spot else ": <<'NO_SPOT_WATCHER'\n"
    spot_watcher_close = "" if is_spot else "\nNO_SPOT_WATCHER"
    return f"""#!/bin/bash
set -x
exec > >(tee -a /var/log/gate1-arm.log) 2>&1
export DEBIAN_FRONTEND=noninteractive
export HOME=/root
export PATH=/root/.local/bin:/usr/local/bin:/usr/bin:/bin

shutdown -h +{lifetime * 60} || true

RUNDIR=/root/gpt/runs/gate1/{arm}

ship() {{
  # Tar to a temp name and move into place so a PUT never reads a partial file.
  #
  # tar exits 1 with "file changed as we read it" whenever training writes a
  # checkpoint mid-archive, which is routine here. Treating that as failure
  # skipped the upload silently and shipped nothing for an entire run. Exit 1
  # is a warning; only >=2 is fatal, and the archive being non-empty is the
  # check that actually matters.
  mkdir -p "$RUNDIR"
  tar --warning=no-file-changed -cf /tmp/ship.tmp -C /root/gpt/runs/gate1 {arm} 2>/dev/null
  rc=$?
  [ "$rc" -le 1 ] && [ -s /tmp/ship.tmp ] || {{ echo "SHIP_TAR_FAIL rc=$rc"; return 1; }}
  mv -f /tmp/ship.tmp /tmp/ship.tar
  curl -fsS -X PUT --upload-file /tmp/ship.tar "{ship_url}" && echo "SHIPPED $(date -u +%H:%M:%S)"
}}

# Periodic shipper: bounded loss window regardless of how the box dies.
( while true; do sleep {ship_every}; ship || echo SHIP_FAILED; done ) &

# Spot gives a two-minute interruption notice on IMDS. Catching it turns a
# preemption from "lose the run" into "lose nothing". Only armed on spot: on an
# on-demand box the endpoint always 404s and the poll loop buries the training
# output under `set -x` trace lines.
{spot_watcher_open}( while true; do
    TOK=$(curl -fsS -X PUT "http://169.254.169.254/latest/api/token" \
      -H "X-aws-ec2-metadata-token-ttl-seconds: 300" 2>/dev/null)
    CODE=$(curl -o /dev/null -w "%{{http_code}}" -fsS \
      -H "X-aws-ec2-metadata-token: $TOK" \
      http://169.254.169.254/latest/meta-data/spot/instance-action 2>/dev/null)
    if [ "$CODE" = "200" ]; then echo SPOT_INTERRUPTION; ship; break; fi
    sleep 5
  done ) &{spot_watcher_close}

curl -fsSL https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH

git clone --branch {branch} {REPO_URL} /root/gpt
cd /root/gpt
uv sync --frozen

mkdir -p /root/gpt/data/gate1
curl -fsSL -o /tmp/union.tar "{corpus_url}"
tar -xf /tmp/union.tar -C /root/gpt/data/gate1
rm -f /tmp/union.tar

uv run --with pytest pytest tests/ -q || {{ echo GATE0_FAILED; shutdown -h now; }}
uv run review-corpus {corpus_name}/train --per_source 15 || {{ echo CORPUS_FAILED; shutdown -h now; }}

uv run train --train_config configs/gate1/{arm}.yaml --model_config configs/model_25m.yaml
RC=$?
echo "ARM_{arm}_RC=$RC"

# Ship straight away rather than waiting out the periodic interval, then again
# after the marker so the final state is always what is in S3.
ship || echo SHIP_FAILED

# Deliberately no shutdown here. A box that terminates the moment training ends
# takes its results with it -- that is how the first bench run was lost. The box
# stays up until the results have been copied off and verified locally; the
# scheduled kill-switch above is the only automatic terminator.
mkdir -p "$RUNDIR"
echo "$RC" > "$RUNDIR/ARM_DONE"
ship || echo SHIP_FAILED
echo "ARM_{arm}_DONE rc=$RC"
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs="+", help="Arm names, e.g. b0-seed1 p1 p2")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--corpus_key", default="union-v2.tar")
    ap.add_argument("--instance_type", default=DEFAULT_TYPE)
    ap.add_argument("--ami", default=DEFAULT_AMI)
    ap.add_argument("--key_name", default="gpt-key")
    ap.add_argument("--security_group", required=True)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--lifetime_hours", type=int, default=8)
    ap.add_argument("--url_ttl", type=int, default=43200, help="Presign TTL seconds")
    ap.add_argument(
        "--ship_every",
        type=int,
        default=600,
        help="Seconds between checkpoint ships; the maximum work a crash can cost",
    )
    ap.add_argument(
        "--spot", nargs="*", default=[], help="Arms to place on spot capacity"
    )
    ap.add_argument(
        "--spot_types",
        nargs="*",
        default=["g6.xlarge", "g5.xlarge"],
        help="Spot instance types to try in order. bf16-capable only: a T4 "
        "(g4dn) would force fp16 and make that arm incomparable.",
    )
    ap.add_argument(
        "--subnets", nargs="*", default=[], help="Subnets (AZs) to try per type"
    )
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    for arm in args.arms:
        cfg = f"configs/gate1/{arm}.yaml"
        if not os.path.exists(cfg):
            sys.exit(f"missing {cfg} -- run `uv run gate1-arms` first")
        # Boxes clone from the remote, so a config that exists only locally is
        # invisible to them: an arm was launched before its config was pushed
        # and died on FileNotFoundError after paying for the whole bootstrap.
        try:
            subprocess.run(
                ["git", "cat-file", "-e", f"origin/{args.branch}:{cfg}"],
                check=True, capture_output=True,
            )
        except subprocess.CalledProcessError:
            sys.exit(
                f"{cfg} is not on origin/{args.branch}. The box clones the "
                f"remote, so commit and push before launching."
            )

    corpus_url = presign(args.bucket, args.corpus_key, args.url_ttl, "get")
    # The tar unpacks to a directory named after the corpus; the arm configs
    # reference that same path, so a mismatch here means training on the wrong
    # corpus or on nothing at all.
    corpus_name = "data/gate1/" + args.corpus_key.removesuffix(".tar")

    launched: Dict[str, str] = {}
    for arm in args.arms:
        # One presigned PUT per arm, reused by the shipper for the whole run.
        ship_url = presign(
            args.bucket, f"results/{arm}.tar", args.url_ttl, "put"
        )
        script = runner_script(
            arm, corpus_url, ship_url, args.branch, args.lifetime_hours,
            args.ship_every, corpus_name, arm in args.spot
        ).replace("{BUCKET}", args.bucket)
        user_data = base64.b64encode(script.encode()).decode()

        cmd = [
            "aws", "ec2", "run-instances",
            "--image-id", args.ami,
            "--instance-type", args.instance_type,
            "--key-name", args.key_name,
            "--security-group-ids", args.security_group,
            "--user-data", user_data,
            "--instance-initiated-shutdown-behavior", "terminate",
            "--block-device-mappings",
            "DeviceName=/dev/sda1,Ebs={VolumeSize=120,VolumeType=gp3,DeleteOnTermination=true}",
            "--tag-specifications",
            f"ResourceType=instance,Tags=[{{Key=Name,Value=gate1-{arm}}},{{Key=Project,Value=gate1}}]",
            "--region", REGION,
            "--query", "Instances[0].InstanceId",
            "--output", "text",
        ]
        if arm in args.spot:
            cmd[3:3] = [
                "--instance-market-options",
                "MarketType=spot,SpotOptions={SpotInstanceType=one-time}",
            ]

        if args.dry_run:
            print(f"[dry-run] {arm}: {args.instance_type}"
                  f"{' (spot)' if arm in args.spot else ''}")
            continue

        if arm not in args.spot:
            try:
                instance_id = sh(cmd)
                launched[arm] = instance_id
                print(f"launched {arm:10s} {instance_id}  {args.instance_type} "
                      f"(on-demand)", flush=True)
            except subprocess.CalledProcessError as e:
                print(f"FAILED  {arm}: {e.stderr.strip()[:300]}", flush=True)
            continue

        # Spot capacity is per (instance type, availability zone), and a single
        # pool being empty says nothing about the others -- so walk the pools
        # instead of giving up the parallelism. Only bf16-capable GPUs are
        # candidates: a T4 would force fp16 and make that arm's numerics
        # incomparable to the rest of the experiment.
        placed = False
        for itype in args.spot_types:
            for subnet in args.subnets or [None]:
                attempt = list(cmd)
                attempt[attempt.index(args.instance_type)] = itype
                if subnet:
                    attempt += ["--subnet-id", subnet]
                try:
                    instance_id = sh(attempt)
                    launched[arm] = instance_id
                    where = f" {subnet}" if subnet else ""
                    print(f"launched {arm:10s} {instance_id}  {itype} (spot){where}",
                          flush=True)
                    placed = True
                    break
                except subprocess.CalledProcessError as e:
                    # "no capacity here" and "this AZ doesn't offer this type"
                    # are both reasons to try the next pool, not to stop: an
                    # AZ that cannot host g6 at all says nothing about the rest.
                    skippable = ("InsufficientInstanceCapacity", "Unsupported")
                    if not any(s in e.stderr for s in skippable):
                        print(f"FAILED  {arm} [{itype}]: {e.stderr.strip()[:200]}",
                              flush=True)
                        break
            if placed:
                break
        if not placed:
            print(f"FAILED  {arm}: no spot capacity in any pool tried", flush=True)

    if launched:
        print("\n" + json.dumps(launched, indent=2))
        print(f"\n{len(launched)}/{len(args.arms)} arms running in parallel.")
        print("Each box writes runs/gate1/<arm>/ARM_DONE when finished and stays")
        print("up. Copy results off and verify them locally before terminating.")


if __name__ == "__main__":
    main()

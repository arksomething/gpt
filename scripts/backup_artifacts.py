"""Back up trained artifacts to the Hub.

Every model this project has produced so far lives on exactly one laptop. That
is one disk failure away from losing work that cost real money and real GPU
hours to produce, and it cannot be reproduced exactly -- the corpora were built
from streaming sources whose contents drift.

Repos are created PRIVATE. Publishing is a separate decision from backing up,
and this script only does the latter.
"""

from __future__ import annotations

import argparse
import os
import sys

from huggingface_hub import HfApi

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# local path -> repo name. Ordered most-valuable-first so a partial run still
# saves the things that would hurt most to lose.
DEFAULT_TARGETS = [
    ("runs/hf/25m-base", "gpt-25m-base-hf"),
    ("runs/hf/25m-chat", "gpt-25m-chat-hf"),
    ("runs/bench-20260730", "gpt-25m-bench-20260730"),
    ("runs/probes/25m-aws-20260730", "gpt-25m-probe-20260730"),
    ("runs/sft/25m-dolly-rehearsal-20260730", "gpt-25m-sft-20260730"),
]


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user", default="ark296")
    ap.add_argument("--public", action="store_true", help="Opt in to public repos")
    ap.add_argument("--only", nargs="*", help="Subset of local paths to back up")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    targets = DEFAULT_TARGETS
    if args.only:
        targets = [t for t in targets if t[0] in set(args.only)]

    api = HfApi()
    failures = []
    for local, repo_name in targets:
        path = os.path.join(REPO_ROOT, local)
        if not os.path.isdir(path):
            print(f"skip   {local} (not present)", flush=True)
            continue
        repo_id = f"{args.user}/{repo_name}"
        size = dir_size(path)
        if args.dry_run:
            print(f"[dry-run] {local} -> {repo_id} ({human(size)})", flush=True)
            continue

        print(f"upload {local} -> {repo_id} ({human(size)})...", flush=True)
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=not args.public,
                exist_ok=True,
            )
            api.upload_folder(
                folder_path=path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Back up {local}",
            )
            print(f"   done https://huggingface.co/{repo_id}", flush=True)
        except Exception as e:  # noqa: BLE001 - report and continue to next target
            print(f"   FAILED {repo_id}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            failures.append(repo_id)

    if failures:
        print(f"\n{len(failures)} backup(s) FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("\nAll backups complete.")


if __name__ == "__main__":
    main()

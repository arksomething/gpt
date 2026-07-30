"""Publish an indexed corpus to the Hub so a fleet can pull it in parallel.

Every Gate 1 arm must train on byte-identical data; the whole point of the
union corpus is that arms differ only in sampling weights. So the corpus is
uploaded once, and each box verifies the manifest fingerprints on arrival
rather than trusting the transfer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus_dir", help="Corpus root (contains train/ and validation/)")
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.corpus_dir):
        sys.exit(f"no such corpus dir: {args.corpus_dir}")

    fingerprints = {}
    for split in ("train", "validation"):
        manifest = os.path.join(args.corpus_dir, split, "manifest.json")
        if os.path.exists(manifest):
            with open(manifest) as f:
                m = json.load(f)
            fingerprints[split] = {
                "tokenizer_sha256": m.get("tokenizer_sha256"),
                "recipe_sha256": m.get("recipe_sha256"),
                "documents": m.get("document_count"),
                "tokens": m.get("token_count"),
            }
    print(json.dumps(fingerprints, indent=2))

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=args.corpus_dir,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Publish Gate 1 union corpus",
    )
    print(f"\nPublished to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()

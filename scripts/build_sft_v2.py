"""Build the SFT v2 chat corpus.

Round 1 trained on Dolly-15k: short-form closed QA, no multi-turn, no small
talk. It was then tested conversationally and answered the distribution it had
seen. SmolTalk is the direct fix -- it was ablated on 135M-1.7B models, our
exact size class, and its everyday-conversations subset is chitchat.

Mixture is weighted toward conversation on purpose: the goal is a
conversationalist, and instruction-following capability is not reachable at 25M
anyway, so tokens spent on hard reasoning prompts buy nothing here.

Emits the repo chat schema; encoding to tokens (with the v2 EOS terminator)
happens in prepare_chat_data.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, Iterator, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# (repo, config, cap, why)
SOURCES = [
    (
        "HuggingFaceTB/smoltalk",
        "everyday-conversations",
        4000,
        "chitchat -- the register round 1 never saw",
    ),
    (
        "HuggingFaceTB/smoltalk",
        "smol-magpie-ultra",
        6000,
        "general instruct, multi-turn",
    ),
]

# Chitchat turns are short by nature -- "Hi", "Thanks!", "Sure, what about
# Tuesday?". An 8-char floor threw away 85% of everyday-conversations, which is
# precisely the register round 1 was missing. Same mistake as applying a
# prose-calibrated readability threshold to IRC logs.
MIN_CHARS = 2
MAX_CHARS = 4000
MAX_TURNS = 12


def _clean(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Normalize to the repo schema, or return [] if unusable."""
    out: List[Dict[str, str]] = []
    for m in messages[:MAX_TURNS]:
        role = str(m.get("role", "")).strip().lower()
        content = str(m.get("content", "")).strip()
        if role == "system":
            continue  # the template supports it, but keep v2 prompts uniform
        if role not in ("user", "assistant"):
            return []
        if not (MIN_CHARS <= len(content) <= MAX_CHARS):
            return []
        out.append({"role": role, "content": content})
    # Must alternate and end on assistant, or supervision has nothing to mask.
    if len(out) < 2 or out[0]["role"] != "user" or out[-1]["role"] != "assistant":
        return []
    for i, m in enumerate(out):
        if m["role"] != ("user" if i % 2 == 0 else "assistant"):
            return []
    return out


def stream_source(repo: str, config: str, cap: int) -> Iterator[List[Dict[str, str]]]:
    from datasets import load_dataset

    ds = load_dataset(repo, config, split="train", streaming=True)
    kept = 0
    for row in ds:
        msgs = row.get("messages")
        if not isinstance(msgs, list):
            continue
        cleaned = _clean(msgs)
        if not cleaned:
            continue
        yield cleaned
        kept += 1
        if kept >= cap:
            return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/chat/v2/conversations.jsonl")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val_fraction", type=float, default=0.05)
    args = ap.parse_args()

    out_path = os.path.join(REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    convs: List[Dict[str, Any]] = []
    provenance = []
    for repo, config, cap, why in SOURCES:
        n = 0
        for msgs in stream_source(repo, config, cap):
            convs.append({"source": f"{repo}:{config}", "messages": msgs})
            n += 1
        provenance.append(
            {"repo": repo, "config": config, "kept": n, "cap": cap, "role": why}
        )
        print(f"{repo}:{config} -> {n} conversations ({why})", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(convs)
    n_val = max(1, int(len(convs) * args.val_fraction))
    val, train = convs[:n_val], convs[n_val:]

    for split, rows in (("train", train), ("validation", val)):
        path = out_path.replace(".jsonl", f".{split}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {path}: {len(rows)} conversations")

    manifest = {
        "sources": provenance,
        "total": len(convs),
        "train": len(train),
        "validation": len(val),
        "seed": args.seed,
        "template_version": 2,
    }
    mpath = os.path.join(os.path.dirname(out_path), "build_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {mpath}")


if __name__ == "__main__":
    main()

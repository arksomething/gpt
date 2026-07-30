#!/usr/bin/env python3
"""Build the SFT-rehearsal conversation file from databricks-dolly-15k.

Produces data/chat/dolly_rehearsal.jsonl in the repo chat schema. Kept as a
script (rather than a one-off) so the corpus is reproducible and its
provenance (source dataset, license, filters, seed) is inspectable.
"""

import json
import random
import urllib.request
from pathlib import Path

URL = (
    "https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
    "resolve/main/databricks-dolly-15k.jsonl"
)
OUT = Path("data/chat/dolly_rehearsal.jsonl")
MAX_USER_CHARS = 600
MAX_ASSISTANT_CHARS = 700
MIN_ASSISTANT_CHARS = 20
KEEP = 6000
SEED = 1337


def main() -> None:
    raw = urllib.request.urlopen(URL, timeout=120).read().decode()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    random.seed(SEED)
    out = []
    for r in rows:
        q = r["instruction"].strip()
        ctx = r.get("context", "").strip()
        a = r["response"].strip()
        if not q or not a:
            continue
        user = q if not ctx else f"{ctx}\n\n{q}"
        if (
            len(user) > MAX_USER_CHARS
            or len(a) > MAX_ASSISTANT_CHARS
            or len(a) < MIN_ASSISTANT_CHARS
        ):
            continue
        out.append(
            {
                "id": f"dolly-{len(out):05d}",
                "source": "databricks-dolly-15k",
                "license": "CC-BY-SA-3.0",
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": a},
                ],
            }
        )
    random.shuffle(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in out[:KEEP]:
            f.write(json.dumps(r) + "\n")
    print(f"kept {min(len(out), KEEP)} of {len(rows)}")


if __name__ == "__main__":
    main()

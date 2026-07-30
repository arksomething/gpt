#!/usr/bin/env python3
"""
Smoke-check the Gate 1 mixture sources against the live Hugging Face hub.

Streams a handful of documents from each registered source and prints the
repo id, config, and a truncated preview of each document. This is the one
place in the data pipeline that is allowed to hit the network on purpose: it
exists to catch wrong repo ids, wrong config names, and renamed text fields
before a weight is set in a real preparation run.

Usage:
  uv run check-sources
  uv run check-sources --docs 5
  uv run check-sources --sources dclm finemath
"""

import argparse
import sys
import traceback
from itertools import islice

from scripts.prepare_data import (
    MIXTURE_SOURCE_IDS,
    MIXTURE_SOURCES_BY_ID,
    MIXTURE_SOURCE_SPECS,
    open_mixture_stream,
)

PREVIEW_CHARS = 80


def check_source(spec, docs: int, seed: int) -> bool:
    """Stream `docs` documents from one source. Returns True on success."""
    label = f"{spec.source_id} [{spec.repo_id}"
    label += f":{spec.config}]" if spec.config else "]"
    print(f"\n=== {label} ({spec.filter_treatment} filters) ===", flush=True)

    stream = open_mixture_stream(spec, seed)
    seen = 0
    try:
        for text in islice(stream, docs):
            seen += 1
            preview = " ".join(text.split())[:PREVIEW_CHARS]
            print(f"  [{seen}] {spec.repo_id}: {preview}", flush=True)
    except Exception:
        traceback.print_exc()
        print(f"  FAILED: {spec.repo_id}", flush=True)
        return False
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    if seen < docs:
        print(f"  WARNING: only {seen}/{docs} documents streamed", flush=True)
    return seen > 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream a few documents from each Gate 1 mixture source."
    )
    parser.add_argument(
        "--docs",
        type=int,
        default=3,
        help="Documents to stream per source (default: 3).",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        choices=list(MIXTURE_SOURCE_IDS),
        help="Subset of source ids to check (default: all).",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Stream seed.")
    args = parser.parse_args()

    specs = (
        [MIXTURE_SOURCES_BY_ID[source_id] for source_id in args.sources]
        if args.sources
        else list(MIXTURE_SOURCE_SPECS)
    )

    failures = []
    for spec in specs:
        if not check_source(spec, args.docs, args.seed):
            failures.append(spec.source_id)

    print("\n" + "=" * 60)
    print(f"Checked {len(specs)} sources, {len(failures)} failed.")
    if failures:
        print("Failed: " + ", ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()

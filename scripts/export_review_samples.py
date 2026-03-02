#!/usr/bin/env python3
"""Export tagged source samples to a text file for manual quality review."""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime, timezone
from typing import Callable, Iterator

from scripts.gutenberg import stream_gutenberg
from scripts.prepare_data import stream_c4, stream_fineweb, stream_wikipedia


def _sanitize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    # Remove large runs of blank lines while preserving paragraph breaks.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _take_samples(
    source_name: str,
    iterator: Iterator[str],
    count: int,
    max_chars: int,
) -> list[tuple[str, int, bool]]:
    samples: list[tuple[str, int, bool]] = []
    try:
        while len(samples) < count:
            text = next(iterator)
            text = _sanitize_text(text)
            if not text:
                continue

            original_chars = len(text)
            truncated = False
            if max_chars > 0 and original_chars > max_chars:
                text = text[:max_chars].rstrip() + "\n\n[...truncated...]"
                truncated = True

            samples.append((text, original_chars, truncated))
    finally:
        close_fn = getattr(iterator, "close", None)
        if callable(close_fn):
            close_fn()

    print(f"[review] Collected {len(samples)} samples from {source_name}")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export source-tagged text samples for manual quality review.",
    )
    parser.add_argument(
        "--sources",
        default="c4,wiki,fineweb,books",
        help="Comma-separated sources: c4,wiki,fineweb,books",
    )
    parser.add_argument(
        "--samples_per_source",
        type=int,
        default=8,
        help="Number of samples to export per source",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=4000,
        help="Max chars per sample (0 to disable truncation)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Base random seed",
    )
    parser.add_argument(
        "--books_split",
        default="train",
        help="Split for books stream (train/validation/test)",
    )
    parser.add_argument(
        "--books_mode",
        default="books_fast",
        choices=["books_fast", "pg19_only", "variety"],
        help="Mode for scripts.gutenberg stream",
    )
    parser.add_argument(
        "--books_clean_level",
        default="balanced",
        choices=["off", "balanced", "aggressive"],
        help="PG-19 cleanup level used by books modes",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output txt path (default: data/review/samples_YYYYmmdd_HHMMSS.txt)",
    )
    args = parser.parse_args()

    requested = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    valid_sources = {"c4", "wiki", "fineweb", "books"}
    unknown = sorted(set(requested) - valid_sources)
    if unknown:
        raise SystemExit(f"Unknown source(s): {', '.join(unknown)}")
    if not requested:
        raise SystemExit("No sources selected.")
    if args.samples_per_source <= 0:
        raise SystemExit("--samples_per_source must be > 0")

    now_utc = datetime.now(timezone.utc)
    ts = now_utc.strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"data/review/samples_{ts}.txt"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Configure books stream mode in-process for deterministic testing.
    os.environ["VARIETY_DATASET_MODE"] = args.books_mode
    os.environ["PG19_CLEAN_LEVEL"] = args.books_clean_level

    builders: dict[str, Callable[[], Iterator[str]]] = {
        "c4": lambda: stream_c4(seed=args.seed),
        "wiki": lambda: stream_wikipedia(seed=args.seed + 1),
        "fineweb": lambda: stream_fineweb(seed=args.seed + 2),
        "books": lambda: stream_gutenberg(seed=args.seed + 3, split=args.books_split),
    }

    all_samples: list[tuple[str, int, str, int, bool]] = []
    for source in requested:
        source_iter = builders[source]()
        samples = _take_samples(
            source_name=source,
            iterator=source_iter,
            count=args.samples_per_source,
            max_chars=args.max_chars,
        )
        for idx, (text, original_chars, truncated) in enumerate(samples, start=1):
            all_samples.append((source, idx, text, original_chars, truncated))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Data Quality Review Samples\n")
        f.write("=" * 80 + "\n")
        f.write(f"generated_utc: {now_utc.isoformat()}\n")
        f.write(f"sources: {', '.join(requested)}\n")
        f.write(f"samples_per_source: {args.samples_per_source}\n")
        f.write(f"max_chars: {args.max_chars}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"books_mode: {args.books_mode}\n")
        f.write(f"books_clean_level: {args.books_clean_level}\n")
        f.write(f"books_split: {args.books_split}\n")
        f.write("=" * 80 + "\n\n")

        global_idx = 0
        for source, source_idx, text, original_chars, truncated in all_samples:
            global_idx += 1
            f.write(f"[sample {global_idx:03d}]\n")
            f.write(f"source: {source}\n")
            f.write(f"source_sample_index: {source_idx}\n")
            f.write(f"original_chars: {original_chars}\n")
            f.write(f"truncated: {'yes' if truncated else 'no'}\n")
            f.write("-" * 80 + "\n")
            f.write(text + "\n")
            f.write("\n" + "=" * 80 + "\n\n")

    print(f"[review] Wrote {len(all_samples)} tagged samples to {output_path}")


if __name__ == "__main__":
    main()

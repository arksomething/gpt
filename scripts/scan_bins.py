#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan tokenized .bin files for range and frequency issues."
    )
    parser.add_argument("--train", help="Path to train.bin", default=None)
    parser.add_argument("--val", help="Path to val.bin", default=None)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--dtype", default="uint16")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument(
        "--special-ids",
        type=int,
        nargs="*",
        default=[0, 1, 2, 3],
        help="Token IDs to report frequency for (default: 0 1 2 3).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Tokens per chunk when scanning (default: 1,000,000).",
    )
    args = parser.parse_args()
    if not args.train and not args.val:
        parser.error("Provide at least one of --train or --val.")
    if args.vocab_size <= 0:
        parser.error("--vocab-size must be > 0.")
    if args.topk <= 0:
        parser.error("--topk must be > 0.")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be > 0.")
    try:
        args.dtype = np.dtype(args.dtype)
    except TypeError as exc:
        parser.error(f"Invalid --dtype '{args.dtype}': {exc}")
    return args


def scan_file(path, dtype, vocab_size, topk, special_ids, chunk_size):
    if not os.path.exists(path):
        raise SystemExit(f"Missing data file: {path}")
    if os.path.isdir(path):
        raise SystemExit(f"Expected file but found directory: {path}")

    size_bytes = os.path.getsize(path)
    itemsize = dtype.itemsize
    if size_bytes % itemsize != 0:
        print(
            f"[warn] {path}: size {size_bytes} not divisible by dtype size {itemsize}"
        )
    total_tokens = size_bytes // itemsize
    print(f"\n{path}")
    print(f"- size: {size_bytes} bytes")
    print(f"- dtype: {dtype}")
    print(f"- tokens: {total_tokens}")

    data = np.memmap(path, dtype=dtype, mode="r")
    counts = np.zeros(vocab_size, dtype=np.int64)
    min_id = None
    max_id = None
    out_of_range = 0

    for start in range(0, data.shape[0], chunk_size):
        end = min(start + chunk_size, data.shape[0])
        chunk = np.asarray(data[start:end])
        if chunk.size == 0:
            continue
        cmin = int(chunk.min())
        cmax = int(chunk.max())
        min_id = cmin if min_id is None else min(min_id, cmin)
        max_id = cmax if max_id is None else max(max_id, cmax)

        in_range = chunk[(chunk >= 0) & (chunk < vocab_size)]
        out_of_range += chunk.size - in_range.size
        if in_range.size:
            counts += np.bincount(in_range, minlength=vocab_size)

    min_id = 0 if min_id is None else min_id
    max_id = 0 if max_id is None else max_id
    print(f"- min_id: {min_id}")
    print(f"- max_id: {max_id}")
    if max_id >= vocab_size or min_id < 0:
        print(
            f"[warn] found out-of-range ids: min={min_id} max={max_id} "
            f"(vocab_size={vocab_size})"
        )
    if out_of_range:
        frac = out_of_range / max(1, total_tokens)
        print(f"- out_of_range: {out_of_range} ({frac:.4%})")

    total_in_range = counts.sum()
    denom = max(1, total_tokens)
    for tok_id in special_ids:
        if 0 <= tok_id < vocab_size:
            count = int(counts[tok_id])
            print(f"- token[{tok_id}] fraction: {count / denom:.4%}")
        else:
            print(f"- token[{tok_id}] fraction: n/a (outside vocab_size)")

    top_ids = counts.argsort()[-topk:][::-1]
    top_list = [(int(i), int(counts[i])) for i in top_ids if counts[i] > 0]
    print(f"- top_{topk}_ids: {top_list}")

    if total_in_range + out_of_range != total_tokens:
        print(
            f"[warn] count mismatch: in_range={total_in_range} "
            f"out_of_range={out_of_range} total={total_tokens}"
        )


def main():
    args = parse_args()
    if args.train:
        scan_file(
            args.train,
            args.dtype,
            args.vocab_size,
            args.topk,
            args.special_ids,
            args.chunk_size,
        )
    if args.val:
        scan_file(
            args.val,
            args.dtype,
            args.vocab_size,
            args.topk,
            args.special_ids,
            args.chunk_size,
        )


if __name__ == "__main__":
    main()

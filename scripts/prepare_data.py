#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import yaml
from datasets import interleave_datasets, load_dataset
import sentencepiece as spm


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _keep_text_only(ds):
    cols = [c for c in ds.column_names if c != "text"]
    return ds.remove_columns(cols) if cols else ds


def _load_streaming_mix(c4_weight, wiki_weight, seed, shuffle_buffer):
    c4 = load_dataset("allenai/c4", "en", split="train", streaming=True)
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    c4 = _keep_text_only(c4)
    wiki = _keep_text_only(wiki)
    ds = interleave_datasets(
        [c4, wiki],
        probabilities=[c4_weight, wiki_weight],
        seed=seed,
        stopping_strategy="first_exhausted",
    )
    return ds.shuffle(seed=seed, buffer_size=shuffle_buffer)


def _clean_text(text):
    text = text or ""
    return " ".join(text.split())


def _write_tokens(arr, idx, tokens):
    remaining = len(arr) - idx
    if remaining <= 0:
        return idx, 0
    n = min(len(tokens), remaining)
    if n > 0:
        arr[idx : idx + n] = tokens[:n]
    return idx + n, n


def main():
    parser = argparse.ArgumentParser(description="Prepare memmapped token data.")
    parser.add_argument("--config", default="configs/train.yaml", help="Config file path")
    parser.add_argument("--tokenizer_model", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--train_tokens", type=int, default=None)
    parser.add_argument("--val_tokens", type=int, default=None)
    parser.add_argument("--c4_weight", type=float, default=None)
    parser.add_argument("--wiki_weight", type=float, default=None)
    parser.add_argument("--shuffle_buffer", type=int, default=None)
    parser.add_argument("--min_chars", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None, help="Seconds between progress logs.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    # Load defaults from config, allow CLI overrides
    cfg = {}
    if os.path.exists(args.config):
        cfg = load_yaml(args.config).get("data_prep", {})

    args.tokenizer_model = args.tokenizer_model or cfg.get("tokenizer_model", "tokenizer/spm.model")
    args.out_dir = args.out_dir or cfg.get("out_dir", "data")
    args.train_tokens = args.train_tokens if args.train_tokens is not None else cfg.get("train_tokens", 2_000_000_000)
    args.val_tokens = args.val_tokens if args.val_tokens is not None else cfg.get("val_tokens", 20_000_000)
    args.c4_weight = args.c4_weight if args.c4_weight is not None else cfg.get("c4_weight", 0.9)
    args.wiki_weight = args.wiki_weight if args.wiki_weight is not None else cfg.get("wiki_weight", 0.1)
    args.shuffle_buffer = args.shuffle_buffer if args.shuffle_buffer is not None else cfg.get("shuffle_buffer", 50_000)
    args.min_chars = args.min_chars if args.min_chars is not None else cfg.get("min_chars", 200)
    args.seed = args.seed if args.seed is not None else cfg.get("seed", 1337)
    args.log_interval = args.log_interval if args.log_interval is not None else cfg.get("log_interval", 30)

    if args.c4_weight + args.wiki_weight <= 0:
        raise SystemExit("c4_weight + wiki_weight must be > 0")

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.bin")
    val_path = os.path.join(args.out_dir, "val.bin")
    if (os.path.exists(train_path) or os.path.exists(val_path)) and not args.overwrite:
        raise SystemExit("train.bin/val.bin exist. Use --overwrite to replace.")

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer_model)
    vocab_size = sp.get_piece_size()
    if vocab_size > np.iinfo(np.uint16).max:
        raise SystemExit("Vocab too large for uint16. Use uint32.")

    train_arr = np.memmap(train_path, dtype=np.uint16, mode="w+", shape=(args.train_tokens,))
    val_arr = np.memmap(val_path, dtype=np.uint16, mode="w+", shape=(args.val_tokens,))

    rng = random.Random(args.seed)
    ds = _load_streaming_mix(args.c4_weight, args.wiki_weight, args.seed, args.shuffle_buffer)
    p_val = args.val_tokens / max(1, args.train_tokens + args.val_tokens)

    train_idx = 0
    val_idx = 0
    doc_lens = []
    out_of_range = 0
    total_docs = 0
    start_time = time.time()
    last_log = start_time
    last_tokens = 0
    last_docs = 0

    for ex in ds:
        if train_idx >= args.train_tokens and val_idx >= args.val_tokens:
            break
        text = _clean_text(ex.get("text"))
        if len(text) < args.min_chars:
            continue
        ids = [1] + sp.encode(text, out_type=int) + [2]
        doc_lens.append(len(ids))
        total_docs += 1
        if any(t >= vocab_size or t < 0 for t in ids):
            out_of_range += 1
            continue
        use_val = rng.random() < p_val
        if use_val and val_idx < args.val_tokens or train_idx >= args.train_tokens:
            val_idx, _ = _write_tokens(val_arr, val_idx, ids)
        else:
            train_idx, _ = _write_tokens(train_arr, train_idx, ids)

        if args.log_interval > 0 and (time.time() - last_log) >= args.log_interval:
            now = time.time()
            total_tokens = train_idx + val_idx
            delta_tokens = total_tokens - last_tokens
            delta_docs = total_docs - last_docs
            elapsed = now - start_time
            interval = max(1e-6, now - last_log)
            tps = delta_tokens / interval
            avg_tps = total_tokens / max(1e-6, elapsed)
            print(
                f"[{elapsed:.0f}s] tokens={total_tokens} train={train_idx} val={val_idx} "
                f"docs={total_docs} (+{delta_docs}) tps={tps:.0f} avg_tps={avg_tps:.0f}",
                flush=True,
            )
            last_log = now
            last_tokens = total_tokens
            last_docs = total_docs

    train_arr.flush()
    val_arr.flush()

    meta = {
        "tokenizer_model": args.tokenizer_model,
        "train_tokens": int(train_idx),
        "val_tokens": int(val_idx),
        "vocab_size": int(vocab_size),
        "docs_seen": int(total_docs),
        "mean_doc_len": float(np.mean(doc_lens)) if doc_lens else 0.0,
        "min_doc_len": int(min(doc_lens)) if doc_lens else 0,
        "max_doc_len": int(max(doc_lens)) if doc_lens else 0,
        "out_of_range_docs": int(out_of_range),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(args.out_dir, "data_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Wrote:")
    print(f"  {train_path} tokens={train_idx}")
    print(f"  {val_path} tokens={val_idx}")
    print(f"  metadata={os.path.join(args.out_dir, 'data_meta.json')}")


if __name__ == "__main__":
    main()

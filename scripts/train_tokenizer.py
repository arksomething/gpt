#!/usr/bin/env python3
import argparse
import json
import os
import random
from datetime import datetime

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


def _iter_texts(ds, min_chars, max_chars):
    total = 0
    for ex in ds:
        text = ex.get("text") or ""
        text = " ".join(text.split())
        if len(text) < min_chars:
            continue
        yield text
        total += len(text)
        if total >= max_chars:
            break


def main():
    parser = argparse.ArgumentParser(description="Train a SentencePiece BPE tokenizer.")
    parser.add_argument("--config", default="configs/train.yaml", help="Config file path")
    parser.add_argument("--output_dir", default=None, help="Output directory.")
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--c4_weight", type=float, default=None)
    parser.add_argument("--wiki_weight", type=float, default=None)
    parser.add_argument("--min_chars", type=int, default=None)
    parser.add_argument("--max_chars", type=int, default=None)
    parser.add_argument("--shuffle_buffer", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Load defaults from config, allow CLI overrides
    cfg = {}
    if os.path.exists(args.config):
        cfg = load_yaml(args.config).get("tokenizer_training", {})

    args.output_dir = args.output_dir or cfg.get("output_dir", "tokenizer")
    args.vocab_size = args.vocab_size if args.vocab_size is not None else cfg.get("vocab_size", 32000)
    args.c4_weight = args.c4_weight if args.c4_weight is not None else cfg.get("c4_weight", 0.9)
    args.wiki_weight = args.wiki_weight if args.wiki_weight is not None else cfg.get("wiki_weight", 0.1)
    args.min_chars = args.min_chars if args.min_chars is not None else cfg.get("min_chars", 200)
    args.max_chars = args.max_chars if args.max_chars is not None else cfg.get("max_chars", 50_000_000)
    args.shuffle_buffer = args.shuffle_buffer if args.shuffle_buffer is not None else cfg.get("shuffle_buffer", 50_000)
    args.seed = args.seed if args.seed is not None else cfg.get("seed", 1337)

    if args.c4_weight + args.wiki_weight <= 0:
        raise SystemExit("c4_weight + wiki_weight must be > 0")

    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)
    ds = _load_streaming_mix(args.c4_weight, args.wiki_weight, args.seed, args.shuffle_buffer)

    corpus_path = os.path.join(args.output_dir, "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for text in _iter_texts(ds, args.min_chars, args.max_chars):
            if rng.random() < 0.001:
                f.flush()
            f.write(text + "\n")

    model_prefix = os.path.join(args.output_dir, "spm")
    spm.SentencePieceTrainer.Train(
        " ".join(
            [
                f"--input={corpus_path}",
                f"--model_prefix={model_prefix}",
                "--model_type=bpe",
                f"--vocab_size={args.vocab_size}",
                "--character_coverage=0.9995",
                "--byte_fallback=true",
                "--normalization_rule_name=identity",
                "--input_sentence_size=2000000",
                "--shuffle_input_sentence=true",
                "--unk_id=0",
                "--bos_id=1",
                "--eos_id=2",
                "--pad_id=3",
                "--hard_vocab_limit=false",
            ]
        )
    )

    meta = {
        "vocab_size": args.vocab_size,
        "c4_weight": args.c4_weight,
        "wiki_weight": args.wiki_weight,
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "seed": args.seed,
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(args.output_dir, "tokenizer_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    main()

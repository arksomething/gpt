#!/usr/bin/env python3
import argparse
import os
import random
import re
import sys

import numpy as np
import sentencepiece as spm


def _parse_dtype(name):
    name = (name or "uint16").lower()
    if name in {"uint16", "u2"}:
        return np.uint16
    if name in {"uint32", "u4"}:
        return np.uint32
    if name in {"int32", "i4"}:
        return np.int32
    raise SystemExit(f"Unsupported dtype: {name}")


def _load_sp(model_path):
    if not os.path.exists(model_path):
        raise SystemExit(f"Missing tokenizer model: {model_path}")
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def _parse_id_list(text):
    parts = re.split(r"[,\s]+", text.strip())
    ids = []
    for part in parts:
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError as exc:
            raise SystemExit(f"Invalid token id: {part}") from exc
    return ids


def _format_list(values, max_items):
    if max_items is None or len(values) <= max_items:
        return values
    return values[:max_items] + ["..."]


def _show_tokens(sp, ids, max_show_tokens, max_text_chars):
    pieces = [sp.id_to_piece(token_id) for token_id in ids]
    decoded = sp.decode(ids)
    if max_text_chars and len(decoded) > max_text_chars:
        decoded = decoded[:max_text_chars] + "..."

    ids_view = _format_list(ids, max_show_tokens)
    pieces_view = _format_list(pieces, max_show_tokens)
    print(f"length: {len(ids)}")
    print("ids:", ids_view)
    print("pieces:", pieces_view)
    print("decoded:", decoded)


def _load_memmap(path, dtype):
    if not os.path.exists(path):
        raise SystemExit(f"Missing data file: {path}")
    return np.memmap(path, dtype=dtype, mode="r")


def _sample_tokens(sp, data, rng, count, length, max_show_tokens, max_text_chars):
    if len(data) <= length:
        raise SystemExit("Sample length exceeds data length.")
    for idx in range(count):
        start = rng.randint(0, len(data) - length - 1)
        ids = data[start : start + length].tolist()
        print(f"\nsample {idx + 1}/{count} start={start} length={length}")
        _show_tokens(sp, ids, max_show_tokens, max_text_chars)


def _run_repl(sp, train_data, val_data, rng, max_show_tokens, max_text_chars):
    print("Tokenizer REPL. Commands:")
    print("  encode <text>")
    print("  decode <id,id,...>")
    print("  piece <id>")
    print("  sample [train|val] [length] [count]")
    print("  quit/exit")
    while True:
        try:
            line = input("tokenizer> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line in {"quit", "exit"}:
            break
        if line.startswith("encode "):
            text = line[len("encode ") :]
            ids = sp.encode(text, out_type=int)
            _show_tokens(sp, ids, max_show_tokens, max_text_chars)
            continue
        if line.startswith("decode "):
            ids = _parse_id_list(line[len("decode ") :])
            _show_tokens(sp, ids, max_show_tokens, max_text_chars)
            continue
        if line.startswith("piece "):
            token_id = int(line[len("piece ") :].strip())
            print(sp.id_to_piece(token_id))
            continue
        if line.startswith("sample"):
            parts = line.split()
            split = parts[1] if len(parts) > 1 else "train"
            length = int(parts[2]) if len(parts) > 2 else 128
            count = int(parts[3]) if len(parts) > 3 else 1
            data = train_data if split == "train" else val_data
            _sample_tokens(sp, data, rng, count, length, max_show_tokens, max_text_chars)
            continue
        print("Unknown command. Type 'encode', 'decode', 'piece', or 'sample'.")


def main():
    parser = argparse.ArgumentParser(description="Inspect tokenizer + tokenized data.")
    parser.add_argument("--tokenizer_model", default="tokenizer/spm.model")
    parser.add_argument("--train_bin", default="data/train.bin")
    parser.add_argument("--val_bin", default="data/val.bin")
    parser.add_argument("--dtype", default="uint16")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--encode", default=None)
    parser.add_argument("--decode", default=None)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--max_show_tokens", type=int, default=120)
    parser.add_argument("--max_text_chars", type=int, default=800)
    parser.add_argument("--repl", action="store_true")
    args = parser.parse_args()

    sp = _load_sp(args.tokenizer_model)
    dtype = _parse_dtype(args.dtype)
    train_data = _load_memmap(args.train_bin, dtype)
    val_data = _load_memmap(args.val_bin, dtype)
    rng = random.Random(args.seed)

    if args.encode is not None:
        ids = sp.encode(args.encode, out_type=int)
        _show_tokens(sp, ids, args.max_show_tokens, args.max_text_chars)
        return
    if args.decode is not None:
        ids = _parse_id_list(args.decode)
        _show_tokens(sp, ids, args.max_show_tokens, args.max_text_chars)
        return
    if args.sample:
        data = train_data if args.split == "train" else val_data
        _sample_tokens(sp, data, rng, args.count, args.length, args.max_show_tokens, args.max_text_chars)
        return

    _run_repl(sp, train_data, val_data, rng, args.max_show_tokens, args.max_text_chars)


if __name__ == "__main__":
    main()

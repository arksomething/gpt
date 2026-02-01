#!/usr/bin/env python3
"""
Train a SentencePiece BPE tokenizer from C4, Wikipedia, and Project Gutenberg.

Streams and filters text from all three sources, then trains a BPE tokenizer.
"""

import argparse
import io
import json
import os
import random
import tempfile
import time
from datetime import datetime
from typing import Iterator, Optional

import sentencepiece as spm
import yaml
from datasets import load_dataset
from tqdm import tqdm

from scripts.filters import (
    apply_wiki_filters,
    apply_gutenberg_filters,
)


def stream_gutenberg(seed: int = 42):
    """Gutenberg streaming disabled due to pg19 compatibility issues."""
    # Return empty iterator - effectively disables Gutenberg
    return iter([])


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stream_c4(seed: int = 42) -> Iterator[str]:
    """Stream text from C4 dataset."""
    ds = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=10000)
    for example in ds:
        yield example["text"]


def stream_wikipedia(seed: int = 42) -> Iterator[str]:
    """Stream text from Wikipedia dataset."""
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=10000)
    for example in ds:
        yield example["text"]



def filter_text_basic(
    text: str,
    source: str,
    min_chars: int = 200,
    max_chars: int = 100000,
) -> Optional[str]:
    """
    Apply minimal filtering for tokenizer training.
    We want the tokenizer to see diverse text patterns,
    so filtering is much lighter than for data prep.
    """
    # Basic length check only
    if len(text) < min_chars:
        return None
    
    # Truncate very long texts
    if len(text) > max_chars:
        text = text[:max_chars]

    # For Gutenberg, strip the boilerplate header/footer
    if source == "gutenberg":
        text, _, _ = apply_gutenberg_filters(text)
    
    # For Wikipedia, strip markup but don't filter
    if source == "wiki":
        text, _, _ = apply_wiki_filters(text)

    # Final length check after transformations
    if len(text) < min_chars:
        return None

    return text


def interleave_and_filter(
    c4_iter: Iterator[str],
    wiki_iter: Iterator[str],
    gutenberg_iter: Iterator[str],
    c4_weight: float,
    wiki_weight: float,
    gutenberg_weight: float,
    min_chars: int,
    max_chars: int,
    seed: int = 42,
) -> Iterator[str]:
    """
    Interleave and filter text from all sources.
    Yields filtered text strings.
    """
    rng = random.Random(seed)

    # Normalize weights
    total = c4_weight + wiki_weight + gutenberg_weight
    c4_norm = c4_weight / total
    wiki_norm = wiki_weight / total
    gutenberg_norm = gutenberg_weight / total

    c4_exhausted = False
    wiki_exhausted = False
    gutenberg_exhausted = False
    
    docs_yielded = 0

    while True:
        if c4_exhausted and wiki_exhausted and gutenberg_exhausted:
            break

        # Calculate available weight
        available_weight = 0.0
        if not c4_exhausted:
            available_weight += c4_norm
        if not wiki_exhausted:
            available_weight += wiki_norm
        if not gutenberg_exhausted:
            available_weight += gutenberg_norm

        if available_weight == 0:
            break

        # Renormalize
        roll = rng.random()
        cumulative = 0.0
        selected = None

        if not c4_exhausted:
            cumulative += c4_norm / available_weight
            if roll < cumulative:
                selected = "c4"
        if selected is None and not wiki_exhausted:
            cumulative += wiki_norm / available_weight
            if roll < cumulative:
                selected = "wiki"
        if selected is None and not gutenberg_exhausted:
            selected = "gutenberg"

        # Get next item
        try:
            if selected == "c4":
                text = next(c4_iter)
                filtered = filter_text_basic(text, "c4", min_chars, max_chars)
                if filtered:
                    docs_yielded += 1
                    yield filtered
            elif selected == "wiki":
                text = next(wiki_iter)
                filtered = filter_text_basic(text, "wiki", min_chars, max_chars)
                if filtered:
                    docs_yielded += 1
                    yield filtered
            elif selected == "gutenberg":
                text = next(gutenberg_iter)
                filtered = filter_text_basic(text, "gutenberg", min_chars, max_chars)
                if filtered:
                    docs_yielded += 1
                    yield filtered
        except StopIteration:
            if selected == "c4":
                c4_exhausted = True
                print(f"[tokenizer] C4 source exhausted after {docs_yielded} total docs")
            elif selected == "wiki":
                wiki_exhausted = True
                print(f"[tokenizer] Wikipedia source exhausted after {docs_yielded} total docs")
            elif selected == "gutenberg":
                gutenberg_exhausted = True
                print(f"[tokenizer] Gutenberg source exhausted after {docs_yielded} total docs")


def collect_text_to_file(
    text_iter: Iterator[str],
    output_path: str,
    max_chars: int,
    log_interval: int = 30,
) -> int:
    """
    Collect text from iterator and write to file.
    Returns total characters written.
    """
    chars_written = 0
    docs_written = 0
    start_time = time.time()
    last_log_time = start_time

    with open(output_path, "w", encoding="utf-8") as f:
        pbar = tqdm(total=max_chars, desc="Collecting text", unit="char", unit_scale=True)
        
        for text in text_iter:
            if chars_written >= max_chars:
                break
            
            # Write each document as a line (SentencePiece expects one sentence per line)
            # Replace newlines with spaces to keep it on one line
            clean_text = text.replace("\n", " ").strip()
            if not clean_text:
                continue
                
            f.write(clean_text + "\n")
            chars_written += len(clean_text)
            docs_written += 1
            pbar.update(len(clean_text))

            # Log progress
            now = time.time()
            if now - last_log_time >= log_interval:
                elapsed = now - start_time
                cps = chars_written / elapsed if elapsed > 0 else 0
                print(
                    f"[collecting] {chars_written:,}/{max_chars:,} chars "
                    f"({chars_written/max_chars*100:.1f}%) | "
                    f"{docs_written:,} docs | "
                    f"{cps:,.0f} char/s"
                )
                last_log_time = now

        pbar.close()

    print(f"Collected {chars_written:,} chars from {docs_written:,} documents")
    return chars_written


def train_sentencepiece(
    input_file: str,
    output_dir: str,
    vocab_size: int,
):
    """
    Train SentencePiece BPE tokenizer from text file.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_prefix = os.path.join(output_dir, "spm")

    print(f"Training SentencePiece tokenizer...")
    print(f"  Input: {input_file}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Output: {model_prefix}.model")
    print()

    # Train from file
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        num_threads=os.cpu_count() or 4,
        # Special tokens
        pad_id=3,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        # Training parameters
        max_sentence_length=16384,
        shuffle_input_sentence=True,
        input_sentence_size=5000000,  # Use up to 5M sentences
        # Normalization
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        add_dummy_prefix=True,
        # Byte fallback for OOV
        byte_fallback=True,
    )

    print(f"\nTokenizer training complete!")
    return model_prefix + ".model", model_prefix + ".vocab"


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece BPE tokenizer.")
    parser.add_argument("--config", default="configs/train.yaml", help="Config file")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=None, help="Vocabulary size")
    parser.add_argument("--c4_weight", type=float, default=None, help="C4 weight")
    parser.add_argument("--wiki_weight", type=float, default=None, help="Wikipedia weight")
    parser.add_argument("--gutenberg_weight", type=float, default=None, help="Gutenberg weight")
    parser.add_argument("--min_chars", type=int, default=None, help="Min chars per document")
    parser.add_argument("--max_chars", type=int, default=None, help="Max chars for training")
    parser.add_argument("--shuffle_buffer", type=int, default=None, help="Shuffle buffer size")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    # Load config
    cfg = load_yaml(args.config)
    tok_cfg = cfg.get("tokenizer_training", {})
    data_prep_cfg = cfg.get("data_prep", {})

    # Resolve parameters (CLI overrides config)
    output_dir = args.output_dir or tok_cfg.get("output_dir", "tokenizer")
    vocab_size = args.vocab_size or tok_cfg.get("vocab_size", 32000)
    c4_weight = args.c4_weight if args.c4_weight is not None else data_prep_cfg.get("c4_weight", 0.30)
    wiki_weight = args.wiki_weight if args.wiki_weight is not None else data_prep_cfg.get("wiki_weight", 0.45)
    gutenberg_weight = (
        args.gutenberg_weight if args.gutenberg_weight is not None else data_prep_cfg.get("gutenberg_weight", 0.25)
    )
    min_chars = args.min_chars or tok_cfg.get("min_chars", 200)
    max_chars = args.max_chars or tok_cfg.get("max_chars", 50_000_000)
    seed = args.seed if args.seed is not None else tok_cfg.get("seed", 1337)

    print("SentencePiece Tokenizer Training")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Source weights: C4={c4_weight}, Wiki={wiki_weight}, Gutenberg={gutenberg_weight}")
    print(f"Max chars for training: {max_chars:,}")
    print(f"Seed: {seed}")
    print()

    # Create source iterators
    print("Initializing data streams...")
    c4_iter = stream_c4(seed)
    wiki_iter = stream_wikipedia(seed)
    gutenberg_iter = stream_gutenberg(seed)

    # Create interleaved and filtered iterator
    text_iter = interleave_and_filter(
        c4_iter,
        wiki_iter,
        gutenberg_iter,
        c4_weight,
        wiki_weight,
        gutenberg_weight,
        min_chars=min_chars,
        max_chars=100000,  # Per-document max
        seed=seed,
    )

    # First collect text to a temporary file
    temp_file = os.path.join(output_dir, "tokenizer_train_text.txt")
    
    print("Phase 1: Collecting training text...")
    start_time = time.time()
    total_chars = collect_text_to_file(text_iter, temp_file, max_chars)
    collect_elapsed = time.time() - start_time
    print(f"Text collection complete in {collect_elapsed/60:.1f} minutes")
    print()

    # Train tokenizer on the collected text
    print("Phase 2: Training tokenizer...")
    train_start = time.time()
    model_path, vocab_path = train_sentencepiece(
        temp_file,
        output_dir,
        vocab_size,
    )
    elapsed = time.time() - start_time
    
    # Clean up temp file
    if os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"Cleaned up temporary file: {temp_file}")

    # Save metadata
    meta = {
        "vocab_size": vocab_size,
        "c4_weight": c4_weight,
        "wiki_weight": wiki_weight,
        "gutenberg_weight": gutenberg_weight,
        "min_chars": min_chars,
        "max_chars": max_chars,
        "seed": seed,
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }
    meta_path = os.path.join(output_dir, "tokenizer_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nTraining complete in {elapsed/60:.1f} minutes")
    print(f"Output files:")
    print(f"  {model_path}")
    print(f"  {vocab_path}")
    print(f"  {meta_path}")

    # Verify the tokenizer
    print("\nVerifying tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    test_text = "The quick brown fox jumps over the lazy dog."
    tokens = sp.encode(test_text, out_type=str)
    print(f"  Test: {test_text!r}")
    print(f"  Tokens: {tokens}")
    print(f"  Token IDs: {sp.encode(test_text, out_type=int)}")
    print(f"  Decoded: {sp.decode(sp.encode(test_text, out_type=int))!r}")


if __name__ == "__main__":
    main()

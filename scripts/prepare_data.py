#!/usr/bin/env python3
"""
Prepare memmapped token data from C4, Wikipedia, and Project Gutenberg.

Features:
- Streams data from HuggingFace datasets
- Applies comprehensive filtering pipeline
- Deduplicates with fast hash-based detection
- Tokenizes with SentencePiece
- Writes to memory-mapped binary files
- Checkpointing and resume support
- Screen/tmux auto-launch for background processing
- File logging

Usage:
  # Basic usage
  prepare-data --train_tokens 1000000000 --val_tokens 10000000
  
  # Run in background with screen
  prepare-data --train_tokens 1000000000 --launch_screen
  
  # Resume from checkpoint
  prepare-data --resume
"""

import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, List, Optional

import numpy as np
import sentencepiece as spm
import yaml
from datasets import load_dataset
from tqdm import tqdm

from scripts.filters import (
    FilterStats,
    MinHashDeduplicator,
    apply_c4_filters,
    apply_global_filters,
    apply_gutenberg_filters,
    apply_wiki_filters,
    chunk_text,
)


# =============================================================================
# Logging and Screen Management
# =============================================================================


class Tee:
    """Duplicate stdout/stderr to a log file."""

    def __init__(self, log_path, stream_name="stdout"):
        self.log_path = log_path
        self.stream_name = stream_name
        self.original = getattr(sys, stream_name)
        self.log_file = open(log_path, "a", buffering=1, encoding="utf-8")
        setattr(sys, stream_name, self)

    def write(self, data):
        self.original.write(data)
        self.log_file.write(data)

    def flush(self):
        self.original.flush()
        self.log_file.flush()

    def close(self):
        setattr(sys, self.stream_name, self.original)
        self.log_file.close()


def setup_file_logging(log_dir: str, enabled: bool = True):
    """Set up file logging. Returns cleanup function."""
    if not enabled:
        return lambda: None

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"prepare_data_{time.strftime('%Y%m%d_%H%M%S')}.log")

    stdout_tee = Tee(log_path, "stdout")
    stderr_tee = Tee(log_path, "stderr")

    print(f"[logging] Writing to {log_path}")

    def cleanup():
        stdout_tee.close()
        stderr_tee.close()

    return cleanup


def maybe_launch_screen(enabled: bool, session_name: Optional[str] = None) -> bool:
    """Launch in screen/tmux if requested and not already in one."""
    if not enabled:
        return False
    if os.environ.get("STY") or os.environ.get("TMUX"):
        return False  # Already in screen/tmux

    session = session_name or f"prepare-data-{time.strftime('%Y%m%d-%H%M%S')}"
    command = [sys.executable, "-u", os.path.abspath(__file__), *sys.argv[1:]]
    # Remove --launch_screen to avoid infinite recursion
    command = [c for c in command if c != "--launch_screen"]

    if shutil.which("screen") is not None:
        subprocess.check_call(["screen", "-dmS", session, *command])
        print(f"Started screen session '{session}'. Attach with: screen -r {session}")
        return True

    if shutil.which("tmux") is not None:
        subprocess.check_call(["tmux", "new-session", "-d", "-s", session, *command])
        print(f"Started tmux session '{session}'. Attach with: tmux attach -t {session}")
        return True

    print("screen/tmux not found; running in foreground.")
    return False


# =============================================================================
# Checkpointing
# =============================================================================


@dataclass
class Checkpoint:
    """Checkpoint state for resuming."""
    
    phase: str  # "train" or "val"
    tokens_written: int
    docs_processed: int
    c4_docs: int
    wiki_docs: int
    gutenberg_docs: int
    dedup_rejects: int
    elapsed_seconds: float
    timestamp: str
    
    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "tokens_written": self.tokens_written,
            "docs_processed": self.docs_processed,
            "c4_docs": self.c4_docs,
            "wiki_docs": self.wiki_docs,
            "gutenberg_docs": self.gutenberg_docs,
            "dedup_rejects": self.dedup_rejects,
            "elapsed_seconds": self.elapsed_seconds,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Checkpoint":
        return cls(
            phase=d["phase"],
            tokens_written=d["tokens_written"],
            docs_processed=d["docs_processed"],
            c4_docs=d["c4_docs"],
            wiki_docs=d["wiki_docs"],
            gutenberg_docs=d["gutenberg_docs"],
            dedup_rejects=d["dedup_rejects"],
            elapsed_seconds=d["elapsed_seconds"],
            timestamp=d["timestamp"],
        )


def load_checkpoint(checkpoint_path: str) -> Optional[Checkpoint]:
    """Load checkpoint if exists."""
    if not os.path.exists(checkpoint_path):
        return None
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Checkpoint.from_dict(data)
    except (json.JSONDecodeError, KeyError):
        return None


def save_checkpoint(checkpoint: Checkpoint, checkpoint_path: str):
    """Save checkpoint to disk."""
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint.to_dict(), f, indent=2)


# =============================================================================
# Data Loading and Streaming
# =============================================================================


def stream_gutenberg(seed: int = 42) -> Iterator[str]:
    """Gutenberg streaming disabled due to pg19 compatibility issues."""
    return iter([])


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class FilterConfig:
    """Configuration for filtering parameters."""

    # Global filters
    min_chars: int = 300
    max_chars: int = 80000
    min_alpha_ratio: float = 0.65
    max_repeated_chars: int = 6
    max_weird_ratio: float = 0.01
    max_short_line_ratio: float = 0.30
    max_caps_ratio: float = 0.20

    # C4-specific
    c4_min_alpha_ratio: float = 0.70
    c4_max_punct_ratio: float = 0.20
    c4_min_entropy: float = 3.5
    c4_max_entropy: float = 5.6

    # Wikipedia-specific
    wiki_max_list_ratio: float = 0.30

    # Gutenberg-specific
    gutenberg_filter_poetry: bool = False

    # Deduplication
    dedup_enabled: bool = True
    dedup_threshold: float = 0.85
    dedup_num_perm: int = 128

    # Chunking
    max_chunk_chars: int = 8000
    min_chunk_chars: int = 500

    @classmethod
    def from_dict(cls, d: dict) -> "FilterConfig":
        """Create from config dict, using defaults for missing keys."""
        return cls(
            min_chars=d.get("min_chars", 300),
            max_chars=d.get("max_chars", 80000),
            min_alpha_ratio=d.get("min_alpha_ratio", 0.65),
            max_repeated_chars=d.get("max_repeated_chars", 6),
            max_weird_ratio=d.get("max_weird_ratio", 0.01),
            max_short_line_ratio=d.get("max_short_line_ratio", 0.30),
            max_caps_ratio=d.get("max_caps_ratio", 0.20),
            c4_min_alpha_ratio=d.get("c4_min_alpha_ratio", 0.70),
            c4_max_punct_ratio=d.get("c4_max_punct_ratio", 0.20),
            c4_min_entropy=d.get("c4_min_entropy", 3.5),
            c4_max_entropy=d.get("c4_max_entropy", 5.6),
            wiki_max_list_ratio=d.get("wiki_max_list_ratio", 0.30),
            gutenberg_filter_poetry=d.get("gutenberg_filter_poetry", False),
            dedup_enabled=d.get("dedup_enabled", True),
            dedup_threshold=d.get("dedup_similarity_threshold", 0.85),
            dedup_num_perm=d.get("dedup_num_perm", 128),
            max_chunk_chars=d.get("max_chunk_chars", 8000),
            min_chunk_chars=d.get("min_chunk_chars", 500),
        )


class ShuffleBuffer:
    """Buffer for streaming shuffle."""

    def __init__(self, size: int, seed: int = 42):
        self.size = size
        self.buffer: deque = deque(maxlen=size)
        self.rng = random.Random(seed)

    def add_and_maybe_yield(self, item) -> Optional[any]:
        """Add item to buffer, yield random item if buffer is full."""
        if len(self.buffer) >= self.size:
            idx = self.rng.randint(0, len(self.buffer) - 1)
            result = self.buffer[idx]
            self.buffer[idx] = item
            return result
        else:
            self.buffer.append(item)
            return None

    def flush(self) -> List:
        """Flush remaining items in random order."""
        items = list(self.buffer)
        self.rng.shuffle(items)
        self.buffer.clear()
        return items


def stream_c4(seed: int = 42) -> Iterator[str]:
    """Stream text from C4 dataset."""
    print("[c4] Loading dataset...", flush=True)
    ds = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True,
    )
    print("[c4] Starting iteration (no shuffle)...", flush=True)
    count = 0
    for example in ds:
        yield example["text"]
        count += 1
        if count == 1:
            print(f"[c4] First document yielded", flush=True)
        elif count % 5000 == 0:
            print(f"[c4] {count:,} docs", flush=True)


def stream_wikipedia(seed: int = 42) -> Iterator[str]:
    """Stream text from Wikipedia dataset."""
    print("[wiki] Loading dataset...", flush=True)
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )
    print("[wiki] Starting iteration (no shuffle)...", flush=True)
    count = 0
    for example in ds:
        yield example["text"]
        count += 1
        if count == 1:
            print(f"[wiki] First document yielded", flush=True)
        elif count % 5000 == 0:
            print(f"[wiki] {count:,} docs", flush=True)


# =============================================================================
# Filtering
# =============================================================================


def filter_and_transform_c4(
    text: str,
    filter_cfg: FilterConfig,
    stats: FilterStats,
) -> List[str]:
    """Apply C4-specific and global filters, return chunks that pass."""
    passed, reason = apply_c4_filters(
        text,
        min_alpha_ratio=filter_cfg.c4_min_alpha_ratio,
        max_punct_ratio=filter_cfg.c4_max_punct_ratio,
        min_entropy=filter_cfg.c4_min_entropy,
        max_entropy=filter_cfg.c4_max_entropy,
    )
    if not passed:
        stats.record_reject(f"c4_{reason}")
        return []

    passed, reason = apply_global_filters(
        text,
        min_chars=filter_cfg.min_chars,
        max_chars=filter_cfg.max_chars,
        min_alpha_ratio=filter_cfg.min_alpha_ratio,
        max_repeat=filter_cfg.max_repeated_chars,
        max_weird_ratio=filter_cfg.max_weird_ratio,
        max_short_line_ratio=filter_cfg.max_short_line_ratio,
        max_caps_ratio=filter_cfg.max_caps_ratio,
    )
    if not passed:
        stats.record_reject(f"c4_{reason}")
        return []

    chunks = chunk_text(text, filter_cfg.max_chunk_chars, filter_cfg.min_chunk_chars)
    if chunks:
        stats.record_pass()
    return chunks


def filter_and_transform_wiki(
    text: str,
    filter_cfg: FilterConfig,
    stats: FilterStats,
) -> List[str]:
    """Apply Wikipedia-specific and global filters, return chunks that pass."""
    text, passed, reason = apply_wiki_filters(text, filter_cfg.wiki_max_list_ratio)
    if not passed:
        stats.record_reject(f"wiki_{reason}")
        return []

    passed, reason = apply_global_filters(
        text,
        min_chars=filter_cfg.min_chars,
        max_chars=filter_cfg.max_chars,
        min_alpha_ratio=filter_cfg.min_alpha_ratio,
        max_repeat=filter_cfg.max_repeated_chars,
        max_weird_ratio=filter_cfg.max_weird_ratio,
        max_short_line_ratio=filter_cfg.max_short_line_ratio,
        max_caps_ratio=filter_cfg.max_caps_ratio,
    )
    if not passed:
        stats.record_reject(f"wiki_{reason}")
        return []

    chunks = chunk_text(text, filter_cfg.max_chunk_chars, filter_cfg.min_chunk_chars)
    if chunks:
        stats.record_pass()
    return chunks


def filter_and_transform_gutenberg(
    text: str,
    filter_cfg: FilterConfig,
    stats: FilterStats,
) -> List[str]:
    """Apply Gutenberg-specific and global filters, return chunks that pass."""
    text, passed, reason = apply_gutenberg_filters(
        text,
        filter_poetry=filter_cfg.gutenberg_filter_poetry,
    )
    if not passed:
        stats.record_reject(f"gutenberg_{reason}")
        return []

    passed, reason = apply_global_filters(
        text,
        min_chars=filter_cfg.min_chars,
        max_chars=filter_cfg.max_chars,
        min_alpha_ratio=filter_cfg.min_alpha_ratio,
        max_repeat=filter_cfg.max_repeated_chars,
        max_weird_ratio=filter_cfg.max_weird_ratio,
        max_short_line_ratio=filter_cfg.max_short_line_ratio,
        max_caps_ratio=filter_cfg.max_caps_ratio,
    )
    if not passed:
        stats.record_reject(f"gutenberg_{reason}")
        return []

    chunks = chunk_text(text, filter_cfg.max_chunk_chars, filter_cfg.min_chunk_chars)
    if chunks:
        stats.record_pass()
    return chunks


# =============================================================================
# Source Interleaving
# =============================================================================


def interleave_sources(
    c4_iter: Iterator[str],
    wiki_iter: Iterator[str],
    gutenberg_iter: Iterator[str],
    c4_weight: float,
    wiki_weight: float,
    gutenberg_weight: float,
    seed: int = 42,
) -> Iterator[tuple]:
    """Interleave sources according to weights. Yields (source_name, text) tuples."""
    rng = random.Random(seed)

    total = c4_weight + wiki_weight + gutenberg_weight
    c4_exhausted = False
    wiki_exhausted = False
    gutenberg_exhausted = False

    while True:
        if c4_exhausted and wiki_exhausted and gutenberg_exhausted:
            break

        available_weight = 0.0
        if not c4_exhausted:
            available_weight += c4_weight
        if not wiki_exhausted:
            available_weight += wiki_weight
        if not gutenberg_exhausted:
            available_weight += gutenberg_weight

        if available_weight == 0:
            break

        roll = rng.random() * available_weight
        cumulative = 0.0
        selected = None

        if not c4_exhausted:
            cumulative += c4_weight
            if roll < cumulative:
                selected = "c4"
        if selected is None and not wiki_exhausted:
            cumulative += wiki_weight
            if roll < cumulative:
                selected = "wiki"
        if selected is None and not gutenberg_exhausted:
            selected = "gutenberg"

        try:
            if selected == "c4":
                text = next(c4_iter)
                yield ("c4", text)
            elif selected == "wiki":
                text = next(wiki_iter)
                yield ("wiki", text)
            elif selected == "gutenberg":
                text = next(gutenberg_iter)
                yield ("gutenberg", text)
        except StopIteration:
            if selected == "c4":
                c4_exhausted = True
            elif selected == "wiki":
                wiki_exhausted = True
            elif selected == "gutenberg":
                gutenberg_exhausted = True


# =============================================================================
# Main Processing
# =============================================================================


class GracefulInterrupt:
    """Handle SIGINT/SIGTERM gracefully for checkpointing."""
    
    def __init__(self):
        self.interrupted = False
        self._original_sigint = None
        self._original_sigterm = None
    
    def __enter__(self):
        self._original_sigint = signal.signal(signal.SIGINT, self._handler)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._handler)
        return self
    
    def __exit__(self, *args):
        signal.signal(signal.SIGINT, self._original_sigint)
        signal.signal(signal.SIGTERM, self._original_sigterm)
    
    def _handler(self, signum, frame):
        print("\n[!] Interrupt received. Finishing current document and saving checkpoint...")
        self.interrupted = True


def write_tokens_to_memmap(
    tokens_iter: Iterator[List[int]],
    output_path: str,
    target_tokens: int,
    dtype: np.dtype,
    eos_token_id: int,
    checkpoint_path: str,
    checkpoint_interval: int = 60,
    log_interval: int = 30,
    stats: Optional[dict] = None,
    interrupt_handler: Optional[GracefulInterrupt] = None,
) -> tuple:
    """
    Write tokenized documents to memory-mapped file.
    Returns (actual_tokens, was_interrupted).
    """
    buffer_size = target_tokens + target_tokens // 10
    mmap = np.memmap(output_path, dtype=dtype, mode="w+", shape=(buffer_size,))

    position = 0
    doc_count = 0
    last_log_time = time.time()
    last_checkpoint_time = time.time()
    start_time = time.time()
    was_interrupted = False

    pbar = tqdm(total=target_tokens, desc="Tokens", unit="tok", unit_scale=True)

    try:
        for tokens in tokens_iter:
            if position >= target_tokens:
                break
            
            # Check for interrupt
            if interrupt_handler and interrupt_handler.interrupted:
                was_interrupted = True
                break

            doc_len = len(tokens)
            if position + doc_len + 1 > buffer_size:
                new_size = buffer_size + max(doc_len + 1, buffer_size // 2)
                mmap.flush()
                del mmap
                mmap = np.memmap(output_path, dtype=dtype, mode="r+", shape=(new_size,))
                buffer_size = new_size

            mmap[position : position + doc_len] = tokens
            position += doc_len
            mmap[position] = eos_token_id
            position += 1

            doc_count += 1
            pbar.update(doc_len + 1)

            now = time.time()
            
            # Progress logging
            if now - last_log_time >= log_interval:
                elapsed = now - start_time
                tps = position / elapsed if elapsed > 0 else 0
                eta = (target_tokens - position) / tps if tps > 0 else 0
                print(
                    f"[data] {position:,}/{target_tokens:,} tokens "
                    f"({position/target_tokens*100:.1f}%) | "
                    f"{doc_count:,} docs | "
                    f"{tps:,.0f} tok/s | "
                    f"ETA: {eta/60:.1f}min"
                )
                last_log_time = now
            
            # Checkpointing
            if now - last_checkpoint_time >= checkpoint_interval:
                checkpoint = Checkpoint(
                    phase="train",
                    tokens_written=position,
                    docs_processed=doc_count,
                    c4_docs=stats["c4"].total_docs if stats else 0,
                    wiki_docs=stats["wiki"].total_docs if stats else 0,
                    gutenberg_docs=stats["gutenberg"].total_docs if stats else 0,
                    dedup_rejects=stats.get("dedup_rejects", 0) if stats else 0,
                    elapsed_seconds=elapsed,
                    timestamp=datetime.utcnow().isoformat() + "Z",
                )
                save_checkpoint(checkpoint, checkpoint_path)
                last_checkpoint_time = now

    finally:
        pbar.close()

    # Truncate to actual size
    mmap.flush()
    del mmap
    mmap = np.memmap(output_path, dtype=dtype, mode="r+", shape=(position,))
    mmap.flush()
    del mmap

    return position, was_interrupted


def process_and_tokenize(
    source_iter: Iterator[tuple],
    filter_cfg: FilterConfig,
    tokenizer: spm.SentencePieceProcessor,
    deduplicator: Optional[MinHashDeduplicator],
    stats: dict,
    shuffle_buffer_size: int = 50000,
    seed: int = 42,
) -> Iterator[List[int]]:
    """Process documents: filter, deduplicate, tokenize."""
    shuffle_buffer = ShuffleBuffer(shuffle_buffer_size, seed)
    dedup_rejects = 0

    for source_name, text in source_iter:
        if source_name == "c4":
            chunks = filter_and_transform_c4(text, filter_cfg, stats["c4"])
        elif source_name == "wiki":
            chunks = filter_and_transform_wiki(text, filter_cfg, stats["wiki"])
        elif source_name == "gutenberg":
            chunks = filter_and_transform_gutenberg(text, filter_cfg, stats["gutenberg"])
        else:
            continue

        for chunk in chunks:
            if deduplicator is not None and deduplicator.is_duplicate(chunk):
                dedup_rejects += 1
                continue

            result = shuffle_buffer.add_and_maybe_yield(chunk)
            if result is not None:
                tokens = tokenizer.encode(result, out_type=int)
                yield tokens

    for chunk in shuffle_buffer.flush():
        tokens = tokenizer.encode(chunk, out_type=int)
        yield tokens

    stats["dedup_rejects"] = dedup_rejects


def main():
    parser = argparse.ArgumentParser(
        description="Prepare memmapped token data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s --train_tokens 1000000000 --val_tokens 10000000
  
  # Run in background with screen
  %(prog)s --train_tokens 1000000000 --launch_screen
  
  # Resume from checkpoint
  %(prog)s --resume
  
  # Quick test run
  %(prog)s --train_tokens 5000000 --val_tokens 100000 --overwrite
        """,
    )
    
    # Config
    parser.add_argument("--config", default="configs/train.yaml", help="Config file")
    
    # Data parameters
    parser.add_argument("--tokenizer_model", default=None, help="SentencePiece model")
    parser.add_argument("--out_dir", default=None, help="Output directory")
    parser.add_argument("--train_tokens", type=int, default=None, help="Target train tokens")
    parser.add_argument("--val_tokens", type=int, default=None, help="Target val tokens")
    parser.add_argument("--c4_weight", type=float, default=None, help="C4 sampling weight")
    parser.add_argument("--wiki_weight", type=float, default=None, help="Wikipedia sampling weight")
    parser.add_argument("--gutenberg_weight", type=float, default=None, help="Gutenberg sampling weight")
    parser.add_argument("--shuffle_buffer", type=int, default=None, help="Shuffle buffer size")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--log_interval", type=int, default=None, help="Seconds between logs")
    
    # Execution options
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--checkpoint_interval", type=int, default=60, help="Seconds between checkpoints")
    
    # Screen/logging options
    parser.add_argument("--launch_screen", action="store_true", help="Run in screen/tmux session")
    parser.add_argument("--screen_name", default=None, help="Name for screen/tmux session")
    parser.add_argument("--no_log", action="store_true", help="Disable file logging")
    
    args = parser.parse_args()

    # Maybe launch in screen/tmux
    if maybe_launch_screen(args.launch_screen, args.screen_name):
        return

    # Load config
    cfg = load_yaml(args.config)
    data_prep_cfg = cfg.get("data_prep", {})

    # Resolve parameters
    tokenizer_model = args.tokenizer_model or data_prep_cfg.get("tokenizer_model", "tokenizer/spm.model")
    out_dir = args.out_dir or data_prep_cfg.get("out_dir", "data")
    train_tokens = args.train_tokens or data_prep_cfg.get("train_tokens", 2_000_000_000)
    val_tokens = args.val_tokens or data_prep_cfg.get("val_tokens", 20_000_000)
    c4_weight = args.c4_weight if args.c4_weight is not None else data_prep_cfg.get("c4_weight", 0.30)
    wiki_weight = args.wiki_weight if args.wiki_weight is not None else data_prep_cfg.get("wiki_weight", 0.45)
    gutenberg_weight = (
        args.gutenberg_weight if args.gutenberg_weight is not None else data_prep_cfg.get("gutenberg_weight", 0.25)
    )
    shuffle_buffer = args.shuffle_buffer or data_prep_cfg.get("shuffle_buffer", 50000)
    seed = args.seed if args.seed is not None else data_prep_cfg.get("seed", 1337)
    log_interval = args.log_interval or data_prep_cfg.get("log_interval", 30)

    filter_cfg = FilterConfig.from_dict(data_prep_cfg)

    # Paths
    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")
    meta_path = os.path.join(out_dir, "data_meta.json")
    checkpoint_path = os.path.join(out_dir, "checkpoint.json")
    log_dir = os.path.join(out_dir, "logs")

    os.makedirs(out_dir, exist_ok=True)

    # Set up logging
    cleanup_logging = setup_file_logging(log_dir, not args.no_log)

    try:
        # Check for resume
        checkpoint = None
        if args.resume:
            checkpoint = load_checkpoint(checkpoint_path)
            if checkpoint:
                print(f"[resume] Found checkpoint from {checkpoint.timestamp}")
                print(f"[resume] Phase: {checkpoint.phase}, tokens: {checkpoint.tokens_written:,}")
            else:
                print("[resume] No checkpoint found, starting fresh.")

        # Check output files
        if not args.overwrite and not args.resume:
            if os.path.exists(train_path) or os.path.exists(val_path):
                print(f"Output files already exist in {out_dir}. Use --overwrite to replace or --resume to continue.")
                sys.exit(1)

        # Load tokenizer
        print(f"Loading tokenizer from {tokenizer_model}")
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_model)
        vocab_size = sp.vocab_size()
        eos_token_id = 2

        dtype = np.uint16 if vocab_size < 65536 else np.uint32
        print(f"Vocab size: {vocab_size}, dtype: {dtype}")

        # Print configuration
        print(f"\nData preparation configuration:")
        print(f"  C4 weight: {c4_weight}")
        print(f"  Wikipedia weight: {wiki_weight}")
        print(f"  Gutenberg weight: {gutenberg_weight}")
        print(f"  Train tokens: {train_tokens:,}")
        print(f"  Val tokens: {val_tokens:,}")
        print(f"  Shuffle buffer: {shuffle_buffer:,}")
        print(f"  Seed: {seed}")
        print(f"  Dedup enabled: {filter_cfg.dedup_enabled}")
        print()

        # Initialize deduplicator
        deduplicator = None
        if filter_cfg.dedup_enabled:
            deduplicator = MinHashDeduplicator(
                num_perm=filter_cfg.dedup_num_perm,
                threshold=filter_cfg.dedup_threshold,
            )

        # Statistics tracking
        stats = {
            "c4": FilterStats(),
            "wiki": FilterStats(),
            "gutenberg": FilterStats(),
            "dedup_rejects": 0,
        }

        # Set up interrupt handler
        with GracefulInterrupt() as interrupt_handler:
            
            # =====================================================================
            # Process training data
            # =====================================================================
            print("=" * 60)
            print("Processing training data...")
            print("=" * 60)

            train_start = time.time()

            c4_iter = stream_c4(seed)
            wiki_iter = stream_wikipedia(seed)
            gutenberg_iter = stream_gutenberg(seed)

            source_iter = interleave_sources(
                c4_iter,
                wiki_iter,
                gutenberg_iter,
                c4_weight,
                wiki_weight,
                gutenberg_weight,
                seed,
            )

            tokens_iter = process_and_tokenize(
                source_iter,
                filter_cfg,
                sp,
                deduplicator,
                stats,
                shuffle_buffer,
                seed,
            )

            actual_train_tokens, was_interrupted = write_tokens_to_memmap(
                tokens_iter,
                train_path,
                train_tokens,
                dtype,
                eos_token_id,
                checkpoint_path,
                args.checkpoint_interval,
                log_interval,
                stats,
                interrupt_handler,
            )

            train_elapsed = time.time() - train_start
            print(f"\nTraining data complete: {actual_train_tokens:,} tokens in {train_elapsed/60:.1f} minutes")

            # Print filter statistics
            print("\nTraining filter statistics:")
            print("-" * 40)
            print("C4:")
            print(stats["c4"].report())
            print("\nWikipedia:")
            print(stats["wiki"].report())
            print("\nGutenberg:")
            print(stats["gutenberg"].report())
            print(f"\nDeduplication rejects: {stats['dedup_rejects']:,}")

            if was_interrupted:
                print("\n[!] Training interrupted. Checkpoint saved. Use --resume to continue.")
                return

            # =====================================================================
            # Process validation data
            # =====================================================================
            print("\n" + "=" * 60)
            print("Processing validation data...")
            print("=" * 60)

            val_start = time.time()

            val_stats = {
                "c4": FilterStats(),
                "wiki": FilterStats(),
                "gutenberg": FilterStats(),
                "dedup_rejects": 0,
            }

            val_seed = seed + 999
            c4_iter = stream_c4(val_seed)
            wiki_iter = stream_wikipedia(val_seed)
            gutenberg_iter = stream_gutenberg(val_seed)

            source_iter = interleave_sources(
                c4_iter,
                wiki_iter,
                gutenberg_iter,
                c4_weight,
                wiki_weight,
                gutenberg_weight,
                val_seed,
            )

            val_deduplicator = None
            if filter_cfg.dedup_enabled:
                val_deduplicator = MinHashDeduplicator(
                    num_perm=filter_cfg.dedup_num_perm,
                    threshold=filter_cfg.dedup_threshold,
                )

            tokens_iter = process_and_tokenize(
                source_iter,
                filter_cfg,
                sp,
                val_deduplicator,
                val_stats,
                shuffle_buffer // 10,
                val_seed,
            )

            actual_val_tokens, was_interrupted = write_tokens_to_memmap(
                tokens_iter,
                val_path,
                val_tokens,
                dtype,
                eos_token_id,
                checkpoint_path,
                args.checkpoint_interval,
                log_interval,
                val_stats,
                interrupt_handler,
            )

            val_elapsed = time.time() - val_start
            print(f"\nValidation data complete: {actual_val_tokens:,} tokens in {val_elapsed/60:.1f} minutes")

        # =====================================================================
        # Save metadata
        # =====================================================================
        meta = {
            "train_tokens": actual_train_tokens,
            "val_tokens": actual_val_tokens,
            "vocab_size": vocab_size,
            "dtype": str(dtype),
            "eos_token_id": eos_token_id,
            "c4_weight": c4_weight,
            "wiki_weight": wiki_weight,
            "gutenberg_weight": gutenberg_weight,
            "seed": seed,
            "filter_config": {
                "min_chars": filter_cfg.min_chars,
                "max_chars": filter_cfg.max_chars,
                "min_alpha_ratio": filter_cfg.min_alpha_ratio,
                "c4_min_alpha_ratio": filter_cfg.c4_min_alpha_ratio,
                "c4_max_punct_ratio": filter_cfg.c4_max_punct_ratio,
                "c4_min_entropy": filter_cfg.c4_min_entropy,
                "c4_max_entropy": filter_cfg.c4_max_entropy,
                "dedup_enabled": filter_cfg.dedup_enabled,
                "dedup_threshold": filter_cfg.dedup_threshold,
            },
            "train_stats": {
                "c4_total": stats["c4"].total_docs,
                "c4_passed": stats["c4"].passed_docs,
                "wiki_total": stats["wiki"].total_docs,
                "wiki_passed": stats["wiki"].passed_docs,
                "gutenberg_total": stats["gutenberg"].total_docs,
                "gutenberg_passed": stats["gutenberg"].passed_docs,
                "dedup_rejects": stats["dedup_rejects"],
            },
            "created_at": datetime.utcnow().isoformat() + "Z",
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"\nMetadata saved to {meta_path}")

        # Clean up checkpoint on success
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        total_elapsed = time.time() - train_start
        print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
        print(f"Output files:")
        print(f"  {train_path} ({actual_train_tokens:,} tokens)")
        print(f"  {val_path} ({actual_val_tokens:,} tokens)")
        print(f"  {meta_path}")

    finally:
        cleanup_logging()


if __name__ == "__main__":
    main()

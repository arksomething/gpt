#!/usr/bin/env python3
"""
Prepare memmapped token data from C4, Wikipedia, and FineWeb.

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
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterator, List, Optional

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
    fineweb_docs: int
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
            "fineweb_docs": self.fineweb_docs,
            # Backward-compatible alias for older checkpoints/metadata readers.
            "gutenberg_docs": self.fineweb_docs,
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
            fineweb_docs=d.get("fineweb_docs", d.get("gutenberg_docs", 0)),
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
    """Save checkpoint atomically to disk."""
    tmp_path = f"{checkpoint_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint.to_dict(), f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, checkpoint_path)


# =============================================================================
# Data Loading and Streaming
# =============================================================================


def _stream_with_retries(
    label: str,
    dataset_factory: Callable[[], Iterator[dict]],
    text_extractor: Callable[[dict], Optional[str]],
) -> Iterator[str]:
    """Yield text docs with retry+fast-forward on transient stream failures."""
    max_retries = int(os.environ.get("DATA_STREAM_MAX_RETRIES", "12") or 12)
    base_delay = float(os.environ.get("DATA_STREAM_RETRY_BASE_SEC", "1.0") or 1.0)

    seen_examples = 0
    yielded_docs = 0
    attempt = 0

    while True:
        ds = None
        iterator = None
        try:
            ds = dataset_factory()
            iterator = iter(ds)

            if seen_examples > 0:
                print(
                    f"[{label}] Recovering stream, fast-forwarding {seen_examples:,} examples...",
                    flush=True,
                )
                skipped = 0
                while skipped < seen_examples:
                    next(iterator)
                    skipped += 1
                    if skipped % 100000 == 0:
                        print(f"[{label}] Fast-forwarded {skipped:,}/{seen_examples:,}", flush=True)

            for example in iterator:
                seen_examples += 1
                text = text_extractor(example)
                if not text:
                    continue
                yielded_docs += 1
                attempt = 0
                if yielded_docs == 1:
                    print(f"[{label}] First document yielded", flush=True)
                elif yielded_docs % 5000 == 0:
                    print(f"[{label}] {yielded_docs:,} docs", flush=True)
                yield text
            return
        except StopIteration:
            return
        except Exception as exc:
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(
                    f"[{label}] Stream failed after {max_retries} retries."
                ) from exc
            delay = min(60.0, base_delay * (2 ** (attempt - 1)))
            print(
                f"[{label}] Stream error after {yielded_docs:,} docs ({seen_examples:,} examples): "
                f"{exc}. Retrying {attempt}/{max_retries} in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
        finally:
            close_iter = getattr(iterator, "close", None)
            if callable(close_iter):
                try:
                    close_iter()
                except Exception:
                    pass
            close_ds = getattr(ds, "close", None)
            if callable(close_ds):
                try:
                    close_ds()
                except Exception:
                    pass


def stream_fineweb(seed: int = 42) -> Iterator[str]:
    """Stream text from FineWeb-Edu."""
    repo = os.environ.get("FINEWEB_REPO", "HuggingFaceFW/fineweb-edu")
    config = os.environ.get("FINEWEB_CONFIG", "sample-10BT")
    split = os.environ.get("FINEWEB_SPLIT", "train")
    text_field = os.environ.get("FINEWEB_TEXT_FIELD", "text")
    shuffle_buffer = int(os.environ.get("FINEWEB_SHUFFLE_BUFFER", "0") or 0)

    def dataset_factory():
        print(
            f"[fineweb] Loading dataset repo={repo} config={config} split={split}...",
            flush=True,
        )
        if config:
            ds = load_dataset(repo, config, split=split, streaming=True)
        else:
            ds = load_dataset(repo, split=split, streaming=True)
        if shuffle_buffer > 1:
            ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
            print(f"[fineweb] Starting iteration (shuffle buffer={shuffle_buffer})...", flush=True)
        else:
            print("[fineweb] Starting iteration (no shuffle)...", flush=True)
        return ds

    def extract_text(example: dict) -> Optional[str]:
        text = example.get(text_field) if isinstance(example, dict) else None
        if isinstance(text, str) and text.strip():
            return text
        return None

    yield from _stream_with_retries("fineweb", dataset_factory, extract_text)


def stream_gutenberg(seed: int = 42) -> Iterator[str]:
    """Deprecated alias retained for backward compatibility."""
    print("[data] stream_gutenberg() is deprecated; using FineWeb stream instead.")
    yield from stream_fineweb(seed=seed)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Compute sha256 for a local file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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

    # FineWeb-specific (reserved for future source-specific filters)
    fineweb_filter_poetry: bool = False

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
            fineweb_filter_poetry=d.get(
                "fineweb_filter_poetry",
                d.get("gutenberg_filter_poetry", False),
            ),
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
    def dataset_factory():
        print("[c4] Loading dataset...", flush=True)
        ds = load_dataset(
            "allenai/c4",
            "en",
            split="train",
            streaming=True,
        )
        print("[c4] Starting iteration (no shuffle)...", flush=True)
        return ds

    def extract_text(example: dict) -> Optional[str]:
        text = example.get("text") if isinstance(example, dict) else None
        if isinstance(text, str) and text.strip():
            return text
        return None

    _ = seed
    yield from _stream_with_retries("c4", dataset_factory, extract_text)


def stream_wikipedia(seed: int = 42) -> Iterator[str]:
    """Stream text from Wikipedia dataset."""
    def dataset_factory():
        print("[wiki] Loading dataset...", flush=True)
        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=True,
        )
        print("[wiki] Starting iteration (no shuffle)...", flush=True)
        return ds

    def extract_text(example: dict) -> Optional[str]:
        text = example.get("text") if isinstance(example, dict) else None
        if isinstance(text, str) and text.strip():
            return text
        return None

    _ = seed
    yield from _stream_with_retries("wiki", dataset_factory, extract_text)


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
    else:
        stats.record_reject("chunk_too_small")
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
    else:
        stats.record_reject("chunk_too_small")
    return chunks


def filter_and_transform_fineweb(
    text: str,
    filter_cfg: FilterConfig,
    stats: FilterStats,
) -> List[str]:
    """Apply global filters to FineWeb and return chunks that pass."""
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
        stats.record_reject(f"fineweb_{reason}")
        return []

    chunks = chunk_text(text, filter_cfg.max_chunk_chars, filter_cfg.min_chunk_chars)
    if chunks:
        stats.record_pass()
    else:
        stats.record_reject("chunk_too_small")
    return chunks


# =============================================================================
# Source Interleaving
# =============================================================================


def interleave_sources(
    c4_iter: Iterator[str],
    wiki_iter: Iterator[str],
    fineweb_iter: Iterator[str],
    c4_weight: float,
    wiki_weight: float,
    fineweb_weight: float,
    seed: int = 42,
) -> Iterator[tuple]:
    """Interleave sources according to weights. Yields (source_name, text) tuples."""
    rng = random.Random(seed)

    c4_exhausted = c4_weight <= 0
    wiki_exhausted = wiki_weight <= 0
    fineweb_exhausted = fineweb_weight <= 0

    try:
        while True:
            if c4_exhausted and wiki_exhausted and fineweb_exhausted:
                break

            available_weight = 0.0
            if not c4_exhausted:
                available_weight += c4_weight
            if not wiki_exhausted:
                available_weight += wiki_weight
            if not fineweb_exhausted:
                available_weight += fineweb_weight

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
            if selected is None and not fineweb_exhausted:
                selected = "fineweb"

            try:
                if selected == "c4":
                    text = next(c4_iter)
                    yield ("c4", text)
                elif selected == "wiki":
                    text = next(wiki_iter)
                    yield ("wiki", text)
                elif selected == "fineweb":
                    text = next(fineweb_iter)
                    yield ("fineweb", text)
            except StopIteration:
                if selected == "c4":
                    c4_exhausted = True
                elif selected == "wiki":
                    wiki_exhausted = True
                elif selected == "fineweb":
                    fineweb_exhausted = True
    finally:
        for source_iter in (c4_iter, wiki_iter, fineweb_iter):
            close_iter = getattr(source_iter, "close", None)
            if callable(close_iter):
                try:
                    close_iter()
                except Exception:
                    pass


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


def count_tokens_in_file(path: str, dtype: np.dtype, cap_at: Optional[int] = None) -> int:
    """Return aligned token count for an existing binary file."""
    if not os.path.exists(path):
        return 0
    bytes_per_token = np.dtype(dtype).itemsize
    size = os.path.getsize(path)
    aligned = size - (size % bytes_per_token)
    if aligned != size:
        print(
            f"[resume] Truncating partial token bytes in {path}: "
            f"{size} -> {aligned} bytes",
            flush=True,
        )
        os.truncate(path, aligned)
        size = aligned

    tokens = size // bytes_per_token
    if cap_at is not None and tokens > cap_at:
        print(
            f"[resume] Existing file has {tokens:,} tokens > target {cap_at:,}; truncating.",
            flush=True,
        )
        os.truncate(path, cap_at * bytes_per_token)
        tokens = cap_at
    return int(tokens)


_TOKENIZER_TLS = threading.local()
_SHORT_DOC_UNK_TOKEN_LIMIT = 200


def _tokenize_chunk_worker(
    text: str,
    tokenizer_model_path: str,
    max_unk_ratio: float,
    short_doc_unk_limit: int,
) -> tuple[Optional[List[int]], bool]:
    """
    Tokenize a chunk in a worker thread.
    Returns (tokens_or_none, rejected_for_unk).
    """
    sp = getattr(_TOKENIZER_TLS, "sp", None)
    loaded_model = getattr(_TOKENIZER_TLS, "model_path", None)
    if sp is None or loaded_model != tokenizer_model_path:
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_model_path)
        _TOKENIZER_TLS.sp = sp
        _TOKENIZER_TLS.model_path = tokenizer_model_path
        _TOKENIZER_TLS.unk_id = sp.unk_id()

    unk_id = _TOKENIZER_TLS.unk_id
    tokens = sp.encode(text, out_type=int)
    if not tokens:
        return None, False

    unk_count = sum(1 for t in tokens if t == unk_id)
    if unk_count > 0 and unk_count / len(tokens) > max_unk_ratio:
        return None, True
    if len(tokens) < short_doc_unk_limit and unk_count > 0:
        return None, True
    return tokens, False


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
    phase: str = "train",
    resume_existing_tokens: int = 0,
) -> tuple:
    """
    Write tokenized documents to binary file with crash-safe append semantics.
    Returns (actual_tokens, was_interrupted).
    """
    bytes_per_token = np.dtype(dtype).itemsize
    mode = "ab" if resume_existing_tokens > 0 else "wb"
    position = resume_existing_tokens
    doc_count = 0
    last_log_time = time.time()
    last_checkpoint_time = time.time()
    start_time = time.time()
    was_interrupted = False
    replayed_tokens = 0

    pbar = tqdm(
        total=target_tokens,
        initial=min(position, target_tokens),
        desc="Tokens",
        unit="tok",
        unit_scale=True,
    )

    with open(output_path, mode) as f:
        try:
            if resume_existing_tokens > 0:
                print(
                    f"[resume] Replaying {resume_existing_tokens:,} existing tokens "
                    "to rebuild filter/dedup state...",
                    flush=True,
                )
                while replayed_tokens < resume_existing_tokens:
                    tokens = next(tokens_iter)
                    replayed_tokens += len(tokens) + 1
                    doc_count += 1
                    if replayed_tokens > resume_existing_tokens:
                        raise SystemExit(
                            "Resume mismatch: regenerated stream does not align with existing output. "
                            "Use --overwrite to restart cleanly."
                        )
                print(
                    f"[resume] Replay complete at {replayed_tokens:,} tokens.",
                    flush=True,
                )

            for tokens in tokens_iter:
                if position >= target_tokens:
                    break

                if interrupt_handler and interrupt_handler.interrupted:
                    was_interrupted = True
                    break

                doc_len = len(tokens)
                write_len = doc_len + 1
                if position + write_len > target_tokens:
                    break

                arr = np.empty(write_len, dtype=dtype)
                arr[:-1] = tokens
                arr[-1] = eos_token_id
                f.write(arr.tobytes(order="C"))
                position += write_len
                doc_count += 1
                pbar.update(write_len)

                now = time.time()
                if now - last_log_time >= log_interval:
                    elapsed = now - start_time
                    written_since_start = max(0, position - resume_existing_tokens)
                    tps = written_since_start / elapsed if elapsed > 0 else 0.0
                    eta = (target_tokens - position) / tps if tps > 0 else 0.0
                    print(
                        f"[data] {position:,}/{target_tokens:,} tokens "
                        f"({position/target_tokens*100:.1f}%) | "
                        f"{doc_count:,} docs | "
                        f"{tps:,.0f} tok/s | "
                        f"ETA: {eta/60:.1f}min"
                    )
                    last_log_time = now

                if now - last_checkpoint_time >= checkpoint_interval:
                    elapsed = now - start_time
                    checkpoint = Checkpoint(
                        phase=phase,
                        tokens_written=position,
                        docs_processed=doc_count,
                        c4_docs=stats["c4"].total_docs if stats else 0,
                        wiki_docs=stats["wiki"].total_docs if stats else 0,
                        fineweb_docs=stats["fineweb"].total_docs if stats else 0,
                        dedup_rejects=stats.get("dedup_rejects", 0) if stats else 0,
                        elapsed_seconds=elapsed,
                        timestamp=datetime.utcnow().isoformat() + "Z",
                    )
                    save_checkpoint(checkpoint, checkpoint_path)
                    f.flush()
                    os.fsync(f.fileno())
                    last_checkpoint_time = now
        finally:
            f.flush()
            os.fsync(f.fileno())
            pbar.close()

    # Ensure file length exactly matches token count.
    expected_bytes = position * bytes_per_token
    actual_bytes = os.path.getsize(output_path)
    if actual_bytes != expected_bytes:
        os.truncate(output_path, expected_bytes)

    if was_interrupted or position < target_tokens:
        elapsed = time.time() - start_time
        checkpoint = Checkpoint(
            phase=phase,
            tokens_written=position,
            docs_processed=doc_count,
            c4_docs=stats["c4"].total_docs if stats else 0,
            wiki_docs=stats["wiki"].total_docs if stats else 0,
            fineweb_docs=stats["fineweb"].total_docs if stats else 0,
            dedup_rejects=stats.get("dedup_rejects", 0) if stats else 0,
            elapsed_seconds=elapsed,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
        save_checkpoint(checkpoint, checkpoint_path)

    close_iter = getattr(tokens_iter, "close", None)
    if callable(close_iter):
        try:
            close_iter()
        except Exception:
            pass

    return position, was_interrupted


def process_and_tokenize(
    source_iter: Iterator[tuple],
    filter_cfg: FilterConfig,
    tokenizer: spm.SentencePieceProcessor,
    deduplicator: Optional[MinHashDeduplicator],
    stats: dict,
    shuffle_buffer_size: int = 50000,
    seed: int = 42,
    max_unk_ratio: float = 0.01,
    tokenizer_workers: int = 1,
    tokenizer_prefetch: int = 256,
    tokenizer_model_path: Optional[str] = None,
) -> Iterator[List[int]]:
    """Process documents: filter, deduplicate, tokenize."""
    shuffle_buffer = ShuffleBuffer(shuffle_buffer_size, seed)
    dedup_rejects = 0
    unk_rejects = 0
    unk_id = tokenizer.unk_id()  # Usually 0

    # For periodic logging
    last_log_time = time.time()
    log_interval = 10  # Log every 10 seconds
    docs_yielded = 0

    tokenizer_workers = max(1, int(tokenizer_workers))
    tokenizer_prefetch = max(1, int(tokenizer_prefetch))
    use_parallel_tokenization = (
        tokenizer_workers > 1
        and tokenizer_model_path is not None
        and tokenizer_model_path != ""
    )
    if use_parallel_tokenization:
        print(
            f"[tokenize] Parallel workers={tokenizer_workers}, prefetch={tokenizer_prefetch}",
            flush=True,
        )

    def tokenize_and_filter_local(text: str) -> Optional[List[int]]:
        """Tokenize and reject if too many unknown tokens."""
        nonlocal unk_rejects
        tokens = tokenizer.encode(text, out_type=int)
        if not tokens:
            return None
        # Count unknown tokens
        unk_count = sum(1 for t in tokens if t == unk_id)
        if unk_count > 0 and unk_count / len(tokens) > max_unk_ratio:
            unk_rejects += 1
            return None
        # Also reject if ANY unknown tokens in short docs
        if len(tokens) < 200 and unk_count > 0:
            unk_rejects += 1
            return None
        return tokens

    def log_filter_progress():
        """Log current filter statistics."""
        nonlocal last_log_time
        now = time.time()
        if now - last_log_time < log_interval:
            return
        last_log_time = now

        total_seen = stats["c4"].total_docs + stats["wiki"].total_docs + stats["fineweb"].total_docs
        c4_pass = stats["c4"].passed_docs
        wiki_pass = stats["wiki"].passed_docs
        fineweb_pass = stats["fineweb"].passed_docs

        c4_rate = (c4_pass / stats["c4"].total_docs * 100) if stats["c4"].total_docs > 0 else 0
        wiki_rate = (wiki_pass / stats["wiki"].total_docs * 100) if stats["wiki"].total_docs > 0 else 0
        fineweb_rate = (
            fineweb_pass / stats["fineweb"].total_docs * 100
            if stats["fineweb"].total_docs > 0
            else 0
        )

        print(
            f"[filter] Seen {total_seen:,} docs | "
            f"Passed: C4 {c4_pass:,} ({c4_rate:.1f}%), "
            f"Wiki {wiki_pass:,} ({wiki_rate:.1f}%), "
            f"FineWeb {fineweb_pass:,} ({fineweb_rate:.1f}%) | "
            f"Yielded: {docs_yielded:,} | "
            f"Dedup: {dedup_rejects:,} | UNK: {unk_rejects:,}",
            flush=True
        )

    pending_futures: deque[Future] = deque()
    executor: Optional[ThreadPoolExecutor] = None

    def submit_tokenization(chunk: str):
        if executor is None:
            return
        pending_futures.append(
            executor.submit(
                _tokenize_chunk_worker,
                chunk,
                tokenizer_model_path,
                max_unk_ratio,
                _SHORT_DOC_UNK_TOKEN_LIMIT,
            )
        )

    def drain_futures(block: bool) -> Iterator[List[int]]:
        nonlocal unk_rejects, docs_yielded
        while pending_futures:
            future = pending_futures[0]
            if not block and not future.done():
                break

            future = pending_futures.popleft()
            tokens, unk_rejected = future.result()
            if unk_rejected:
                unk_rejects += 1
            if tokens:
                docs_yielded += 1
                yield tokens

            if block:
                break

    try:
        if use_parallel_tokenization:
            executor = ThreadPoolExecutor(
                max_workers=tokenizer_workers,
                thread_name_prefix="tok",
            )

        for source_name, text in source_iter:
            if source_name == "c4":
                chunks = filter_and_transform_c4(text, filter_cfg, stats["c4"])
            elif source_name == "wiki":
                chunks = filter_and_transform_wiki(text, filter_cfg, stats["wiki"])
            elif source_name == "fineweb":
                chunks = filter_and_transform_fineweb(text, filter_cfg, stats["fineweb"])
            else:
                continue

            # Log progress periodically
            log_filter_progress()

            for chunk in chunks:
                if deduplicator is not None and deduplicator.is_duplicate(chunk):
                    dedup_rejects += 1
                    continue

                result = shuffle_buffer.add_and_maybe_yield(chunk)
                if result is None:
                    continue

                if use_parallel_tokenization:
                    submit_tokenization(result)
                    while len(pending_futures) >= tokenizer_prefetch:
                        yield from drain_futures(block=True)
                    yield from drain_futures(block=False)
                else:
                    tokens = tokenize_and_filter_local(result)
                    if tokens:
                        docs_yielded += 1
                        yield tokens

        for chunk in shuffle_buffer.flush():
            if use_parallel_tokenization:
                submit_tokenization(chunk)
                while len(pending_futures) >= tokenizer_prefetch:
                    yield from drain_futures(block=True)
                yield from drain_futures(block=False)
            else:
                tokens = tokenize_and_filter_local(chunk)
                if tokens:
                    docs_yielded += 1
                    yield tokens

        if use_parallel_tokenization:
            while pending_futures:
                yield from drain_futures(block=True)
    finally:
        close_source_iter = getattr(source_iter, "close", None)
        if callable(close_source_iter):
            try:
                close_source_iter()
            except Exception:
                pass
        if executor is not None:
            for future in pending_futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
        stats["dedup_rejects"] = dedup_rejects
        stats["unk_rejects"] = unk_rejects


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
    parser.add_argument("--fineweb_weight", type=float, default=None, help="FineWeb sampling weight")
    parser.add_argument(
        "--gutenberg_weight",
        type=float,
        default=None,
        help="Deprecated alias for --fineweb_weight.",
    )
    parser.add_argument("--shuffle_buffer", type=int, default=None, help="Shuffle buffer size")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--log_interval", type=int, default=None, help="Seconds between logs")
    parser.add_argument(
        "--tokenizer_workers",
        type=int,
        default=None,
        help="Tokenization worker threads (<=0 means auto).",
    )
    parser.add_argument(
        "--tokenizer_prefetch",
        type=int,
        default=None,
        help="Maximum queued tokenization tasks.",
    )
    
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
    fineweb_weight = (
        args.fineweb_weight if args.fineweb_weight is not None else data_prep_cfg.get("fineweb_weight")
    )
    if fineweb_weight is None:
        fineweb_weight = (
            args.gutenberg_weight
            if args.gutenberg_weight is not None
            else data_prep_cfg.get("gutenberg_weight", 0.0)
        )
        if args.gutenberg_weight is not None:
            print("[data] --gutenberg_weight is deprecated; treating it as --fineweb_weight.")
    elif args.gutenberg_weight is not None:
        print("[data] --gutenberg_weight ignored because --fineweb_weight is set.")

    if c4_weight <= 0 and wiki_weight <= 0 and fineweb_weight <= 0:
        raise SystemExit("At least one of c4_weight, wiki_weight, or fineweb_weight must be > 0.")
    shuffle_buffer = args.shuffle_buffer or data_prep_cfg.get("shuffle_buffer", 50000)
    seed = args.seed if args.seed is not None else data_prep_cfg.get("seed", 1337)
    log_interval = args.log_interval or data_prep_cfg.get("log_interval", 30)
    tokenizer_workers = (
        args.tokenizer_workers
        if args.tokenizer_workers is not None
        else data_prep_cfg.get("tokenizer_workers", 1)
    )
    tokenizer_workers = int(tokenizer_workers if tokenizer_workers is not None else 1)
    if tokenizer_workers <= 0:
        tokenizer_workers = max(1, min(os.cpu_count() or 1, 16))

    tokenizer_prefetch = (
        args.tokenizer_prefetch
        if args.tokenizer_prefetch is not None
        else data_prep_cfg.get("tokenizer_prefetch", tokenizer_workers * 32)
    )
    tokenizer_prefetch = max(1, int(tokenizer_prefetch))

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
        eos_token_id = sp.eos_id()
        if eos_token_id is None or eos_token_id < 0:
            raise SystemExit(
                "Tokenizer must define eos_id. Re-train tokenizer with eos_id set."
            )
        tokenizer_special_ids = {
            "unk": sp.unk_id(),
            "bos": sp.bos_id(),
            "eos": eos_token_id,
            "pad": sp.pad_id(),
        }
        tokenizer_sha256 = file_sha256(tokenizer_model)

        dtype = np.uint16 if vocab_size < 65536 else np.uint32
        print(f"Vocab size: {vocab_size}, dtype: {dtype}")

        existing_train_tokens = 0
        existing_val_tokens = 0
        if args.resume:
            existing_train_tokens = count_tokens_in_file(train_path, dtype, cap_at=train_tokens)
            existing_val_tokens = count_tokens_in_file(val_path, dtype, cap_at=val_tokens)
            print(
                f"[resume] Existing files: train={existing_train_tokens:,}/{train_tokens:,} "
                f"val={existing_val_tokens:,}/{val_tokens:,}",
                flush=True,
            )

        # Print configuration
        print(f"\nData preparation configuration:")
        print(f"  C4 weight: {c4_weight}")
        print(f"  Wikipedia weight: {wiki_weight}")
        print(f"  FineWeb weight: {fineweb_weight}")
        print(f"  Train tokens: {train_tokens:,}")
        print(f"  Val tokens: {val_tokens:,}")
        print(f"  Shuffle buffer: {shuffle_buffer:,}")
        print(f"  Seed: {seed}")
        print(f"  Dedup enabled: {filter_cfg.dedup_enabled}")
        print(f"  Tokenizer workers: {tokenizer_workers}")
        print(f"  Tokenizer prefetch: {tokenizer_prefetch}")
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
            "fineweb": FilterStats(),
            "dedup_rejects": 0,
        }

        # Set up interrupt handler
        with GracefulInterrupt() as interrupt_handler:
            overall_start = time.time()

            # =====================================================================
            # Process training data
            # =====================================================================
            print("=" * 60)
            print("Processing training data...")
            print("=" * 60)

            train_start = time.time()
            train_resume_tokens = existing_train_tokens if args.resume else 0
            if train_resume_tokens >= train_tokens:
                print(
                    f"[resume] Training data already complete "
                    f"({train_resume_tokens:,}/{train_tokens:,} tokens). Skipping.",
                    flush=True,
                )
                actual_train_tokens = train_tokens
                was_interrupted = False
            else:
                c4_iter = stream_c4(seed)
                wiki_iter = stream_wikipedia(seed)
                fineweb_iter = stream_fineweb(seed) if fineweb_weight > 0 else iter(())

                source_iter = interleave_sources(
                    c4_iter,
                    wiki_iter,
                    fineweb_iter,
                    c4_weight,
                    wiki_weight,
                    fineweb_weight,
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
                    tokenizer_workers=tokenizer_workers,
                    tokenizer_prefetch=tokenizer_prefetch,
                    tokenizer_model_path=tokenizer_model,
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
                    phase="train",
                    resume_existing_tokens=train_resume_tokens,
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
            print("\nFineWeb:")
            print(stats["fineweb"].report())
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
            val_resume_tokens = existing_val_tokens if args.resume else 0

            val_stats = {
                "c4": FilterStats(),
                "wiki": FilterStats(),
                "fineweb": FilterStats(),
                "dedup_rejects": 0,
            }
            if val_resume_tokens >= val_tokens:
                print(
                    f"[resume] Validation data already complete "
                    f"({val_resume_tokens:,}/{val_tokens:,} tokens). Skipping.",
                    flush=True,
                )
                actual_val_tokens = val_tokens
                was_interrupted = False
            else:
                val_seed = seed + 999
                c4_iter = stream_c4(val_seed)
                wiki_iter = stream_wikipedia(val_seed)
                fineweb_iter = stream_fineweb(val_seed) if fineweb_weight > 0 else iter(())

                source_iter = interleave_sources(
                    c4_iter,
                    wiki_iter,
                    fineweb_iter,
                    c4_weight,
                    wiki_weight,
                    fineweb_weight,
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
                    tokenizer_workers=tokenizer_workers,
                    tokenizer_prefetch=tokenizer_prefetch,
                    tokenizer_model_path=tokenizer_model,
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
                    phase="val",
                    resume_existing_tokens=val_resume_tokens,
                )

            val_elapsed = time.time() - val_start
            print(f"\nValidation data complete: {actual_val_tokens:,} tokens in {val_elapsed/60:.1f} minutes")
            if was_interrupted:
                print("\n[!] Validation interrupted. Checkpoint saved. Use --resume to continue.")
                return

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
            "fineweb_weight": fineweb_weight,
            # Backward-compatible alias.
            "gutenberg_weight": fineweb_weight,
            "seed": seed,
            "tokenizer_workers": tokenizer_workers,
            "tokenizer_prefetch": tokenizer_prefetch,
            "tokenizer_model": os.path.abspath(tokenizer_model),
            "tokenizer_sha256": tokenizer_sha256,
            "tokenizer_vocab_size": vocab_size,
            "tokenizer_special_ids": tokenizer_special_ids,
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
                "fineweb_total": stats["fineweb"].total_docs,
                "fineweb_passed": stats["fineweb"].passed_docs,
                # Backward-compatible aliases.
                "gutenberg_total": stats["fineweb"].total_docs,
                "gutenberg_passed": stats["fineweb"].passed_docs,
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

        total_elapsed = time.time() - overall_start
        print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
        print(f"Output files:")
        print(f"  {train_path} ({actual_train_tokens:,} tokens)")
        print(f"  {val_path} ({actual_val_tokens:,} tokens)")
        print(f"  {meta_path}")

    finally:
        cleanup_logging()


if __name__ == "__main__":
    main()

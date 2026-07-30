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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

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
    filter_by_length,
)
from scripts.canonical_writing import iter_canonical_manifest
from scripts.indexed_shards import (
    ResumableIndexedShardWriter,
    Source,
    content_hash,
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
    tokenizer_sha256: Optional[str] = None

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
            "tokenizer_sha256": self.tokenizer_sha256,
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
            tokenizer_sha256=d.get("tokenizer_sha256"),
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


@dataclass(frozen=True)
class PreparedDocument:
    """A tokenized document with provenance retained through shuffling."""

    tokens: List[int]
    source_id: str
    content_sha256: str
    metadata: dict


class ShuffleBuffer:
    """Buffer for streaming shuffle."""

    def __init__(self, size: int, seed: int = 42):
        self.size = max(1, int(size))
        self.buffer: deque = deque(maxlen=self.size)
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
# Gate 1 Mixture Sources
# =============================================================================

# Filter treatments applied to the bakeoff sources.
#   "global"       - the same global filter stack FineWeb gets.
#   "conversation" - global filters with a lower length floor: forum posts,
#                    IRC logs, transcripts and commit messages are legitimately
#                    short, so the 300-char floor would delete the source.
#   "code"         - length only (plus the pipeline-wide dedup). Alpha-ratio,
#                    weird-symbol, line-structure and boilerplate filters are
#                    prose heuristics that reject essentially all source code.
GLOBAL_TREATMENT = "global"
CONVERSATION_TREATMENT = "conversation"
CODE_TREATMENT = "code"

# Length floor for conversation-style sources.
CONVERSATION_MIN_CHARS = 200

DEFAULT_LICENSE_NOTE = "Record-level rights may vary; review upstream terms."
COMMON_PILE_LICENSE_NOTE = (
    "Common Pile openly licensed subset; per-record licenses vary, review upstream terms."
)


@dataclass(frozen=True)
class SourceSpec:
    """A streaming source added for the Gate 1 mixture bakeoff."""

    source_id: str
    name: str
    repo_id: str
    config: Optional[str] = None
    split: str = "train"
    text_field: str = "text"
    filter_treatment: str = GLOBAL_TREATMENT
    license_note: str = DEFAULT_LICENSE_NOTE

    @property
    def is_common_pile(self) -> bool:
        return self.repo_id.startswith("common-pile/")


def _common_pile(
    source_id: str,
    name: str,
    repo_id: str,
    filter_treatment: str = GLOBAL_TREATMENT,
) -> SourceSpec:
    """Register a Common Pile dataset: one shared schema, one factory."""
    return SourceSpec(
        source_id=source_id,
        name=name,
        repo_id=repo_id,
        config=None,
        split="train",
        text_field="text",
        filter_treatment=filter_treatment,
        license_note=COMMON_PILE_LICENSE_NOTE,
    )


# Order is display order only; sampling is weight-driven and name-sorted.
MIXTURE_SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(
        source_id="dclm",
        name="DCLM-baseline 1.0",
        repo_id="mlfoundations/dclm-baseline-1.0",
    ),
    SourceSpec(
        source_id="finepdfs",
        name="FinePDFs (English)",
        repo_id="HuggingFaceFW/finepdfs",
        config="eng_Latn",
    ),
    _common_pile(
        "stackexchange",
        "Common Pile StackExchange (filtered)",
        "common-pile/stackexchange_filtered",
        CONVERSATION_TREATMENT,
    ),
    _common_pile(
        "youtube",
        "Common Pile YouTube transcripts (filtered)",
        "common-pile/youtube_filtered",
        CONVERSATION_TREATMENT,
    ),
    _common_pile(
        "ubuntu_irc",
        "Common Pile Ubuntu IRC logs (filtered)",
        "common-pile/ubuntu_irc_filtered",
        CONVERSATION_TREATMENT,
    ),
    _common_pile(
        "github_archive",
        "Common Pile GitHub Archive (filtered)",
        "common-pile/github_archive_filtered",
        CONVERSATION_TREATMENT,
    ),
    _common_pile(
        "uk_hansard",
        "Common Pile UK Hansard (filtered)",
        "common-pile/uk_hansard_filtered",
        CONVERSATION_TREATMENT,
    ),
    SourceSpec(
        source_id="finemath",
        name="FineMath 4+",
        repo_id="HuggingFaceTB/finemath",
        config="finemath-4plus",
    ),
    _common_pile(
        "code",
        "Common Pile Stack v2 educational code (filtered)",
        "common-pile/stackv2_edu_filtered",
        CODE_TREATMENT,
    ),
    SourceSpec(
        source_id="cosmopedia",
        name="Cosmopedia v2 (SmolLM corpus)",
        repo_id="HuggingFaceTB/smollm-corpus",
        config="cosmopedia-v2",
    ),
    _common_pile(
        "wikiteam",
        "Common Pile WikiTeam wikis (filtered)",
        "common-pile/wikiteam_filtered",
    ),
    _common_pile(
        "libretexts",
        "Common Pile LibreTexts (filtered)",
        "common-pile/libretexts_filtered",
    ),
    _common_pile(
        "narrative",
        "Common Pile pre-1929 books (filtered)",
        "common-pile/pre_1929_books_filtered",
    ),
)

MIXTURE_SOURCES_BY_ID: Dict[str, SourceSpec] = {
    spec.source_id: spec for spec in MIXTURE_SOURCE_SPECS
}
MIXTURE_SOURCE_IDS: tuple[str, ...] = tuple(
    spec.source_id for spec in MIXTURE_SOURCE_SPECS
)


def stream_hf_text(
    label: str,
    repo_id: str,
    seed: int = 42,
    *,
    config: Optional[str] = None,
    split: str = "train",
    text_field: str = "text",
) -> Iterator[str]:
    """Stream a plain-text field from any streaming-capable HF dataset."""

    def dataset_factory():
        print(
            f"[{label}] Loading dataset repo={repo_id} config={config} split={split}...",
            flush=True,
        )
        if config:
            ds = load_dataset(repo_id, config, split=split, streaming=True)
        else:
            ds = load_dataset(repo_id, split=split, streaming=True)
        print(f"[{label}] Starting iteration (no shuffle)...", flush=True)
        return ds

    def extract_text(example: dict) -> Optional[str]:
        text = example.get(text_field) if isinstance(example, dict) else None
        if isinstance(text, str) and text.strip():
            return text
        return None

    _ = seed
    yield from _stream_with_retries(label, dataset_factory, extract_text)


def stream_common_pile(repo_id: str, seed: int = 42) -> Iterator[str]:
    """Stream any Common Pile dataset.

    Every Common Pile release shares one schema (a "text" field, single train
    split), so all of them are instances of this factory rather than
    copy-pasted per-source readers.
    """
    label = repo_id.split("/")[-1]
    if label.endswith("_filtered"):
        label = label[: -len("_filtered")]
    yield from stream_hf_text(
        label,
        repo_id,
        seed,
        config=None,
        split="train",
        text_field="text",
    )


def open_mixture_stream(spec: SourceSpec, seed: int = 42) -> Iterator[str]:
    """Open the streaming iterator for a registered mixture source."""
    if spec.is_common_pile:
        return stream_common_pile(spec.repo_id, seed)
    return stream_hf_text(
        spec.source_id,
        spec.repo_id,
        seed,
        config=spec.config,
        split=spec.split,
        text_field=spec.text_field,
    )


def resolve_mixture_weights(args, data_prep_cfg: dict) -> Dict[str, float]:
    """Resolve every mixture weight from CLI, then config, then 0.0."""
    weights: Dict[str, float] = {}
    for source_id in MIXTURE_SOURCE_IDS:
        override = getattr(args, f"{source_id}_weight", None)
        value = (
            override
            if override is not None
            else data_prep_cfg.get(f"{source_id}_weight", 0.0)
        )
        weights[source_id] = float(value or 0.0)
    return weights


def active_mixture_weights(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Only the mixture sources that are actually enabled."""
    return {
        source_id: float(weight)
        for source_id, weight in sorted((weights or {}).items())
        if float(weight) > 0
    }


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


def filter_and_transform_mixture(
    text: str,
    spec: SourceSpec,
    filter_cfg: FilterConfig,
    stats: FilterStats,
) -> List[str]:
    """Apply the registered filter treatment for a Gate 1 mixture source."""
    if spec.filter_treatment == CODE_TREATMENT:
        # Length + dedup only. Every other global filter is a prose heuristic
        # (alpha ratio, weird symbols, line structure) that rejects code.
        min_chars = filter_cfg.min_chars
        if not filter_by_length(text, min_chars, filter_cfg.max_chars):
            stats.record_reject(f"{spec.source_id}_G0_length")
            return []
    else:
        min_chars = (
            CONVERSATION_MIN_CHARS
            if spec.filter_treatment == CONVERSATION_TREATMENT
            else filter_cfg.min_chars
        )
        passed, reason = apply_global_filters(
            text,
            min_chars=min_chars,
            max_chars=filter_cfg.max_chars,
            min_alpha_ratio=filter_cfg.min_alpha_ratio,
            max_repeat=filter_cfg.max_repeated_chars,
            max_weird_ratio=filter_cfg.max_weird_ratio,
            max_short_line_ratio=filter_cfg.max_short_line_ratio,
            max_caps_ratio=filter_cfg.max_caps_ratio,
        )
        if not passed:
            stats.record_reject(f"{spec.source_id}_{reason}")
            return []

    # A relaxed length floor is meaningless if chunking then drops everything
    # under min_chunk_chars, so lower the chunk floor to match. Sources on the
    # plain global treatment keep the FineWeb chunking behaviour exactly.
    min_chunk_chars = filter_cfg.min_chunk_chars
    if spec.filter_treatment != GLOBAL_TREATMENT:
        min_chunk_chars = min(min_chunk_chars, min_chars)

    chunks = chunk_text(text, filter_cfg.max_chunk_chars, min_chunk_chars)
    if chunks:
        stats.record_pass()
    else:
        stats.record_reject("chunk_too_small")
    return chunks


def filter_and_transform_generic(
    text: str,
    source_id: str,
    filter_cfg: FilterConfig,
    stats: FilterStats,
) -> List[str]:
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
        stats.record_reject(f"{source_id}_{reason}")
        return []
    chunks = chunk_text(
        text,
        filter_cfg.max_chunk_chars,
        filter_cfg.min_chunk_chars,
    )
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


def interleave_named_sources(
    lanes: dict[str, tuple[Iterator[tuple[str, str, dict]], float]],
    *,
    seed: int,
) -> Iterator[tuple[str, str, dict]]:
    """Deterministically interleave arbitrary provenance-carrying source lanes."""
    active = {
        name: {"iterator": iterator, "weight": float(weight)}
        for name, (iterator, weight) in lanes.items()
        if float(weight) > 0
    }
    rng = random.Random(seed)
    try:
        while active:
            names = sorted(active)
            total = sum(active[name]["weight"] for name in names)
            roll = rng.random() * total
            selected = names[-1]
            cumulative = 0.0
            for name in names:
                cumulative += active[name]["weight"]
                if roll < cumulative:
                    selected = name
                    break
            try:
                yield next(active[selected]["iterator"])
            except StopIteration:
                del active[selected]
    finally:
        for lane in active.values():
            close_iter = getattr(lane["iterator"], "close", None)
            if callable(close_iter):
                try:
                    close_iter()
                except Exception:
                    pass


def partition_documents(
    source_iter: Iterator[tuple],
    *,
    split: str,
    validation_fraction: float,
    seed: int,
) -> Iterator[tuple]:
    """Assign each raw document to exactly one deterministic content split."""
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    threshold = int(validation_fraction * (1 << 64))
    salt = int(seed).to_bytes(8, byteorder="little", signed=True)
    try:
        for item in source_iter:
            source_id, text = item[:2]
            digest = hashlib.blake2b(
                text.encode("utf-8"),
                digest_size=8,
                key=salt,
            ).digest()
            bucket = int.from_bytes(digest, byteorder="big", signed=False)
            is_validation = bucket < threshold
            if (split == "validation") == is_validation:
                yield item
    finally:
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


def verify_flat_resume_provenance(
    out_dir: str,
    bin_paths: List[str],
    tokenizer_sha256: str,
    checkpoint: Optional["Checkpoint"],
) -> None:
    """Refuse to resume onto flat .bin files whose tokenizer provenance is
    absent or different from the tokenizer in use.

    Provenance is accepted from either a completed run's data_meta.json or an
    in-progress run's checkpoint. Existing bins with neither, or with a
    mismatched hash, abort the run: silently appending tokens from a second
    tokenizer produced the unusable v3 val.bin.
    """
    existing = [p for p in bin_paths if os.path.exists(p) and os.path.getsize(p) > 0]
    if not existing:
        return

    recorded = None
    provenance = None
    meta_path = os.path.join(out_dir, "data_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                recorded = json.load(f).get("tokenizer_sha256")
            provenance = meta_path
        except (json.JSONDecodeError, OSError):
            pass
    if recorded is None and checkpoint is not None and checkpoint.tokenizer_sha256:
        recorded = checkpoint.tokenizer_sha256
        provenance = "checkpoint"

    names = ", ".join(os.path.basename(p) for p in existing)
    if recorded is None:
        raise SystemExit(
            f"[resume] Refusing to resume: {names} exist in {out_dir} but no "
            "tokenizer provenance was found in data_meta.json or the "
            "checkpoint. Re-run with --overwrite to regenerate."
        )
    if recorded.lower() != tokenizer_sha256.lower():
        raise SystemExit(
            f"[resume] Tokenizer mismatch: {provenance} records "
            f"{recorded[:16]}... but the current tokenizer is "
            f"{tokenizer_sha256[:16]}.... Existing {names} would mix token "
            "vocabularies. Re-run with --overwrite, or restore the original "
            "tokenizer."
        )


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


def indexed_manifest_tokens(corpus_dir: str) -> int:
    manifest_path = os.path.join(corpus_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return 0
    manifest = _load_json_manifest(manifest_path)
    return int(manifest.get("token_count", 0))


def _load_json_manifest(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid indexed staging manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Invalid indexed staging manifest {path}")
    return value


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
    tokens_iter: Iterator[PreparedDocument | List[int]],
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
    tokenizer_sha256: Optional[str] = None,
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

    existing_tokens_view = None
    if resume_existing_tokens > 0:
        existing_tokens_view = np.memmap(
            output_path,
            dtype=dtype,
            mode="r",
            shape=(resume_existing_tokens,),
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
                    item = next(tokens_iter)
                    tokens = item.tokens if isinstance(item, PreparedDocument) else item
                    write_len = len(tokens) + 1
                    replay_stop = replayed_tokens + write_len
                    if replay_stop > resume_existing_tokens:
                        raise SystemExit(
                            "Resume mismatch: regenerated document boundaries do "
                            "not align with existing output. Use --overwrite."
                        )
                    expected = np.empty(write_len, dtype=dtype)
                    expected[:-1] = tokens
                    expected[-1] = eos_token_id
                    assert existing_tokens_view is not None
                    if not np.array_equal(
                        existing_tokens_view[replayed_tokens:replay_stop],
                        expected,
                    ):
                        raise SystemExit(
                            "Resume mismatch: regenerated tokens differ from the "
                            "existing output. The source, split, tokenizer, or "
                            "recipe changed; use --overwrite."
                        )
                    replayed_tokens = replay_stop
                    doc_count += 1
                print(
                    f"[resume] Replay complete at {replayed_tokens:,} tokens.",
                    flush=True,
                )

            for item in tokens_iter:
                tokens = item.tokens if isinstance(item, PreparedDocument) else item
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
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        tokenizer_sha256=tokenizer_sha256,
                    )
                    save_checkpoint(checkpoint, checkpoint_path)
                    f.flush()
                    os.fsync(f.fileno())
                    last_checkpoint_time = now
        finally:
            f.flush()
            os.fsync(f.fileno())
            pbar.close()
            if existing_tokens_view is not None:
                del existing_tokens_view

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
            timestamp=datetime.now(timezone.utc).isoformat(),
            tokenizer_sha256=tokenizer_sha256,
        )
        save_checkpoint(checkpoint, checkpoint_path)

    close_iter = getattr(tokens_iter, "close", None)
    if callable(close_iter):
        try:
            close_iter()
        except Exception:
            pass

    return position, was_interrupted


def write_documents_to_indexed(
    documents_iter: Iterator[PreparedDocument],
    output_dir: str,
    target_tokens: int,
    token_dtype: str,
    eos_token_id: int,
    tokenizer_sha256: str,
    recipe_sha256: str,
    sources: List[Source],
    target_shard_tokens: int,
    checkpoint_interval: int = 60,
    log_interval: int = 30,
    interrupt_handler: Optional[GracefulInterrupt] = None,
    split: str = "train",
    resume: bool = False,
) -> tuple[int, bool, Optional[str]]:
    """Transactionally publish an indexed corpus.

    Interrupted builds are aborted rather than exposing a partial immutable
    corpus. Resume support requires a separate resumable staging format and is
    intentionally not simulated here.
    """
    position = 0
    document_count = 0
    start_time = time.time()
    last_log_time = start_time
    last_checkpoint_time = start_time
    was_interrupted = False
    writer = ResumableIndexedShardWriter(
        output_dir,
        sources=sources,
        tokenizer_sha256=tokenizer_sha256,
        recipe_sha256=recipe_sha256,
        token_dtype=token_dtype,
        target_shard_tokens=target_shard_tokens,
        metadata={
            "split": split,
            "target_tokens": int(target_tokens),
            "eos_token_id": int(eos_token_id),
        },
        resume=resume,
    )
    position = writer.token_count
    document_count = writer.document_count

    try:
        for committed in writer.existing_documents:
            try:
                replayed = next(documents_iter)
            except StopIteration as exc:
                raise SystemExit(
                    f"Indexed {split} resume mismatch: source stream ended while "
                    "replaying the committed prefix."
                ) from exc
            replayed_tokens = np.asarray(
                [*replayed.tokens, eos_token_id],
                dtype=np.dtype(token_dtype),
            )
            committed_tokens = writer.read_committed_tokens(
                committed.document_id
            )
            if (
                replayed.source_id != committed.source_id
                or replayed.content_sha256 != committed.content_sha256
                or not np.array_equal(replayed_tokens, committed_tokens)
            ):
                raise SystemExit(
                    f"Indexed {split} resume mismatch at document "
                    f"{committed.document_id}. The source, split, tokenizer, "
                    "filtering, or recipe changed; restart with --overwrite."
                )

        for document in documents_iter:
            if interrupt_handler and interrupt_handler.interrupted:
                was_interrupted = True
                break
            tokens = [*document.tokens, eos_token_id]
            if position + len(tokens) > target_tokens:
                break
            writer.add_document(
                tokens,
                source_id=document.source_id,
                content_sha256=document.content_sha256,
                metadata=document.metadata,
            )
            position += len(tokens)
            document_count += 1

            now = time.time()
            if now - last_log_time >= log_interval:
                elapsed = now - start_time
                tokens_per_second = position / max(elapsed, 1e-6)
                print(
                    f"[indexed:{split}] {position:,}/{target_tokens:,} tokens "
                    f"({position / target_tokens * 100:.1f}%) | "
                    f"{document_count:,} docs | {tokens_per_second:,.0f} tok/s",
                    flush=True,
                )
                last_log_time = now
            if now - last_checkpoint_time >= checkpoint_interval:
                writer.checkpoint()
                last_checkpoint_time = now
    except BaseException:
        writer.discard_uncommitted()
        raise
    finally:
        close_iter = getattr(documents_iter, "close", None)
        if callable(close_iter):
            try:
                close_iter()
            except Exception:
                pass

    if was_interrupted:
        writer.suspend()
        return position, True, None
    if document_count == 0:
        writer.discard_uncommitted()
        raise SystemExit(
            f"Indexed {split} build produced no documents before target limit."
        )
    corpus_path = writer.close()
    corpus_fingerprint = json.loads(
        (corpus_path / "manifest.json").read_text(encoding="utf-8")
    )["corpus_sha256"]
    return position, False, corpus_fingerprint


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
) -> Iterator[PreparedDocument]:
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

    def prepared_document(
        source_id: str,
        text: str,
        tokens: List[int],
        metadata: Optional[dict] = None,
    ) -> PreparedDocument:
        document_metadata = dict(metadata or {})
        document_metadata["chunk_characters"] = len(text)
        return PreparedDocument(
            tokens=tokens,
            source_id=source_id,
            content_sha256=content_hash(text),
            metadata=document_metadata,
        )

    def log_filter_progress():
        """Log current filter statistics."""
        nonlocal last_log_time
        now = time.time()
        if now - last_log_time < log_interval:
            return
        last_log_time = now

        source_stats = {
            source_id: value
            for source_id, value in stats.items()
            if isinstance(value, FilterStats)
        }
        total_seen = sum(value.total_docs for value in source_stats.values())
        passed = []
        for source_id, value in source_stats.items():
            rate = (
                value.passed_docs / value.total_docs * 100
                if value.total_docs
                else 0
            )
            passed.append(
                f"{source_id} {value.passed_docs:,} ({rate:.1f}%)"
            )

        print(
            f"[filter] Seen {total_seen:,} docs | "
            f"Passed: {', '.join(passed)} | "
            f"Yielded: {docs_yielded:,} | "
            f"Dedup: {dedup_rejects:,} | UNK: {unk_rejects:,}",
            flush=True
        )

    pending_futures: deque[tuple[Future, str, str, dict]] = deque()
    executor: Optional[ThreadPoolExecutor] = None

    def submit_tokenization(source_id: str, chunk: str, metadata: dict):
        if executor is None:
            return
        pending_futures.append(
            (
                executor.submit(
                    _tokenize_chunk_worker,
                    chunk,
                    tokenizer_model_path,
                    max_unk_ratio,
                    _SHORT_DOC_UNK_TOKEN_LIMIT,
                ),
                source_id,
                chunk,
                metadata,
            )
        )

    def drain_futures(block: bool) -> Iterator[PreparedDocument]:
        nonlocal unk_rejects, docs_yielded
        while pending_futures:
            future, source_id, chunk, metadata = pending_futures[0]
            if not block and not future.done():
                break

            future, source_id, chunk, metadata = pending_futures.popleft()
            tokens, unk_rejected = future.result()
            if unk_rejected:
                unk_rejects += 1
            if tokens:
                docs_yielded += 1
                yield prepared_document(source_id, chunk, tokens, metadata)

            if block:
                break

    try:
        if use_parallel_tokenization:
            executor = ThreadPoolExecutor(
                max_workers=tokenizer_workers,
                thread_name_prefix="tok",
            )

        for source_item in source_iter:
            source_name, text = source_item[:2]
            upstream_metadata = (
                dict(source_item[2])
                if len(source_item) > 2 and isinstance(source_item[2], dict)
                else {}
            )
            if source_name == "c4":
                chunks = filter_and_transform_c4(text, filter_cfg, stats["c4"])
            elif source_name == "wiki":
                chunks = filter_and_transform_wiki(text, filter_cfg, stats["wiki"])
            elif source_name == "fineweb":
                chunks = filter_and_transform_fineweb(text, filter_cfg, stats["fineweb"])
            elif source_name in MIXTURE_SOURCES_BY_ID:
                source_stats = stats.setdefault(source_name, FilterStats())
                chunks = filter_and_transform_mixture(
                    text,
                    MIXTURE_SOURCES_BY_ID[source_name],
                    filter_cfg,
                    source_stats,
                )
            else:
                source_stats = stats.setdefault(source_name, FilterStats())
                chunks = filter_and_transform_generic(
                    text,
                    source_name,
                    filter_cfg,
                    source_stats,
                )

            # Log progress periodically
            log_filter_progress()

            parent_content_sha256 = content_hash(text)
            for chunk_index, chunk in enumerate(chunks):
                if deduplicator is not None and deduplicator.is_duplicate(chunk):
                    dedup_rejects += 1
                    continue

                chunk_metadata = {
                    **upstream_metadata,
                    "parent_content_sha256": parent_content_sha256,
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                }
                result = shuffle_buffer.add_and_maybe_yield(
                    (source_name, chunk, chunk_metadata)
                )
                if result is None:
                    continue
                result_source, result_chunk, result_metadata = result

                if use_parallel_tokenization:
                    submit_tokenization(
                        result_source,
                        result_chunk,
                        result_metadata,
                    )
                    while len(pending_futures) >= tokenizer_prefetch:
                        yield from drain_futures(block=True)
                    yield from drain_futures(block=False)
                else:
                    tokens = tokenize_and_filter_local(result_chunk)
                    if tokens:
                        docs_yielded += 1
                        yield prepared_document(
                            result_source,
                            result_chunk,
                            tokens,
                            result_metadata,
                        )

        for source_name, chunk, chunk_metadata in shuffle_buffer.flush():
            if use_parallel_tokenization:
                submit_tokenization(source_name, chunk, chunk_metadata)
                while len(pending_futures) >= tokenizer_prefetch:
                    yield from drain_futures(block=True)
                yield from drain_futures(block=False)
            else:
                tokens = tokenize_and_filter_local(chunk)
                if tokens:
                    docs_yielded += 1
                    yield prepared_document(
                        source_name,
                        chunk,
                        tokens,
                        chunk_metadata,
                    )

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
            for future, _, _, _ in pending_futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
        stats["dedup_rejects"] = dedup_rejects
        stats["unk_rejects"] = unk_rejects


def preparation_recipe_sha256(
    *,
    filter_cfg: FilterConfig,
    tokenizer_sha256: str,
    c4_weight: float,
    wiki_weight: float,
    fineweb_weight: float,
    shuffle_buffer: int,
    seed: int,
    max_unk_ratio: float,
    validation_fraction: float,
    canonical_manifest_sha256: str | None = None,
    writing_weight: float = 0.0,
    mixture_weights: Optional[Dict[str, float]] = None,
) -> str:
    """Fingerprint deterministic preparation inputs shared by both splits."""
    payload = {
        "schema_version": 1,
        "tokenizer_sha256": tokenizer_sha256,
        "source_weights": {
            "c4": float(c4_weight),
            "wiki": float(wiki_weight),
            "fineweb": float(fineweb_weight),
        },
        "shuffle_buffer": int(shuffle_buffer),
        "seed": int(seed),
        "max_unk_ratio": float(max_unk_ratio),
        "validation_fraction": float(validation_fraction),
        "canonical_manifest_sha256": canonical_manifest_sha256,
        "writing_weight": float(writing_weight),
        "filter_config": asdict(filter_cfg),
    }
    # Only enabled mixture sources enter the fingerprint, so a legacy-weights
    # run keeps the exact recipe hash it had before these sources existed.
    active = active_mixture_weights(mixture_weights)
    if active:
        payload["mixture_source_weights"] = active
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def indexed_source_records(
    c4_weight: float,
    wiki_weight: float,
    fineweb_weight: float,
    mixture_weights: Optional[Dict[str, float]] = None,
) -> List[Source]:
    """Describe enabled upstream sources without overstating text licensing."""
    candidates = (
        (
            c4_weight,
            Source(
                "c4",
                "AllenAI C4 English",
                metadata={
                    "dataset": "allenai/c4",
                    "config": "en",
                    "license_note": "Record-level rights may vary; review upstream terms.",
                },
            ),
        ),
        (
            wiki_weight,
            Source(
                "wiki",
                "English Wikipedia 2023-11-01",
                metadata={
                    "dataset": "wikimedia/wikipedia",
                    "config": "20231101.en",
                    "license_note": "Wikipedia content attribution and share-alike terms apply.",
                },
            ),
        ),
        (
            fineweb_weight,
            Source(
                "fineweb",
                "FineWeb-Edu",
                metadata={
                    "dataset": os.environ.get(
                        "FINEWEB_REPO",
                        "HuggingFaceFW/fineweb-edu",
                    ),
                    "config": os.environ.get("FINEWEB_CONFIG", "sample-10BT"),
                    "license_note": "Record-level rights may vary; review upstream terms.",
                },
            ),
        ),
    )
    records = [source for weight, source in candidates if weight > 0]
    for source_id in active_mixture_weights(mixture_weights):
        spec = MIXTURE_SOURCES_BY_ID[source_id]
        records.append(
            Source(
                spec.source_id,
                spec.name,
                metadata={
                    "dataset": spec.repo_id,
                    "config": spec.config,
                    "split": spec.split,
                    "filter_treatment": spec.filter_treatment,
                    "license_note": spec.license_note,
                },
            )
        )
    return records


def canonical_source_records(manifest_path: str | None) -> List[Source]:
    if not manifest_path:
        return []
    manifest = _load_json_manifest(manifest_path)
    records = []
    for source_id, source in sorted((manifest.get("sources") or {}).items()):
        records.append(
            Source(
                source_id,
                source.get("name") or source_id,
                source.get("license"),
                metadata={
                    "roles": source.get("roles") or [],
                    "canonical_manifest": os.path.abspath(manifest_path),
                    "selection_policy": manifest.get("selection_policy"),
                },
            )
        )
    return records


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
    parser.add_argument(
        "--output_format",
        choices=("flat", "indexed"),
        default=None,
        help="Output flat .bin files or immutable indexed corpora.",
    )
    parser.add_argument("--train_tokens", type=int, default=None, help="Target train tokens")
    parser.add_argument("--val_tokens", type=int, default=None, help="Target val tokens")
    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=None,
        help="Deterministic raw-document fraction reserved for validation.",
    )
    parser.add_argument("--c4_weight", type=float, default=None, help="C4 sampling weight")
    parser.add_argument("--wiki_weight", type=float, default=None, help="Wikipedia sampling weight")
    parser.add_argument("--fineweb_weight", type=float, default=None, help="FineWeb sampling weight")
    parser.add_argument(
        "--canonical_manifest",
        default=None,
        help="Verified canonical-writing manifest.",
    )
    parser.add_argument(
        "--writing_weight",
        type=float,
        default=None,
        help="Sampling weight for the canonical writing lane.",
    )
    parser.add_argument(
        "--gutenberg_weight",
        type=float,
        default=None,
        help="Deprecated alias for --fineweb_weight.",
    )
    for spec in MIXTURE_SOURCE_SPECS:
        parser.add_argument(
            f"--{spec.source_id}_weight",
            type=float,
            default=None,
            help=(
                f"Sampling weight for {spec.name} ({spec.repo_id}"
                + (f":{spec.config}" if spec.config else "")
                + "). Default 0.0 (source never initialized)."
            ),
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
    output_format = args.output_format or data_prep_cfg.get("output_format", "flat")
    if output_format not in {"flat", "indexed"}:
        raise SystemExit("data_prep.output_format must be one of: flat, indexed")
    train_tokens = args.train_tokens or data_prep_cfg.get("train_tokens", 2_000_000_000)
    val_tokens = args.val_tokens or data_prep_cfg.get("val_tokens", 20_000_000)
    default_validation_fraction = val_tokens / max(1, train_tokens + val_tokens)
    validation_fraction = (
        args.validation_fraction
        if args.validation_fraction is not None
        else data_prep_cfg.get(
            "validation_fraction",
            default_validation_fraction,
        )
    )
    validation_fraction = float(validation_fraction)
    if not 0.0 < validation_fraction < 1.0:
        raise SystemExit("validation_fraction must be between 0 and 1")
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
    canonical_cfg = data_prep_cfg.get("canonical_writing", {}) or {}
    canonical_manifest_path = (
        args.canonical_manifest
        if args.canonical_manifest is not None
        else canonical_cfg.get("manifest")
    )
    writing_weight = (
        args.writing_weight
        if args.writing_weight is not None
        else canonical_cfg.get("weight", 0.0)
    )
    writing_weight = float(writing_weight or 0.0)
    if writing_weight > 0 and not canonical_manifest_path:
        raise SystemExit(
            "canonical_writing.manifest is required when writing weight is positive"
        )
    if canonical_manifest_path and not os.path.exists(canonical_manifest_path):
        raise SystemExit(
            f"Canonical writing manifest not found: {canonical_manifest_path}"
        )

    mixture_weights = resolve_mixture_weights(args, data_prep_cfg)
    enabled_mixture_weights = active_mixture_weights(mixture_weights)

    if (
        c4_weight <= 0
        and wiki_weight <= 0
        and fineweb_weight <= 0
        and writing_weight <= 0
        and not enabled_mixture_weights
    ):
        raise SystemExit("At least one source sampling weight must be > 0.")
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
    target_shard_tokens = int(
        data_prep_cfg.get("target_shard_tokens", 10_000_000)
    )
    if target_shard_tokens <= 0:
        raise SystemExit("data_prep.target_shard_tokens must be positive")

    # Paths
    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")
    indexed_train_dir = os.path.join(out_dir, "train")
    indexed_val_dir = os.path.join(out_dir, "validation")
    indexed_train_staging = indexed_train_dir + ".staging"
    indexed_val_staging = indexed_val_dir + ".staging"
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
            if output_format == "flat" and (
                os.path.exists(train_path) or os.path.exists(val_path)
            ):
                print(f"Output files already exist in {out_dir}. Use --overwrite to replace or --resume to continue.")
                sys.exit(1)
            if output_format == "indexed" and (
                os.path.exists(indexed_train_dir)
                or os.path.exists(indexed_val_dir)
            ):
                raise SystemExit(
                    f"Indexed output already exists in {out_dir}. "
                    "Use --overwrite to replace both immutable splits."
                )
        if output_format == "indexed" and args.overwrite:
            for corpus_dir in (
                indexed_train_dir,
                indexed_val_dir,
                indexed_train_staging,
                indexed_val_staging,
            ):
                if os.path.isdir(corpus_dir):
                    shutil.rmtree(corpus_dir)

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
        recipe_sha256 = preparation_recipe_sha256(
            filter_cfg=filter_cfg,
            tokenizer_sha256=tokenizer_sha256,
            c4_weight=c4_weight,
            wiki_weight=wiki_weight,
            fineweb_weight=fineweb_weight,
            shuffle_buffer=shuffle_buffer,
            seed=seed,
            max_unk_ratio=0.01,
            validation_fraction=validation_fraction,
            canonical_manifest_sha256=(
                file_sha256(canonical_manifest_path)
                if canonical_manifest_path
                else None
            ),
            writing_weight=writing_weight,
            mixture_weights=mixture_weights,
        )
        source_records = indexed_source_records(
            c4_weight,
            wiki_weight,
            fineweb_weight,
            mixture_weights,
        )
        source_records.extend(
            canonical_source_records(canonical_manifest_path)
        )

        dtype = np.uint16 if vocab_size < 65536 else np.uint32
        print(f"Vocab size: {vocab_size}, dtype: {dtype}")

        existing_train_tokens = 0
        existing_val_tokens = 0
        train_final_exists = False
        val_final_exists = False
        train_staging_exists = False
        val_staging_exists = False
        if args.resume and output_format == "flat":
            verify_flat_resume_provenance(
                out_dir,
                [train_path, val_path],
                tokenizer_sha256,
                checkpoint,
            )
            existing_train_tokens = count_tokens_in_file(
                train_path, dtype, cap_at=train_tokens
            )
            existing_val_tokens = count_tokens_in_file(
                val_path, dtype, cap_at=val_tokens
            )
        elif args.resume and output_format == "indexed":
            train_final_exists = os.path.isdir(indexed_train_dir)
            val_final_exists = os.path.isdir(indexed_val_dir)
            train_staging_exists = os.path.isdir(indexed_train_staging)
            val_staging_exists = os.path.isdir(indexed_val_staging)
            if train_final_exists and train_staging_exists:
                raise SystemExit("Both final and staging train corpora exist.")
            if val_final_exists and val_staging_exists:
                raise SystemExit("Both final and staging validation corpora exist.")
            existing_train_tokens = indexed_manifest_tokens(
                indexed_train_dir
                if train_final_exists
                else indexed_train_staging
            )
            existing_val_tokens = indexed_manifest_tokens(
                indexed_val_dir if val_final_exists else indexed_val_staging
            )
        if args.resume:
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
        print(f"  Canonical writing weight: {writing_weight}")
        for source_id, weight in enabled_mixture_weights.items():
            spec = MIXTURE_SOURCES_BY_ID[source_id]
            config_suffix = f":{spec.config}" if spec.config else ""
            print(
                f"  {source_id} weight: {weight} "
                f"({spec.repo_id}{config_suffix}, {spec.filter_treatment} filters)"
            )
        if canonical_manifest_path:
            print(f"  Canonical manifest: {canonical_manifest_path}")
        print(f"  Output format: {output_format}")
        print(f"  Train tokens: {train_tokens:,}")
        print(f"  Val tokens: {val_tokens:,}")
        print(f"  Validation fraction: {validation_fraction:.4%}")
        print(f"  Shuffle buffer: {shuffle_buffer:,}")
        print(f"  Seed: {seed}")
        print(f"  Dedup enabled: {filter_cfg.dedup_enabled}")
        print(f"  Tokenizer workers: {tokenizer_workers}")
        print(f"  Tokenizer prefetch: {tokenizer_prefetch}")
        if output_format == "indexed":
            print(f"  Target shard tokens: {target_shard_tokens:,}")
            print(f"  Recipe sha256: {recipe_sha256}")
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
        for source_id in enabled_mixture_weights:
            stats.setdefault(source_id, FilterStats())
        for source in canonical_source_records(canonical_manifest_path):
            stats.setdefault(source.source_id, FilterStats())

        def build_interleaved_source(split_seed):
            def wrap(source_id, iterator):
                for text in iterator:
                    yield source_id, text, {}

            lanes = {
                "c4": (wrap("c4", stream_c4(split_seed)), c4_weight),
                "wiki": (
                    wrap("wiki", stream_wikipedia(split_seed)),
                    wiki_weight,
                ),
                "fineweb": (
                    wrap("fineweb", stream_fineweb(split_seed)),
                    fineweb_weight,
                ),
            }
            # Zero-weight mixture sources are never constructed, so a disabled
            # source never touches the network.
            for source_id, weight in enabled_mixture_weights.items():
                spec = MIXTURE_SOURCES_BY_ID[source_id]
                lanes[source_id] = (
                    wrap(source_id, open_mixture_stream(spec, split_seed)),
                    weight,
                )
            if canonical_manifest_path and writing_weight > 0:
                lanes["canonical_writing"] = (
                    iter_canonical_manifest(
                        Path(canonical_manifest_path)
                    ),
                    writing_weight,
                )
            return interleave_named_sources(lanes, seed=split_seed)

        # Set up interrupt handler
        train_corpus_sha256 = None
        val_corpus_sha256 = None
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
            if train_final_exists or train_resume_tokens >= train_tokens:
                print(
                    f"[resume] Training data already complete "
                    f"({train_resume_tokens:,}/{train_tokens:,} tokens). Skipping.",
                    flush=True,
                )
                actual_train_tokens = train_resume_tokens
                was_interrupted = False
            else:
                source_iter = build_interleaved_source(seed)
                source_iter = partition_documents(
                    source_iter,
                    split="train",
                    validation_fraction=validation_fraction,
                    seed=seed,
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

                if output_format == "indexed":
                    actual_train_tokens, was_interrupted, train_corpus_sha256 = (
                        write_documents_to_indexed(
                            tokens_iter,
                            indexed_train_dir,
                            train_tokens,
                            np.dtype(dtype).name,
                            eos_token_id,
                            tokenizer_sha256,
                            recipe_sha256,
                            source_records,
                            target_shard_tokens,
                            checkpoint_interval=args.checkpoint_interval,
                            log_interval=log_interval,
                            interrupt_handler=interrupt_handler,
                            split="train",
                            resume=train_staging_exists,
                        )
                    )
                else:
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
                        tokenizer_sha256=tokenizer_sha256,
                    )

            train_elapsed = time.time() - train_start
            print(f"\nTraining data complete: {actual_train_tokens:,} tokens in {train_elapsed/60:.1f} minutes")

            # Print filter statistics
            print("\nTraining filter statistics:")
            print("-" * 40)
            for source_id, source_stats in stats.items():
                if isinstance(source_stats, FilterStats):
                    print(f"\n{source_id}:")
                    print(source_stats.report())
            print(f"\nDeduplication rejects: {stats['dedup_rejects']:,}")

            if was_interrupted:
                if output_format == "indexed":
                    print(
                        "\n[!] Indexed training build interrupted. Committed "
                        "staging is valid; restart with --resume."
                    )
                else:
                    print(
                        "\n[!] Training interrupted. Checkpoint saved. "
                        "Use --resume to continue."
                    )
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
            for source_id in enabled_mixture_weights:
                val_stats.setdefault(source_id, FilterStats())
            for source in canonical_source_records(canonical_manifest_path):
                val_stats.setdefault(source.source_id, FilterStats())
            if val_final_exists or val_resume_tokens >= val_tokens:
                print(
                    f"[resume] Validation data already complete "
                    f"({val_resume_tokens:,}/{val_tokens:,} tokens). Skipping.",
                    flush=True,
                )
                actual_val_tokens = val_resume_tokens
                was_interrupted = False
            else:
                val_seed = seed + 999
                source_iter = build_interleaved_source(val_seed)
                source_iter = partition_documents(
                    source_iter,
                    split="validation",
                    validation_fraction=validation_fraction,
                    seed=seed,
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

                if output_format == "indexed":
                    actual_val_tokens, was_interrupted, val_corpus_sha256 = (
                        write_documents_to_indexed(
                            tokens_iter,
                            indexed_val_dir,
                            val_tokens,
                            np.dtype(dtype).name,
                            eos_token_id,
                            tokenizer_sha256,
                            recipe_sha256,
                            source_records,
                            target_shard_tokens,
                            checkpoint_interval=args.checkpoint_interval,
                            log_interval=log_interval,
                            interrupt_handler=interrupt_handler,
                            split="validation",
                            resume=val_staging_exists,
                        )
                    )
                else:
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
                        tokenizer_sha256=tokenizer_sha256,
                    )

            val_elapsed = time.time() - val_start
            print(f"\nValidation data complete: {actual_val_tokens:,} tokens in {val_elapsed/60:.1f} minutes")
            if was_interrupted:
                if output_format == "indexed":
                    print(
                        "\n[!] Indexed validation build interrupted. Committed "
                        "staging is valid; restart with --resume."
                    )
                else:
                    print(
                        "\n[!] Validation interrupted. Checkpoint saved. "
                        "Use --resume to continue."
                    )
                return

        # =====================================================================
        # Save metadata
        # =====================================================================
        meta = {
            "output_format": output_format,
            "train_tokens": actual_train_tokens,
            "val_tokens": actual_val_tokens,
            "vocab_size": vocab_size,
            "dtype": str(dtype),
            "eos_token_id": eos_token_id,
            "c4_weight": c4_weight,
            "wiki_weight": wiki_weight,
            "fineweb_weight": fineweb_weight,
            "writing_weight": writing_weight,
            "mixture_source_weights": enabled_mixture_weights,
            "mixture_sources": {
                source_id: {
                    "dataset": MIXTURE_SOURCES_BY_ID[source_id].repo_id,
                    "config": MIXTURE_SOURCES_BY_ID[source_id].config,
                    "filter_treatment": MIXTURE_SOURCES_BY_ID[
                        source_id
                    ].filter_treatment,
                }
                for source_id in enabled_mixture_weights
            },
            "canonical_manifest": (
                os.path.abspath(canonical_manifest_path)
                if canonical_manifest_path
                else None
            ),
            "validation_fraction": validation_fraction,
            # Backward-compatible alias.
            "gutenberg_weight": fineweb_weight,
            "seed": seed,
            "tokenizer_workers": tokenizer_workers,
            "tokenizer_prefetch": tokenizer_prefetch,
            "tokenizer_model": os.path.abspath(tokenizer_model),
            "tokenizer_sha256": tokenizer_sha256,
            "tokenizer_vocab_size": vocab_size,
            "tokenizer_special_ids": tokenizer_special_ids,
            "preparation_recipe_sha256": recipe_sha256,
            "indexed": {
                "train_dir": (
                    os.path.abspath(indexed_train_dir)
                    if output_format == "indexed"
                    else None
                ),
                "validation_dir": (
                    os.path.abspath(indexed_val_dir)
                    if output_format == "indexed"
                    else None
                ),
                "train_corpus_sha256": train_corpus_sha256,
                "validation_corpus_sha256": val_corpus_sha256,
                "target_shard_tokens": (
                    target_shard_tokens if output_format == "indexed" else None
                ),
            },
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
                "by_source": {
                    source_id: {
                        "total": source_stats.total_docs,
                        "passed": source_stats.passed_docs,
                    }
                    for source_id, source_stats in stats.items()
                    if isinstance(source_stats, FilterStats)
                },
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
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
        if output_format == "indexed":
            print(f"  {indexed_train_dir} ({actual_train_tokens:,} tokens)")
            print(f"  {indexed_val_dir} ({actual_val_tokens:,} tokens)")
        else:
            print(f"  {train_path} ({actual_train_tokens:,} tokens)")
            print(f"  {val_path} ({actual_val_tokens:,} tokens)")
        print(f"  {meta_path}")

    finally:
        cleanup_logging()


if __name__ == "__main__":
    main()

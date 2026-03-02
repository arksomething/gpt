"""Helpers for streaming high-quality auxiliary text.

The public API keeps the legacy `stream_gutenberg()` name for compatibility.
By default, this module now prefers fast Project Gutenberg PG-19 HTTP streaming
for book-heavy data, with the previous HuggingFace "variety" path available as
a fallback mode.
"""

from __future__ import annotations

import os
import random
import re
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Iterator, Optional

from datasets import load_dataset

# Fast books-first mode (default): direct PG-19 HTTP files.
PG19_FILE_LIST_URL = "https://huggingface.co/datasets/deepmind/pg19/resolve/main/data/{split}_files.txt"
PG19_ASSET_ROOT_URL = "https://storage.googleapis.com/deepmind-gutenberg/"
_PG19_FILE_LIST_CACHE: dict[str, list[str]] = {}

# Previous "variety" stream defaults (kept as fallback / opt-in mode).
DEFAULT_PRIMARY_REPO = "HuggingFaceH4/ultrachat_200k"
DEFAULT_PRIMARY_CONFIG = None
DEFAULT_PRIMARY_TEXT_FIELD = "messages"
DEFAULT_PRIMARY_SPLIT = "train_sft"

DEFAULT_FALLBACK_REPO = "HuggingFaceFW/fineweb-edu"
DEFAULT_FALLBACK_CONFIG = "sample-10BT"
DEFAULT_FALLBACK_TEXT_FIELD = "text"
DEFAULT_FALLBACK_SPLIT = "train"

# Modes:
# - books_fast (default): PG-19 HTTP first, then variety fallback if needed.
# - variety: HuggingFace variety sources only (legacy behavior).
# - pg19_only: PG-19 HTTP only (fail if unavailable).
DEFAULT_STREAM_MODE = "books_fast"

# Keep shuffle modest for better startup/steady-state latency in streaming mode.
DEFAULT_SHUFFLE_BUFFER = 512

# PG-19 HTTP streaming tuning (books_fast / pg19_only modes).
DEFAULT_PG19_HTTP_WORKERS = 4
DEFAULT_PG19_HTTP_PREFETCH = 12
DEFAULT_PG19_HTTP_TIMEOUT_SEC = 30
DEFAULT_PG19_CLEAN_LEVEL = "balanced"

_PG19_CLEAN_LEVELS = {"off", "balanced", "aggressive"}
_PG19_CHAPTER_HEADING_RE = re.compile(
    r"^\s*(?:chapter|book|part)\s+(?:[ivxlcdm]+|\d+|first|second|third|fourth|fifth)\b",
    flags=re.IGNORECASE,
)
_PG19_TOC_ROW_RE = re.compile(
    r"^\s*(?:chap(?:ter)?\.?|book|part)\b.*\b\d+\s*$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class DatasetSource:
    repo_id: str
    config: Optional[str]
    text_field: Optional[str] = None
    split: Optional[str] = None


def _env_optional(name: str, default: Optional[str]) -> Optional[str]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw or None


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"[variety] Invalid {name}={raw!r}; using default={default}.")
        return default
    if value <= 0:
        print(f"[variety] Non-positive {name}={raw!r}; using default={default}.")
        return default
    return value


def _env_nonnegative_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"[books] Invalid {name}={raw!r}; using default={default}.")
        return default
    if value < 0:
        print(f"[books] Negative {name}={raw!r}; using default={default}.")
        return default
    return value


def _env_mode(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    return raw or default


def _extract_text(example: dict, preferred_field: Optional[str]) -> Optional[str]:
    def _flatten_messages(value: object) -> Optional[str]:
        if not isinstance(value, list):
            return None
        parts: list[str] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, str):
                text = content.strip()
                if text:
                    parts.append(text)
        if not parts:
            return None
        return "\n\n".join(parts)

    if preferred_field:
        value = example.get(preferred_field)
        if isinstance(value, str) and value.strip():
            return value
        flattened = _flatten_messages(value)
        if flattened:
            return flattened

    for key in ("text", "content", "body"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for key in ("messages", "conversation"):
        flattened = _flatten_messages(example.get(key))
        if flattened:
            return flattened
    return None


def _fetch_pg19_file_list(split: str) -> list[str]:
    if split in _PG19_FILE_LIST_CACHE:
        return list(_PG19_FILE_LIST_CACHE[split])

    url = PG19_FILE_LIST_URL.format(split=split)
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read().decode("utf-8")

    files = [line.strip() for line in data.splitlines() if line.strip()]
    if not files:
        raise RuntimeError(f"PG-19 file list is empty for split={split!r}")

    _PG19_FILE_LIST_CACHE[split] = files
    return list(files)


def _normalize_pg19_split(split: str) -> str:
    normalized = split.strip().lower()
    if normalized == "val":
        return "validation"
    if normalized in {"train", "validation", "test"}:
        return normalized
    return "train"


def _strip_gutenberg_boilerplate(text: str) -> str:
    upper = text.upper()
    start_idx = None
    for marker in (
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK",
    ):
        idx = upper.find(marker)
        if idx != -1:
            next_nl = text.find("\n", idx)
            start_idx = next_nl + 1 if next_nl != -1 else idx
            break

    if start_idx is None:
        start_idx = 0

    end_idx = len(text)
    for marker in (
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK",
    ):
        idx = upper.find(marker, start_idx)
        if idx != -1:
            end_idx = idx
            break

    return text[start_idx:end_idx].strip()


def _is_pg19_front_signal(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    lower = stripped.lower()
    return any(
        token in lower
        for token in (
            "produced by",
            "transcribed from",
            "proofreading team",
            "distributed proofreading",
            "pgdp.net",
            "project gutenberg",
            "internet archive",
            "transcriber's note",
            "all rights reserved",
            "copyright",
            "published by",
            "printed by",
            "list of illustrations",
        )
    ) or "@" in lower


def _is_pg19_toc_marker(line: str) -> bool:
    lower = line.strip().lower()
    return lower in {"contents", "table of contents"} or lower.startswith("contents")


def _is_pg19_toc_row(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _PG19_TOC_ROW_RE.match(stripped):
        return True
    if re.match(r"^\s*chapter\s+[ivxlcdm0-9]+[.)-]*\s*--", stripped, flags=re.IGNORECASE):
        return True
    if re.match(r"^[ivxlcdm0-9]+\.\s+.+\s+\d+\s*$", stripped, flags=re.IGNORECASE):
        return True
    return False


def _looks_like_pg19_body_line(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 55 or stripped.count(" ") < 8:
        return False
    letters = [c for c in stripped if c.isalpha()]
    if len(letters) < 40:
        return False
    lower_ratio = sum(1 for c in letters if c.islower()) / len(letters)
    return lower_ratio > 0.55


def _is_pg19_heading_like(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    lower = stripped.lower()
    if lower in {"conclusion", "appendix", "index"} or lower.startswith("supplement"):
        return True
    if _PG19_CHAPTER_HEADING_RE.match(stripped):
        return True
    if _is_pg19_toc_row(stripped):
        return True
    letters = [c for c in stripped if c.isalpha()]
    if len(letters) >= 6:
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_ratio > 0.85 and len(stripped) <= 90:
            return True
    return False


def _is_pg19_probable_toc_chapter(lines: list[str], idx: int) -> bool:
    if idx < 0 or idx >= len(lines):
        return False
    if not _PG19_CHAPTER_HEADING_RE.match(lines[idx].strip()):
        return False

    lookahead = [line.strip() for line in lines[idx + 1 : idx + 30] if line.strip()]
    if not lookahead:
        return False

    chapter_count = sum(1 for line in lookahead[:15] if _PG19_CHAPTER_HEADING_RE.match(line))
    if chapter_count >= 2:
        return True

    if any(
        line.lower() in {"page", "facing page", "contents", "list of illustrations", "index"}
        for line in lookahead[:12]
    ):
        return True

    if any(_is_pg19_toc_row(line) for line in lookahead[:12]):
        return True

    if any(
        re.match(r"^[A-Z][A-Z0-9 ,.'\"()\-]{6,}\s+\d+\s*$", line)
        for line in lookahead[:12]
    ):
        return True

    return False


def _clean_pg19_text(text: str, clean_level: str) -> str:
    if clean_level == "off":
        return text.strip()

    # Remove bracketed illustration blocks that are almost always metadata.
    text = re.sub(r"\[\s*illustration:.*?\]", " ", text, flags=re.IGNORECASE | re.DOTALL)
    lines = [line.rstrip() for line in text.splitlines()]
    if not lines:
        return ""

    max_scan = min(len(lines), 1200)
    has_signal = False
    has_contents = False
    has_preface = False
    chapter_idx: Optional[int] = None
    first_body_idx: Optional[int] = None

    for idx in range(max_scan):
        line = lines[idx]
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if _is_pg19_front_signal(stripped):
            has_signal = True
        if _is_pg19_toc_marker(stripped):
            has_contents = True
            has_signal = True
        if lower.startswith("preface"):
            has_preface = True
            has_signal = True
        if chapter_idx is None and _PG19_CHAPTER_HEADING_RE.match(stripped):
            if _is_pg19_probable_toc_chapter(lines, idx):
                has_contents = True
                has_signal = True
            else:
                chapter_idx = idx
        if first_body_idx is None and _looks_like_pg19_body_line(stripped):
            first_body_idx = idx

    body_start = 0
    if clean_level == "aggressive":
        if chapter_idx is not None and (has_signal or has_preface or has_contents):
            body_start = chapter_idx
        elif has_signal and first_body_idx is not None:
            body_start = first_body_idx
    else:  # balanced
        if has_contents and chapter_idx is not None:
            body_start = chapter_idx
        elif has_signal and first_body_idx is not None:
            body_start = first_body_idx

    cleaned_lines: list[str] = []
    skipping_contents = False
    for idx, line in enumerate(lines[body_start:], start=body_start):
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue

        if _is_pg19_front_signal(stripped):
            continue

        if _is_pg19_toc_marker(stripped):
            skipping_contents = True
            continue

        if _PG19_CHAPTER_HEADING_RE.match(stripped) and _is_pg19_probable_toc_chapter(lines, idx):
            skipping_contents = True
            continue

        if skipping_contents:
            if _PG19_CHAPTER_HEADING_RE.match(stripped):
                if _is_pg19_probable_toc_chapter(lines, idx):
                    continue
                skipping_contents = False
                cleaned_lines.append(stripped)
                continue
            if _is_pg19_toc_row(stripped) or len(stripped) <= 100:
                continue
            # Fail safe: if we hit prose, keep it.
            skipping_contents = False

        if _is_pg19_toc_row(stripped):
            continue

        cleaned_lines.append(stripped)

    # Trim residual heading-only front matter before the first prose paragraph.
    first_body_idx: Optional[int] = None
    for idx, line in enumerate(cleaned_lines):
        if _looks_like_pg19_body_line(line):
            first_body_idx = idx
            break
    if first_body_idx is not None and first_body_idx > 0:
        prefix = cleaned_lines[:first_body_idx]
        chapter_positions = [
            idx for idx, line in enumerate(prefix) if _PG19_CHAPTER_HEADING_RE.match(line.strip())
        ]
        if chapter_positions:
            keep_from = chapter_positions[-1]
            prev = keep_from - 1
            while prev >= 0 and not prefix[prev].strip():
                prev -= 1
            if prev >= 0 and not _PG19_CHAPTER_HEADING_RE.match(prefix[prev].strip()):
                keep_from = prev
            cleaned_lines = cleaned_lines[keep_from:]
        else:
            trim_from = 0
            while trim_from < first_body_idx and _is_pg19_heading_like(prefix[trim_from]):
                trim_from += 1
            if trim_from > 0:
                cleaned_lines = cleaned_lines[trim_from:]

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _download_pg19_text(
    rel_path: str,
    timeout_sec: int,
    max_bytes_per_doc: int,
    strip_boilerplate: bool,
    clean_level: str,
) -> Optional[str]:
    url = PG19_ASSET_ROOT_URL + rel_path
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as response:
            if max_bytes_per_doc > 0:
                raw = response.read(max_bytes_per_doc)
            else:
                raw = response.read()
    except Exception as exc:
        print(f"[books] Failed to fetch {url}: {exc}")
        return None

    text = raw.decode("utf-8", errors="replace")
    if strip_boilerplate:
        text = _strip_gutenberg_boilerplate(text)
    text = _clean_pg19_text(text, clean_level=clean_level)
    text = text.strip()
    return text or None


def _stream_pg19_via_http(seed: int, split: str) -> Iterator[str]:
    pg_split = _normalize_pg19_split(split)
    files = _fetch_pg19_file_list(pg_split)
    rng = random.Random(seed)
    rng.shuffle(files)

    workers = _env_positive_int("PG19_HTTP_WORKERS", DEFAULT_PG19_HTTP_WORKERS)
    prefetch = _env_positive_int("PG19_HTTP_PREFETCH", DEFAULT_PG19_HTTP_PREFETCH)
    timeout_sec = _env_positive_int("PG19_HTTP_TIMEOUT_SEC", DEFAULT_PG19_HTTP_TIMEOUT_SEC)
    max_bytes_per_doc = _env_nonnegative_int("PG19_MAX_BYTES_PER_DOC", 0)
    strip_boilerplate = _env_mode("PG19_STRIP_BOILERPLATE", "1") not in {"0", "false", "no"}
    clean_level = _env_mode("PG19_CLEAN_LEVEL", DEFAULT_PG19_CLEAN_LEVEL)
    if clean_level not in _PG19_CLEAN_LEVELS:
        print(
            f"[books] Unknown PG19_CLEAN_LEVEL={clean_level!r}; "
            f"using {DEFAULT_PG19_CLEAN_LEVEL!r}"
        )
        clean_level = DEFAULT_PG19_CLEAN_LEVEL

    prefetch = max(prefetch, workers)
    print(
        f"[books] HTTP workers={workers} prefetch={prefetch} "
        f"timeout={timeout_sec}s max_bytes={max_bytes_per_doc or 'all'} "
        f"clean_level={clean_level}"
    )

    file_iter = iter(files)
    in_flight: dict[Future, str] = {}

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        try:
            rel_path = next(file_iter)
        except StopIteration:
            return False
        future = executor.submit(
            _download_pg19_text,
            rel_path,
            timeout_sec,
            max_bytes_per_doc,
            strip_boilerplate,
            clean_level,
        )
        in_flight[future] = rel_path
        return True

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while len(in_flight) < prefetch and submit_next(executor):
            pass

        while in_flight:
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                in_flight.pop(future, None)
                text = future.result()
                if text:
                    yield text

                while len(in_flight) < prefetch and submit_next(executor):
                    pass


def _iter_sources_from_env() -> list[DatasetSource]:
    primary = DatasetSource(
        repo_id=_env_optional("VARIETY_DATASET_REPO", DEFAULT_PRIMARY_REPO) or DEFAULT_PRIMARY_REPO,
        config=_env_optional("VARIETY_DATASET_CONFIG", DEFAULT_PRIMARY_CONFIG),
        text_field=_env_optional("VARIETY_DATASET_TEXT_FIELD", DEFAULT_PRIMARY_TEXT_FIELD),
        split=_env_optional("VARIETY_DATASET_SPLIT", DEFAULT_PRIMARY_SPLIT),
    )
    fallback_repo = _env_optional("VARIETY_DATASET_FALLBACK_REPO", DEFAULT_FALLBACK_REPO)
    fallback = None
    if fallback_repo:
        fallback = DatasetSource(
            repo_id=fallback_repo,
            config=_env_optional("VARIETY_DATASET_FALLBACK_CONFIG", DEFAULT_FALLBACK_CONFIG),
            text_field=_env_optional(
                "VARIETY_DATASET_FALLBACK_TEXT_FIELD",
                DEFAULT_FALLBACK_TEXT_FIELD,
            ),
            split=_env_optional("VARIETY_DATASET_FALLBACK_SPLIT", DEFAULT_FALLBACK_SPLIT),
        )

    sources = [primary]
    if fallback and fallback.repo_id != primary.repo_id:
        sources.append(fallback)
    return sources


def _stream_source(
    source: DatasetSource,
    split: str,
    seed: int,
    shuffle_buffer: int,
) -> Iterator[str]:
    effective_split = source.split or split
    if source.config:
        ds = load_dataset(source.repo_id, source.config, split=effective_split, streaming=True)
    else:
        ds = load_dataset(source.repo_id, split=effective_split, streaming=True)

    if shuffle_buffer > 1:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    for example in ds:
        text = _extract_text(example, source.text_field)
        if text:
            yield text


def _stream_variety_sources(seed: int, split: str) -> Iterator[str]:
    shuffle_buffer = _env_positive_int("VARIETY_DATASET_SHUFFLE_BUFFER", DEFAULT_SHUFFLE_BUFFER)
    sources = _iter_sources_from_env()
    if not sources:
        raise RuntimeError("No variety dataset source configured.")

    last_error: Optional[Exception] = None

    for source in sources:
        label = source.repo_id + (f"/{source.config}" if source.config else "")
        effective_split = source.split or split
        print(f"[variety] Loading {label} split={effective_split}...")
        yielded = 0
        try:
            for text in _stream_source(source, split=split, seed=seed, shuffle_buffer=shuffle_buffer):
                yielded += 1
                if yielded == 1:
                    print("[variety] First document yielded", flush=True)
                elif yielded % 5000 == 0:
                    print(f"[variety] {yielded:,} docs", flush=True)
                yield text
            if yielded > 0:
                return
        except Exception as exc:
            last_error = exc
            print(f"[variety] Source failed ({label}): {exc}")
            continue

    if last_error is not None:
        raise RuntimeError("All configured variety sources failed.") from last_error
    raise RuntimeError("All configured variety sources were empty.")


def stream_gutenberg(seed: int = 42, split: str = "train") -> Iterator[str]:
    """Stream text from the configured auxiliary source mode.

    Environment variables:
      - VARIETY_DATASET_MODE: books_fast | variety | pg19_only
      - VARIETY_DATASET_SHUFFLE_BUFFER: shuffle buffer for variety mode
    """
    mode = _env_mode("VARIETY_DATASET_MODE", DEFAULT_STREAM_MODE)
    if mode not in {"books_fast", "variety", "pg19_only"}:
        print(
            f"[books] Unknown VARIETY_DATASET_MODE={mode!r}; "
            f"using default={DEFAULT_STREAM_MODE!r}"
        )
        mode = DEFAULT_STREAM_MODE

    if mode in {"books_fast", "pg19_only"}:
        print(f"[books] Loading PG-19 split={_normalize_pg19_split(split)}...")
        yielded = 0
        try:
            for text in _stream_pg19_via_http(seed=seed, split=split):
                yielded += 1
                if yielded == 1:
                    print("[books] First document yielded", flush=True)
                elif yielded % 5000 == 0:
                    print(f"[books] {yielded:,} docs", flush=True)
                yield text
            if yielded > 0:
                return
        except Exception as exc:
            if mode == "pg19_only":
                raise RuntimeError("PG-19 streaming failed in pg19_only mode.") from exc
            print(f"[books] PG-19 streaming failed, falling back to variety sources: {exc}")
            if mode != "books_fast":
                raise

    # variety mode, or fallback from books_fast
    yield from _stream_variety_sources(seed=seed, split=split)

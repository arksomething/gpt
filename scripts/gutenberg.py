"""Helpers for streaming Project Gutenberg (PG-19) text."""

from __future__ import annotations

import random
import urllib.request
from typing import Iterator, List

from datasets import load_dataset

PG19_FILE_LIST_URL = (
    "https://huggingface.co/datasets/deepmind/pg19/resolve/main/data/{split}_files.txt"
)
PG19_ASSET_ROOT_URL = "https://storage.googleapis.com/deepmind-gutenberg/"
_PG19_FILE_LIST_CACHE: dict[str, List[str]] = {}


def _fetch_pg19_file_list(split: str) -> List[str]:
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


def _stream_pg19_via_http(seed: int, split: str) -> Iterator[str]:
    files = _fetch_pg19_file_list(split)
    rng = random.Random(seed)
    rng.shuffle(files)

    for rel_path in files:
        url = PG19_ASSET_ROOT_URL + rel_path
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                text = response.read().decode("utf-8", errors="replace")
        except Exception as exc:
            print(f"[gutenberg] Failed to fetch {url}: {exc}")
            continue

        if text:
            yield text


def stream_gutenberg(seed: int = 42, split: str = "train") -> Iterator[str]:
    """Stream text from Project Gutenberg (PG-19).

    Falls back to direct HTTP fetches when dataset scripts are disabled.
    """
    try:
        ds = load_dataset("deepmind/pg19", split=split, streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=1000)
        for example in ds:
            yield example["text"]
        return
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise
        print("[gutenberg] PG-19 dataset script unsupported; using HTTP fallback.")

    yield from _stream_pg19_via_http(seed, split)

#!/usr/bin/env python3
"""Prepare immutable assistant-supervised chat corpora from JSONL messages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import sentencepiece as spm

from scripts.chat_format import encode_conversation, labels_to_spans, template_metadata
from scripts.indexed_shards import (
    IndexedShardWriter,
    ShardFormatError,
    Source,
    content_hash,
    sha256_file,
)


SCHEMA_VERSION = 1


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _iter_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, value


def _source_from_record(record: dict[str, Any]) -> Source:
    source_id = record.get("source_id", "chat")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("source_id must be a non-empty string")
    source_name = record.get("source_name", source_id)
    license_name = record.get("license")
    if not isinstance(source_name, str) or not source_name:
        raise ValueError("source_name must be a non-empty string")
    if license_name is not None and not isinstance(license_name, str):
        raise ValueError("license must be a string or null")
    return Source(
        source_id=source_id,
        name=source_name,
        license=license_name,
        metadata={"synthetic": bool(record.get("synthetic", False))},
    )


def _collect_sources(input_path: Path) -> list[Source]:
    by_id: dict[str, Source] = {}
    for line_number, record in _iter_records(input_path):
        try:
            source = _source_from_record(record)
        except ValueError as exc:
            raise ValueError(f"{input_path}:{line_number}: {exc}") from exc
        previous = by_id.get(source.source_id)
        if previous is not None and previous != source:
            raise ValueError(
                f"{input_path}:{line_number}: inconsistent metadata for "
                f"source_id {source.source_id!r}"
            )
        by_id[source.source_id] = source
    if not by_id:
        raise ValueError(f"{input_path}: contains no records")
    return [by_id[source_id] for source_id in sorted(by_id)]


def _is_validation(content_sha256: str, validation_fraction: float) -> bool:
    bucket = int(content_sha256[:16], 16) / float(16**16)
    return bucket < validation_fraction


def prepare_chat_corpus(
    *,
    input_path: Path,
    output_dir: Path,
    tokenizer_path: Path,
    validation_fraction: float,
    target_shard_tokens: int,
    max_tokens: int | None,
    require_human_keep: bool = False,
) -> dict[str, Any]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable corpus: {output_dir}")
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(tokenizer_path)

    sources = _collect_sources(input_path)
    tokenizer_sha256 = sha256_file(tokenizer_path)
    input_sha256 = sha256_file(input_path)
    recipe = {
        "schema_version": SCHEMA_VERSION,
        "input_sha256": input_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "validation_fraction": validation_fraction,
        "target_shard_tokens": target_shard_tokens,
        "max_tokens": max_tokens,
        "require_human_keep": require_human_keep,
        "template": template_metadata(),
        "split": "canonical-conversation-sha256",
    }
    recipe_sha256 = _canonical_sha256(recipe)

    tokenizer = spm.SentencePieceProcessor()
    if not tokenizer.load(str(tokenizer_path)):
        raise ValueError(f"failed to load tokenizer: {tokenizer_path}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.building-",
            dir=output_dir.parent,
        )
    )
    writers: dict[str, IndexedShardWriter] = {}
    counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    try:
        for split in ("train", "validation"):
            writers[split] = IndexedShardWriter(
                staging / split,
                sources=sources,
                tokenizer_sha256=tokenizer_sha256,
                recipe_sha256=recipe_sha256,
                token_dtype=("uint16" if tokenizer.vocab_size() <= 65536 else "uint32"),
                target_shard_tokens=target_shard_tokens,
                metadata={
                    "kind": "assistant-supervised-chat",
                    "split": split,
                    "template": template_metadata(),
                },
            )

        for line_number, record in _iter_records(input_path):
            if require_human_keep:
                human_review = record.get("human_review")
                if (
                    not isinstance(human_review, dict)
                    or human_review.get("keep") is not True
                ):
                    raise ValueError(
                        f"{input_path}:{line_number}: record is not an explicit "
                        "human keep"
                    )
            messages = record.get("messages")
            try:
                encoded = encode_conversation(tokenizer, messages)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{input_path}:{line_number}: {exc}") from exc
            if max_tokens is not None and len(encoded.input_ids) > max_tokens:
                counts["rejected_too_long"] += 1
                continue

            canonical_conversation = {
                "messages": messages,
                "source_id": record.get("source_id", "chat"),
            }
            conversation_sha256 = _canonical_sha256(canonical_conversation)
            split = (
                "validation"
                if _is_validation(conversation_sha256, validation_fraction)
                else "train"
            )
            source_id = str(record.get("source_id", "chat"))
            metadata = {
                "kind": "chat",
                "record_id": record.get("id"),
                "conversation_sha256": conversation_sha256,
                "supervision_spans": [
                    list(span) for span in labels_to_spans(encoded.labels)
                ],
                "supervised_tokens": encoded.supervised_tokens,
                "message_count": len(messages),
                "synthetic": bool(record.get("synthetic", False)),
                "generator": record.get("generator"),
                "license_evidence": record.get("license_evidence"),
            }
            writers[split].add_document(
                encoded.input_ids,
                source_id=source_id,
                content_sha256=content_hash(_canonical_bytes(messages)),
                quality_score=record.get("quality_score"),
                metadata=metadata,
            )
            counts[f"{split}_documents"] += 1
            counts[f"{split}_tokens"] += len(encoded.input_ids)
            counts[f"{split}_supervised_tokens"] += encoded.supervised_tokens
            source_counts[f"{split}:{source_id}"] += 1

        if counts["train_documents"] == 0 or counts["validation_documents"] == 0:
            raise ValueError(
                "deterministic split produced an empty train or validation set; "
                "provide more conversations or adjust validation_fraction"
            )
        for writer in writers.values():
            writer.close()

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "assistant-supervised-chat-corpus",
            "input": {
                "path": str(input_path.resolve()),
                "sha256": input_sha256,
            },
            "tokenizer": {
                "path": str(tokenizer_path.resolve()),
                "sha256": tokenizer_sha256,
                "vocab_size": tokenizer.vocab_size(),
            },
            "recipe": recipe,
            "recipe_sha256": recipe_sha256,
            "counts": dict(sorted(counts.items())),
            "source_document_counts": dict(sorted(source_counts.items())),
            "splits": {
                split: json.loads(
                    (staging / split / "manifest.json").read_text(encoding="utf-8")
                )["corpus_sha256"]
                for split in ("train", "validation")
            },
        }
        manifest["manifest_sha256"] = _canonical_sha256(manifest)
        (staging / "chat_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_dir)
        return manifest
    except Exception:
        for writer in writers.values():
            try:
                writer.discard_uncommitted()
            except Exception:
                pass
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare immutable assistant-supervised indexed chat data."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--tokenizer", default=Path("tokenizer/spm.model"), type=Path)
    parser.add_argument("--validation_fraction", type=float, default=0.02)
    parser.add_argument("--target_shard_tokens", type=int, default=10_000_000)
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Reject conversations above this token count instead of truncating.",
    )
    parser.add_argument(
        "--require_human_keep",
        action="store_true",
        help="Fail if any input record lacks human_review.keep=true.",
    )
    args = parser.parse_args()
    try:
        manifest = prepare_chat_corpus(
            input_path=args.input,
            output_dir=args.output_dir,
            tokenizer_path=args.tokenizer,
            validation_fraction=args.validation_fraction,
            target_shard_tokens=args.target_shard_tokens,
            max_tokens=args.max_tokens,
            require_human_keep=args.require_human_keep,
        )
    except (OSError, ValueError, ShardFormatError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

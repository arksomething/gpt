"""Immutable, document-aware token shards with a verifiable manifest.

This module is intentionally independent of the current data preparation and
training paths.  It defines the storage boundary that a future data pipeline
can write and a future sampler can consume without losing document provenance.

Layout::

    corpus/
      manifest.json
      documents.jsonl
      tokens-00000.bin
      tokens-00001.bin

Token files contain little-endian unsigned integers.  Documents never span
shards.  The JSONL index records each document's source, token range, content
hash, optional quality score, and arbitrary JSON metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from scripts.chat_format import ChatFormatError, apply_supervision_spans


FORMAT_NAME = "gpt-indexed-token-shards"
FORMAT_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
INDEX_FILENAME = "documents.jsonl"
_ALLOWED_DTYPES = {"uint16": np.dtype("<u2"), "uint32": np.dtype("<u4")}


class ShardFormatError(ValueError):
    """Raised when a shard corpus is invalid or fails integrity checks."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(text_or_bytes: str | bytes) -> str:
    """Return the canonical SHA-256 used for source-document content."""

    payload = (
        text_or_bytes.encode("utf-8")
        if isinstance(text_or_bytes, str)
        else text_or_bytes
    )
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_safe_filename(value: object) -> bool:
    return isinstance(value, str) and bool(value) and Path(value).name == value


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and bool(np.isfinite(value))
    )


@dataclass(frozen=True)
class Source:
    """A source represented in the corpus."""

    source_id: str
    name: str
    license: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.source_id or not isinstance(self.source_id, str):
            raise ShardFormatError("source_id must be a non-empty string")
        if not self.name or not isinstance(self.name, str):
            raise ShardFormatError(f"source {self.source_id!r} has no name")
        if not isinstance(self.metadata, Mapping):
            raise ShardFormatError(f"source {self.source_id!r} metadata is not a map")
        _ensure_json_value(self.metadata, f"source {self.source_id!r} metadata")


@dataclass(frozen=True)
class Document:
    """An indexed document record."""

    document_id: int
    source_id: str
    shard: str
    token_start: int
    token_count: int
    content_sha256: str
    quality_score: float | None
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class PackedSegment:
    """Provenance for one document segment placed into a packed sequence."""

    batch_index: int
    packed_start: int
    token_count: int
    document_id: int
    document_token_start: int
    source_id: str


@dataclass(frozen=True)
class PackedBatch:
    """A fully packed causal-LM batch with reset positions at document edges."""

    input_ids: np.ndarray
    labels: np.ndarray
    position_ids: np.ndarray
    segments: tuple[PackedSegment, ...]


def _ensure_json_value(value: Any, label: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ShardFormatError(f"{label} is not valid JSON: {exc}") from exc


class IndexedShardWriter:
    """Write a corpus transactionally and publish its manifest last."""

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        *,
        sources: Sequence[Source],
        tokenizer_sha256: str,
        recipe_sha256: str,
        token_dtype: str = "uint32",
        target_shard_tokens: int = 10_000_000,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        if self.output_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite immutable corpus: {self.output_dir}"
            )
        if token_dtype not in _ALLOWED_DTYPES:
            raise ShardFormatError(
                f"token_dtype must be one of {sorted(_ALLOWED_DTYPES)}"
            )
        if target_shard_tokens <= 0:
            raise ShardFormatError("target_shard_tokens must be positive")
        if not _is_sha256(tokenizer_sha256):
            raise ShardFormatError("tokenizer_sha256 must be a SHA-256 hex digest")
        if not _is_sha256(recipe_sha256):
            raise ShardFormatError("recipe_sha256 must be a SHA-256 hex digest")

        source_list = list(sources)
        if not source_list:
            raise ShardFormatError("at least one source is required")
        for source in source_list:
            source.validate()
        source_ids = [source.source_id for source in source_list]
        if len(set(source_ids)) != len(source_ids):
            raise ShardFormatError("source_id values must be unique")
        _ensure_json_value(metadata or {}, "corpus metadata")

        self.sources = source_list
        self._source_ids = set(source_ids)
        self.tokenizer_sha256 = tokenizer_sha256.lower()
        self.recipe_sha256 = recipe_sha256.lower()
        self.token_dtype = token_dtype
        self._dtype = _ALLOWED_DTYPES[token_dtype]
        self.target_shard_tokens = target_shard_tokens
        self.metadata = dict(metadata or {})

        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        self._work_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{self.output_dir.name}.building-",
                dir=self.output_dir.parent,
            )
        )
        self._index_handle = (self._work_dir / INDEX_FILENAME).open(
            "w", encoding="utf-8", newline="\n"
        )
        self._token_handle = None
        self._shard_number = -1
        self._current_shard_name = ""
        self._current_shard_tokens = 0
        self._current_shard_documents = 0
        self._shards: list[dict[str, Any]] = []
        self._document_count = 0
        self._token_count = 0
        self._closed = False
        self._start_shard()

    def _start_shard(self) -> None:
        self._finalize_shard()
        self._shard_number += 1
        self._current_shard_name = f"tokens-{self._shard_number:05d}.bin"
        self._token_handle = (self._work_dir / self._current_shard_name).open("wb")
        self._current_shard_tokens = 0
        self._current_shard_documents = 0

    def _finalize_shard(self) -> None:
        if self._token_handle is None:
            return
        self._token_handle.flush()
        os.fsync(self._token_handle.fileno())
        self._token_handle.close()
        path = self._work_dir / self._current_shard_name
        self._shards.append(
            {
                "filename": self._current_shard_name,
                "token_count": self._current_shard_tokens,
                "document_count": self._current_shard_documents,
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        self._token_handle = None

    def add_document(
        self,
        tokens: Iterable[int],
        *,
        source_id: str,
        content_sha256: str,
        quality_score: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Document:
        if self._closed:
            raise RuntimeError("writer is closed")
        if source_id not in self._source_ids:
            raise ShardFormatError(f"unknown source_id: {source_id!r}")
        if not _is_sha256(content_sha256):
            raise ShardFormatError("content_sha256 must be a SHA-256 hex digest")
        if quality_score is not None:
            if not _is_finite_number(quality_score):
                raise ShardFormatError("quality_score must be a finite number")
            quality_score = float(quality_score)
        document_metadata = dict(metadata or {})
        _ensure_json_value(document_metadata, "document metadata")

        try:
            token_values = list(tokens)
        except TypeError as exc:
            raise ShardFormatError(f"tokens must be iterable: {exc}") from exc
        if any(
            isinstance(token, bool) or not isinstance(token, Integral)
            for token in token_values
        ):
            raise ShardFormatError("tokens must contain integers only")
        try:
            token_array = np.asarray(token_values, dtype=np.int64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ShardFormatError(f"tokens must be integers: {exc}") from exc
        if token_array.ndim != 1 or token_array.size == 0:
            raise ShardFormatError("a document must contain at least one token")
        maximum = np.iinfo(self._dtype).max
        if np.any(token_array < 0) or np.any(token_array > maximum):
            raise ShardFormatError(
                f"token IDs must be between 0 and {maximum} for {self.token_dtype}"
            )
        if self._current_shard_tokens and (
            self._current_shard_tokens + token_array.size
            > self.target_shard_tokens
        ):
            self._start_shard()

        record = Document(
            document_id=self._document_count,
            source_id=source_id,
            shard=self._current_shard_name,
            token_start=self._current_shard_tokens,
            token_count=int(token_array.size),
            content_sha256=content_sha256.lower(),
            quality_score=quality_score,
            metadata=document_metadata,
        )
        encoded = token_array.astype(self._dtype, copy=False)
        assert self._token_handle is not None
        self._token_handle.write(encoded.tobytes(order="C"))
        self._index_handle.write(
            json.dumps(
                asdict(record),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        self._current_shard_tokens += record.token_count
        self._current_shard_documents += 1
        self._document_count += 1
        self._token_count += record.token_count
        return record

    def close(self) -> Path:
        if self._closed:
            return self.output_dir
        if self._document_count == 0:
            self.abort()
            raise ShardFormatError("cannot publish an empty corpus")

        self._finalize_shard()
        self._index_handle.flush()
        os.fsync(self._index_handle.fileno())
        self._index_handle.close()
        index_path = self._work_dir / INDEX_FILENAME
        manifest: dict[str, Any] = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "token_dtype": self.token_dtype,
            "byte_order": "little",
            "tokenizer_sha256": self.tokenizer_sha256,
            "recipe_sha256": self.recipe_sha256,
            "document_count": self._document_count,
            "token_count": self._token_count,
            "sources": [asdict(source) for source in self.sources],
            "document_index": {
                "filename": INDEX_FILENAME,
                "record_count": self._document_count,
                "byte_size": index_path.stat().st_size,
                "sha256": sha256_file(index_path),
            },
            "shards": self._shards,
            "metadata": self.metadata,
        }
        manifest["corpus_sha256"] = _canonical_hash(manifest)
        manifest_path = self._work_dir / MANIFEST_FILENAME
        with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                manifest,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(self._work_dir, self.output_dir)
        self._closed = True
        return self.output_dir

    def abort(self) -> None:
        if self._closed:
            return
        if self._token_handle is not None:
            self._token_handle.close()
            self._token_handle = None
        if not self._index_handle.closed:
            self._index_handle.close()
        shutil.rmtree(self._work_dir)
        self._closed = True

    def __enter__(self) -> "IndexedShardWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


class IndexedShardReader:
    """Read and fully validate an indexed token corpus."""

    def __init__(
        self,
        corpus_dir: str | os.PathLike[str],
        *,
        expected_tokenizer_sha256: str | None = None,
        expected_recipe_sha256: str | None = None,
        verify_hashes: bool = True,
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        manifest_path = self.corpus_dir / MANIFEST_FILENAME
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                self.manifest = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ShardFormatError(f"cannot read manifest: {exc}") from exc
        self._validate_manifest(
            expected_tokenizer_sha256=expected_tokenizer_sha256,
            expected_recipe_sha256=expected_recipe_sha256,
        )
        if verify_hashes:
            self._verify_files()
        self.documents = tuple(self._read_and_validate_index())
        self._token_arrays: dict[str, np.memmap] = {}

    def _validate_manifest(
        self,
        *,
        expected_tokenizer_sha256: str | None,
        expected_recipe_sha256: str | None,
    ) -> None:
        manifest = self.manifest
        if manifest.get("format") != FORMAT_NAME:
            raise ShardFormatError("unsupported corpus format")
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ShardFormatError("unsupported corpus format version")
        if manifest.get("token_dtype") not in _ALLOWED_DTYPES:
            raise ShardFormatError("unsupported token dtype")
        if manifest.get("byte_order") != "little":
            raise ShardFormatError("unsupported byte order")
        for name in ("tokenizer_sha256", "recipe_sha256", "corpus_sha256"):
            if not _is_sha256(manifest.get(name)):
                raise ShardFormatError(f"invalid {name}")
        if expected_tokenizer_sha256 is not None and (
            manifest["tokenizer_sha256"] != expected_tokenizer_sha256.lower()
        ):
            raise ShardFormatError("tokenizer fingerprint mismatch")
        if expected_recipe_sha256 is not None and (
            manifest["recipe_sha256"] != expected_recipe_sha256.lower()
        ):
            raise ShardFormatError("recipe fingerprint mismatch")
        without_fingerprint = dict(manifest)
        corpus_fingerprint = without_fingerprint.pop("corpus_sha256")
        if _canonical_hash(without_fingerprint) != corpus_fingerprint:
            raise ShardFormatError("manifest fingerprint mismatch")

        sources = manifest.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ShardFormatError("manifest has no sources")
        source_ids = []
        for raw_source in sources:
            try:
                source = Source(**raw_source)
                source.validate()
            except (TypeError, ShardFormatError) as exc:
                raise ShardFormatError(f"invalid source record: {exc}") from exc
            source_ids.append(source.source_id)
        if len(source_ids) != len(set(source_ids)):
            raise ShardFormatError("manifest contains duplicate source IDs")
        self.sources = {source["source_id"]: source for source in sources}

        shards = manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ShardFormatError("manifest has no shards")
        for item in shards:
            if not isinstance(item, dict):
                raise ShardFormatError("invalid shard manifest record")
            if (
                not _is_safe_filename(item.get("filename"))
                or not _is_positive_int(item.get("token_count"))
                or not _is_positive_int(item.get("document_count"))
                or not _is_positive_int(item.get("byte_size"))
            ):
                raise ShardFormatError("invalid shard manifest record")
        shard_names = [item.get("filename") for item in shards]
        if len(shard_names) != len(set(shard_names)):
            raise ShardFormatError("manifest contains duplicate shard names")
        self._shards = {item["filename"]: item for item in shards}
        index_record = manifest.get("document_index")
        if (
            not isinstance(index_record, dict)
            or not _is_safe_filename(index_record.get("filename"))
            or not _is_positive_int(index_record.get("record_count"))
            or not _is_positive_int(index_record.get("byte_size"))
        ):
            raise ShardFormatError("invalid document_index manifest record")
        if (
            manifest.get("document_count") != index_record["record_count"]
            or not _is_positive_int(manifest.get("token_count"))
        ):
            raise ShardFormatError("invalid corpus counts")

    def _verify_file(self, record: Mapping[str, Any]) -> None:
        filename = record.get("filename")
        if not _is_safe_filename(filename):
            raise ShardFormatError(f"unsafe or invalid filename: {filename!r}")
        assert isinstance(filename, str)
        path = self.corpus_dir / filename
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ShardFormatError(f"missing corpus file {filename}: {exc}") from exc
        if size != record.get("byte_size"):
            raise ShardFormatError(f"byte-size mismatch for {filename}")
        if not _is_sha256(record.get("sha256")):
            raise ShardFormatError(f"invalid file hash for {filename}")
        if sha256_file(path) != record["sha256"]:
            raise ShardFormatError(f"SHA-256 mismatch for {filename}")

    def _verify_files(self) -> None:
        index_record = self.manifest.get("document_index")
        self._verify_file(index_record)
        dtype = _ALLOWED_DTYPES[self.manifest["token_dtype"]]
        for record in self.manifest["shards"]:
            self._verify_file(record)
            expected_size = record.get("token_count", -1) * dtype.itemsize
            if record["byte_size"] != expected_size:
                raise ShardFormatError(
                    f"token count does not match byte size for {record['filename']}"
                )

    def _read_and_validate_index(self) -> Iterator[Document]:
        index_record = self.manifest["document_index"]
        index_path = self.corpus_dir / index_record["filename"]
        positions = {name: 0 for name in self._shards}
        counts = {name: 0 for name in self._shards}
        total_tokens = 0
        line_count = 0
        try:
            handle = index_path.open("r", encoding="utf-8")
        except OSError as exc:
            raise ShardFormatError(f"cannot read document index: {exc}") from exc
        with handle:
            for line_number, line in enumerate(handle, start=1):
                line_count = line_number
                try:
                    raw = json.loads(line)
                    document = Document(**raw)
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ShardFormatError(
                        f"invalid document index record on line {line_number}: {exc}"
                    ) from exc
                if (
                    isinstance(document.document_id, bool)
                    or document.document_id != line_number - 1
                ):
                    raise ShardFormatError("document IDs are not contiguous")
                if (
                    not isinstance(document.source_id, str)
                    or document.source_id not in self.sources
                ):
                    raise ShardFormatError(
                        f"unknown source ID in document {document.document_id}"
                    )
                if (
                    not isinstance(document.shard, str)
                    or document.shard not in self._shards
                ):
                    raise ShardFormatError(
                        f"unknown shard in document {document.document_id}"
                    )
                if not _is_positive_int(document.token_count):
                    raise ShardFormatError("document token_count must be positive")
                if (
                    not isinstance(document.token_start, int)
                    or isinstance(document.token_start, bool)
                    or document.token_start != positions[document.shard]
                ):
                    raise ShardFormatError(
                        f"non-contiguous token range in {document.shard}"
                    )
                if not _is_sha256(document.content_sha256):
                    raise ShardFormatError("invalid document content hash")
                if (
                    document.quality_score is not None
                    and not _is_finite_number(document.quality_score)
                ):
                    raise ShardFormatError("invalid document quality score")
                if not isinstance(document.metadata, Mapping):
                    raise ShardFormatError("document metadata is not a map")
                _ensure_json_value(document.metadata, "document metadata")
                positions[document.shard] += document.token_count
                counts[document.shard] += 1
                total_tokens += document.token_count
                yield document

        if line_count != index_record.get("record_count"):
            raise ShardFormatError("document index record count mismatch")
        if len(positions) and any(
            positions[name] != self._shards[name].get("token_count")
            or counts[name] != self._shards[name].get("document_count")
            for name in positions
        ):
            raise ShardFormatError("document index totals do not match shard manifest")
        if line_count != self.manifest.get("document_count"):
            raise ShardFormatError("manifest document count mismatch")
        if total_tokens != self.manifest.get("token_count"):
            raise ShardFormatError("manifest token count mismatch")

    def iter_documents(self) -> Iterator[Document]:
        return iter(self.documents)

    def read_tokens(self, document: Document | int) -> np.ndarray:
        if isinstance(document, int):
            try:
                document = self.documents[document]
            except IndexError as exc:
                raise KeyError(f"unknown document ID: {document}") from exc
        elif (
            document.document_id >= len(self.documents)
            or self.documents[document.document_id] != document
        ):
            raise ShardFormatError("document does not belong to this corpus")
        dtype = _ALLOWED_DTYPES[self.manifest["token_dtype"]]
        if document.shard not in self._token_arrays:
            path = self.corpus_dir / document.shard
            self._token_arrays[document.shard] = np.memmap(
                path,
                mode="r",
                dtype=dtype,
            )
        shard_tokens = self._token_arrays[document.shard]
        start = document.token_start
        stop = start + document.token_count
        return np.asarray(
            shard_tokens[start:stop],
            dtype=np.dtype(self.manifest["token_dtype"]),
        ).copy()


class ResumableIndexedShardWriter:
    """Append to a checkpointed staging corpus and atomically seal it.

    Every checkpoint finalizes the current shard and atomically rewrites a
    complete manifest. Files not referenced by that manifest are uncommitted
    crash tails and are removed on resume.
    """

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        *,
        sources: Sequence[Source],
        tokenizer_sha256: str,
        recipe_sha256: str,
        token_dtype: str = "uint32",
        target_shard_tokens: int = 10_000_000,
        metadata: Mapping[str, Any] | None = None,
        resume: bool = False,
        staging_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.staging_dir = (
            Path(staging_dir)
            if staging_dir is not None
            else self.output_dir.with_name(self.output_dir.name + ".staging")
        )
        if self.output_dir.exists():
            raise FileExistsError(f"final corpus already exists: {self.output_dir}")
        if token_dtype not in _ALLOWED_DTYPES:
            raise ShardFormatError(
                f"token_dtype must be one of {sorted(_ALLOWED_DTYPES)}"
            )
        if target_shard_tokens <= 0:
            raise ShardFormatError("target_shard_tokens must be positive")
        if not _is_sha256(tokenizer_sha256) or not _is_sha256(recipe_sha256):
            raise ShardFormatError("tokenizer and recipe fingerprints must be SHA-256")
        source_list = list(sources)
        if not source_list:
            raise ShardFormatError("at least one source is required")
        for source in source_list:
            source.validate()
        if len({source.source_id for source in source_list}) != len(source_list):
            raise ShardFormatError("source_id values must be unique")
        _ensure_json_value(metadata or {}, "corpus metadata")

        self.sources = source_list
        self._source_ids = {source.source_id for source in source_list}
        self.tokenizer_sha256 = tokenizer_sha256.lower()
        self.recipe_sha256 = recipe_sha256.lower()
        self.token_dtype = token_dtype
        self._dtype = _ALLOWED_DTYPES[token_dtype]
        self.target_shard_tokens = int(target_shard_tokens)
        self.metadata = dict(metadata or {})
        self._closed = False
        self._token_handle = None
        self._current_shard_name = ""
        self._current_shard_tokens = 0
        self._current_shard_documents = 0

        if resume:
            self._restore_committed_state()
        else:
            if self.staging_dir.exists():
                raise FileExistsError(
                    f"staging corpus already exists: {self.staging_dir}; "
                    "resume it or remove it explicitly"
                )
            self.staging_dir.parent.mkdir(parents=True, exist_ok=True)
            self.staging_dir.mkdir()
            self._created_at = datetime.now(timezone.utc).isoformat()
            self._shards: list[dict[str, Any]] = []
            self._document_count = 0
            self._token_count = 0
            self.existing_documents: tuple[Document, ...] = ()
            self._index_handle = (self.staging_dir / INDEX_FILENAME).open(
                "w", encoding="utf-8", newline="\n"
            )
            self._shard_number = -1
        self._start_shard()

    def _restore_committed_state(self) -> None:
        if not self.staging_dir.is_dir():
            raise FileNotFoundError(
                f"resumable staging corpus not found: {self.staging_dir}"
            )
        reader = IndexedShardReader(self.staging_dir)
        manifest = reader.manifest
        expected_sources = [asdict(source) for source in self.sources]
        checks = (
            (manifest["tokenizer_sha256"], self.tokenizer_sha256, "tokenizer"),
            (manifest["recipe_sha256"], self.recipe_sha256, "recipe"),
            (manifest["token_dtype"], self.token_dtype, "token dtype"),
            (manifest.get("sources"), expected_sources, "sources"),
            (manifest.get("metadata"), self.metadata, "metadata"),
        )
        for actual, expected, label in checks:
            if actual != expected:
                raise ShardFormatError(
                    f"staging {label} mismatch; refuse resumable append"
                )
        self._created_at = manifest["created_at"]
        self._shards = list(manifest["shards"])
        self._document_count = int(manifest["document_count"])
        self._token_count = int(manifest["token_count"])
        self.existing_documents = reader.documents
        self._resume_reader = reader

        committed_names = {item["filename"] for item in self._shards}
        for candidate in self.staging_dir.glob("tokens-*.bin"):
            if candidate.name not in committed_names:
                candidate.unlink()
        index_record = manifest["document_index"]
        index_path = self.staging_dir / index_record["filename"]
        os.truncate(index_path, int(index_record["byte_size"]))
        self._index_handle = index_path.open("a", encoding="utf-8", newline="\n")
        numbers = [
            int(item["filename"].removeprefix("tokens-").removesuffix(".bin"))
            for item in self._shards
        ]
        self._shard_number = max(numbers, default=-1)

    def read_committed_tokens(self, document_id: int) -> np.ndarray:
        if not hasattr(self, "_resume_reader"):
            raise KeyError("writer was not opened from a committed staging corpus")
        return self._resume_reader.read_tokens(document_id)

    @property
    def document_count(self) -> int:
        return self._document_count

    @property
    def token_count(self) -> int:
        return self._token_count

    def _start_shard(self) -> None:
        self._shard_number += 1
        self._current_shard_name = f"tokens-{self._shard_number:05d}.bin"
        self._token_handle = (
            self.staging_dir / self._current_shard_name
        ).open("xb")
        self._current_shard_tokens = 0
        self._current_shard_documents = 0

    def _finalize_current_shard(self) -> bool:
        if self._token_handle is None:
            return False
        self._token_handle.flush()
        os.fsync(self._token_handle.fileno())
        self._token_handle.close()
        self._token_handle = None
        path = self.staging_dir / self._current_shard_name
        if self._current_shard_documents == 0:
            path.unlink()
            return False
        self._shards.append(
            {
                "filename": self._current_shard_name,
                "token_count": self._current_shard_tokens,
                "document_count": self._current_shard_documents,
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        return True

    def add_document(
        self,
        tokens: Iterable[int],
        *,
        source_id: str,
        content_sha256: str,
        quality_score: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Document:
        if self._closed:
            raise RuntimeError("writer is closed")
        if source_id not in self._source_ids:
            raise ShardFormatError(f"unknown source_id: {source_id!r}")
        if not _is_sha256(content_sha256):
            raise ShardFormatError("content_sha256 must be a SHA-256 hex digest")
        if quality_score is not None and not _is_finite_number(quality_score):
            raise ShardFormatError("quality_score must be a finite number")
        document_metadata = dict(metadata or {})
        _ensure_json_value(document_metadata, "document metadata")
        token_values = list(tokens)
        if not token_values or any(
            isinstance(token, bool) or not isinstance(token, Integral)
            for token in token_values
        ):
            raise ShardFormatError("tokens must contain at least one integer")
        token_array = np.asarray(token_values, dtype=np.int64)
        maximum = np.iinfo(self._dtype).max
        if np.any(token_array < 0) or np.any(token_array > maximum):
            raise ShardFormatError(
                f"token IDs must be between 0 and {maximum} for {self.token_dtype}"
            )
        if self._current_shard_tokens and (
            self._current_shard_tokens + token_array.size
            > self.target_shard_tokens
        ):
            self.checkpoint()

        record = Document(
            document_id=self._document_count,
            source_id=source_id,
            shard=self._current_shard_name,
            token_start=self._current_shard_tokens,
            token_count=int(token_array.size),
            content_sha256=content_sha256.lower(),
            quality_score=(
                float(quality_score) if quality_score is not None else None
            ),
            metadata=document_metadata,
        )
        assert self._token_handle is not None
        encoded = token_array.astype(self._dtype, copy=False)
        self._token_handle.write(encoded.tobytes(order="C"))
        self._index_handle.write(
            json.dumps(
                asdict(record),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )
        self._current_shard_tokens += record.token_count
        self._current_shard_documents += 1
        self._document_count += 1
        self._token_count += record.token_count
        return record

    def _write_manifest(self) -> None:
        self._index_handle.flush()
        os.fsync(self._index_handle.fileno())
        index_path = self.staging_dir / INDEX_FILENAME
        manifest: dict[str, Any] = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "created_at": self._created_at,
            "token_dtype": self.token_dtype,
            "byte_order": "little",
            "tokenizer_sha256": self.tokenizer_sha256,
            "recipe_sha256": self.recipe_sha256,
            "document_count": self._document_count,
            "token_count": self._token_count,
            "sources": [asdict(source) for source in self.sources],
            "document_index": {
                "filename": INDEX_FILENAME,
                "record_count": self._document_count,
                "byte_size": index_path.stat().st_size,
                "sha256": sha256_file(index_path),
            },
            "shards": self._shards,
            "metadata": self.metadata,
        }
        manifest["corpus_sha256"] = _canonical_hash(manifest)
        target = self.staging_dir / MANIFEST_FILENAME
        temporary = self.staging_dir / f"{MANIFEST_FILENAME}.tmp"
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                manifest,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)

    def checkpoint(self) -> None:
        """Commit all appended documents and begin a fresh tail shard."""
        if self._closed or self._current_shard_documents == 0:
            return
        self._finalize_current_shard()
        self._write_manifest()
        self._start_shard()

    def suspend(self) -> None:
        """Commit progress but leave the corpus in staging for a later resume."""
        if self._closed:
            return
        if self._document_count == 0:
            if self._token_handle is not None:
                self._token_handle.close()
            self._index_handle.close()
            shutil.rmtree(self.staging_dir)
            self._closed = True
            return
        if self._current_shard_documents:
            self._finalize_current_shard()
            self._write_manifest()
        else:
            self._finalize_current_shard()
        self._index_handle.close()
        self._closed = True

    def discard_uncommitted(self) -> None:
        """Close and remove only the tail that is absent from the manifest."""
        if self._closed:
            return
        if self._token_handle is not None:
            self._token_handle.close()
            tail = self.staging_dir / self._current_shard_name
            if tail.exists():
                tail.unlink()
        self._index_handle.close()
        manifest_path = self.staging_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            shutil.rmtree(self.staging_dir)
            self._closed = True
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        os.truncate(
            self.staging_dir / INDEX_FILENAME,
            int(manifest["document_index"]["byte_size"]),
        )
        self._closed = True

    def close(self) -> Path:
        if self._closed:
            return self.output_dir
        if self._document_count == 0:
            self.discard_uncommitted()
            raise ShardFormatError("cannot publish an empty corpus")
        if self._current_shard_documents:
            self._finalize_current_shard()
        else:
            self._finalize_current_shard()
        self._write_manifest()
        self._index_handle.close()
        os.replace(self.staging_dir, self.output_dir)
        self._closed = True
        return self.output_dir


class IndexedCorpusSampler:
    """Deterministically build source-pure packed rows from an indexed corpus.

    Each batch row chooses one source, then fills the row with uniformly sampled
    document segments from that source. Position IDs restart for each segment,
    which Transformers 5 recognizes as packed-sequence boundaries and converts
    into block-diagonal causal attention. Labels at segment starts are ignored
    so no document is trained to predict the first token of the next document.

    The sampler has no mutable random state: callers provide a NumPy generator.
    Saving that generator is therefore sufficient for exact batch resume.
    """

    def __init__(
        self,
        reader: IndexedShardReader,
        *,
        source_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.reader = reader
        documents_by_source: dict[str, list[Document]] = {
            source_id: [] for source_id in reader.sources
        }
        for document in reader.documents:
            documents_by_source[document.source_id].append(document)
        documents_by_source = {
            source_id: documents
            for source_id, documents in documents_by_source.items()
            if documents
        }
        if not documents_by_source:
            raise ShardFormatError("indexed corpus contains no sampleable documents")

        if source_weights is None:
            weights = {
                source_id: float(sum(doc.token_count for doc in documents))
                for source_id, documents in documents_by_source.items()
            }
        else:
            unknown = set(source_weights) - set(documents_by_source)
            if unknown:
                raise ShardFormatError(
                    "source_weights contains unknown or empty sources: "
                    + ", ".join(sorted(unknown))
                )
            weights = {}
            for source_id, raw_weight in source_weights.items():
                if not _is_finite_number(raw_weight) or float(raw_weight) < 0:
                    raise ShardFormatError(
                        f"source weight for {source_id!r} must be finite and non-negative"
                    )
                if float(raw_weight) > 0:
                    weights[source_id] = float(raw_weight)
        if not weights or sum(weights.values()) <= 0:
            raise ShardFormatError("source_weights must contain positive total weight")

        self.source_ids = tuple(sorted(weights))
        raw_weights = np.asarray(
            [weights[source_id] for source_id in self.source_ids],
            dtype=np.float64,
        )
        self.source_probabilities = raw_weights / raw_weights.sum()
        self._documents: dict[str, tuple[Document, ...]] = {}
        for source_id in self.source_ids:
            self._documents[source_id] = tuple(documents_by_source[source_id])

    def _sample_document(
        self,
        source_id: str,
        rng: np.random.Generator,
    ) -> Document:
        documents = self._documents[source_id]
        return documents[int(rng.integers(0, len(documents)))]

    @staticmethod
    def _supervision_spans(document: Document) -> tuple[tuple[int, int], ...] | None:
        raw_spans = document.metadata.get("supervision_spans")
        if raw_spans is None:
            return None
        if not isinstance(raw_spans, list) or not raw_spans:
            raise ShardFormatError(
                f"document {document.document_id} has empty or invalid supervision_spans"
            )
        spans: list[tuple[int, int]] = []
        for raw_span in raw_spans:
            if (
                not isinstance(raw_span, list)
                or len(raw_span) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_span)
            ):
                raise ShardFormatError(
                    f"document {document.document_id} has invalid supervision span"
                )
            start, stop = raw_span
            if start < 0 or stop <= start or stop > document.token_count:
                raise ShardFormatError(
                    f"document {document.document_id} supervision span is out of bounds"
                )
            spans.append((start, stop))
        return tuple(spans)

    @staticmethod
    def _window_start_with_supervision(
        document: Document,
        take: int,
        spans: tuple[tuple[int, int], ...],
        rng: np.random.Generator,
    ) -> int:
        """Choose a crop that contains a target after the causal boundary."""

        candidates = [(start, stop) for start, stop in spans if stop > 1]
        if not candidates:
            raise ShardFormatError(
                f"document {document.document_id} has no causally learnable target"
            )
        span_start, span_stop = candidates[int(rng.integers(0, len(candidates)))]
        target_low = max(span_start, 1)
        target = int(rng.integers(target_low, span_stop))
        latest = min(target - 1, document.token_count - take)
        earliest = max(0, target - take + 1)
        if latest < earliest:
            return max(0, min(document.token_count - take, target - 1))
        return int(rng.integers(earliest, latest + 1))

    def sample_batch(
        self,
        *,
        batch_size: int,
        block_size: int,
        rng: np.random.Generator,
        source_id: str | None = None,
    ) -> PackedBatch:
        if batch_size <= 0 or block_size <= 1:
            raise ShardFormatError("batch_size must be positive and block_size must exceed 1")

        dtype = _ALLOWED_DTYPES[self.reader.manifest["token_dtype"]]
        input_ids = np.empty((batch_size, block_size), dtype=dtype)
        labels = np.empty((batch_size, block_size), dtype=np.int64)
        position_ids = np.empty((batch_size, block_size), dtype=np.int64)
        segments: list[PackedSegment] = []
        if source_id is not None and source_id not in self._documents:
            raise ShardFormatError(f"unknown sample source: {source_id!r}")

        for batch_index in range(batch_size):
            if source_id is None:
                source_index = int(
                    rng.choice(len(self.source_ids), p=self.source_probabilities)
                )
                row_source_id = self.source_ids[source_index]
            else:
                row_source_id = source_id
            packed_start = 0
            while packed_start < block_size:
                document = self._sample_document(row_source_id, rng)
                document_tokens = self.reader.read_tokens(document)
                supervision_spans = self._supervision_spans(document)
                remaining = block_size - packed_start
                take = min(remaining, document.token_count)
                if document.token_count > take:
                    if supervision_spans is None:
                        document_start = int(
                            rng.integers(0, document.token_count - take + 1)
                        )
                    else:
                        document_start = self._window_start_with_supervision(
                            document,
                            take,
                            supervision_spans,
                            rng,
                        )
                else:
                    document_start = 0
                document_stop = document_start + take
                packed_stop = packed_start + take
                input_ids[batch_index, packed_start:packed_stop] = document_tokens[
                    document_start:document_stop
                ]
                sampled_tokens = document_tokens[document_start:document_stop]
                if supervision_spans is None:
                    sampled_labels = sampled_tokens.astype(np.int64, copy=False)
                else:
                    try:
                        sampled_labels = apply_supervision_spans(
                            sampled_tokens,
                            document_start=document_start,
                            spans=supervision_spans,
                        )
                    except ChatFormatError as exc:
                        raise ShardFormatError(
                            f"document {document.document_id} supervision is invalid: {exc}"
                        ) from exc
                labels[batch_index, packed_start:packed_stop] = sampled_labels
                labels[batch_index, packed_start] = -100
                position_ids[batch_index, packed_start:packed_stop] = np.arange(
                    take, dtype=np.int64
                )
                segments.append(
                    PackedSegment(
                        batch_index=batch_index,
                        packed_start=packed_start,
                        token_count=take,
                        document_id=document.document_id,
                        document_token_start=document_start,
                        source_id=row_source_id,
                    )
                )
                packed_start = packed_stop

        return PackedBatch(
            input_ids=input_ids,
            labels=labels,
            position_ids=position_ids,
            segments=tuple(segments),
        )

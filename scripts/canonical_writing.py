#!/usr/bin/env python3
"""Convert manually accepted writing samples into a verified canonical lane."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml


SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _load_registry(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("sources"), dict):
        raise ValueError(f"invalid source registry: {path}")
    return value


def _load_decisions(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    decisions = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["source_id"], row["document_id"])
            if key in decisions:
                raise ValueError(f"duplicate review decision: {key}")
            decisions[key] = row
    return decisions


def iter_accepted_documents(
    *,
    registry_path: Path,
    acquisition_manifest_path: Path,
    scores_path: Path,
    root: Path = ROOT,
) -> Iterator[dict[str, Any]]:
    registry = _load_registry(registry_path)
    manifest = json.loads(
        acquisition_manifest_path.read_text(encoding="utf-8")
    )
    decisions = _load_decisions(scores_path)

    for source_id, acquired in sorted(manifest.get("sources", {}).items()):
        policy = registry["sources"].get(source_id)
        if not isinstance(policy, dict):
            raise ValueError(f"acquired source absent from registry: {source_id}")
        if policy.get("tier") != "A":
            continue
        if policy.get("redistribution") != "redistributable":
            continue
        documents_path = _resolve(acquired["documents_path"], root)
        if sha256_file(documents_path) != acquired["documents_sha256"]:
            raise ValueError(f"acquisition file hash mismatch: {documents_path}")
        with documents_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                document = json.loads(line)
                if document.get("source_id") != source_id:
                    raise ValueError(
                        f"{documents_path}:{line_number}: source mismatch"
                    )
                text = document.get("text")
                document_id = document.get("document_id")
                if not isinstance(text, str) or sha256_text(text) != document_id:
                    raise ValueError(
                        f"{documents_path}:{line_number}: content hash mismatch"
                    )
                decision = decisions.get((source_id, document_id))
                if not decision:
                    continue
                keep = str(decision.get("keep_yes_no", "")).strip().lower()
                if keep not in {"yes", "y", "keep"}:
                    continue
                yield {
                    "schema_version": SCHEMA_VERSION,
                    "source_id": source_id,
                    "document_id": document_id,
                    "text": text,
                    "url": document.get("url"),
                    "license": document.get("license") or policy.get("license"),
                    "roles": policy.get("roles") or [],
                    "metadata": {
                        "source_name": policy.get("name"),
                        "acquisition_run_id": manifest.get("run_id"),
                        "original_metadata": document.get("metadata") or {},
                        "review": {
                            key: value
                            for key, value in decision.items()
                            if key
                            not in {
                                "source_id",
                                "document_id",
                                "document_index",
                                "url",
                            }
                        },
                    },
                }


def build_canonical_lane(
    *,
    registry_path: Path,
    acquisition_manifest_path: Path,
    scores_path: Path,
    output_dir: Path,
    root: Path = ROOT,
) -> tuple[Path, dict[str, Any]]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite canonical lane: {output_dir}")
    documents = list(
        iter_accepted_documents(
            registry_path=registry_path,
            acquisition_manifest_path=acquisition_manifest_path,
            scores_path=scores_path,
            root=root,
        )
    )
    if not documents:
        raise ValueError(
            "no manually accepted documents; complete keep_yes_no decisions "
            "before building a training lane"
        )
    output_dir.mkdir(parents=True)
    documents_path = output_dir / "documents.jsonl"
    with documents_path.open("x", encoding="utf-8", newline="\n") as handle:
        for document in documents:
            handle.write(
                json.dumps(
                    document,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())

    sources = {}
    registry = _load_registry(registry_path)
    for source_id in sorted({item["source_id"] for item in documents}):
        policy = registry["sources"][source_id]
        sources[source_id] = {
            "name": policy.get("name"),
            "license": policy.get("license"),
            "roles": policy.get("roles") or [],
            "document_count": sum(
                item["source_id"] == source_id for item in documents
            ),
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len(documents),
        "documents_path": str(documents_path.resolve()),
        "documents_sha256": sha256_file(documents_path),
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": sha256_file(registry_path),
        "acquisition_manifest_path": str(acquisition_manifest_path.resolve()),
        "acquisition_manifest_sha256": sha256_file(acquisition_manifest_path),
        "scores_path": str(scores_path.resolve()),
        "scores_sha256": sha256_file(scores_path),
        "selection_policy": "manual_keep_yes_no",
        "sources": sources,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return manifest_path, manifest


def iter_canonical_manifest(
    manifest_path: Path,
) -> Iterator[tuple[str, str, dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents_path = Path(manifest["documents_path"])
    if sha256_file(documents_path) != manifest["documents_sha256"]:
        raise ValueError(f"canonical document hash mismatch: {documents_path}")
    with documents_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            document = json.loads(line)
            text = document.get("text")
            if (
                not isinstance(text, str)
                or sha256_text(text) != document.get("document_id")
            ):
                raise ValueError(
                    f"{documents_path}:{line_number}: canonical hash mismatch"
                )
            metadata = dict(document.get("metadata") or {})
            metadata.update(
                {
                    "canonical_document_id": document["document_id"],
                    "url": document.get("url"),
                    "license": document.get("license"),
                    "roles": document.get("roles") or [],
                }
            )
            yield document["source_id"], text, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=ROOT / "data/sources.yaml")
    parser.add_argument("--acquisition-manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        path, manifest = build_canonical_lane(
            registry_path=args.registry,
            acquisition_manifest_path=args.acquisition_manifest,
            scores_path=args.scores,
            output_dir=args.output_dir,
        )
    except (FileExistsError, OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"Canonical lane not built: {exc}") from exc
    print(f"Canonical lane: {path} ({manifest['document_count']} documents)")


if __name__ == "__main__":
    main()

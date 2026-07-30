#!/usr/bin/env python3
"""Create and apply human review decisions for conversational JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from scripts.indexed_shards import sha256_file


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not value.get("id"):
                raise ValueError(f"{path}:{line_number}: record requires an id")
            records.append(value)
    return records


def create_pack(args: argparse.Namespace) -> None:
    records = _read_jsonl(args.input)
    by_id = {str(record["id"]): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("input contains duplicate record IDs")
    if not records:
        raise ValueError("input contains no conversations")
    if args.sample_size is not None and args.sample_size <= 0:
        raise ValueError("sample_size must be positive")
    rng = random.Random(args.seed)
    selected_ids = sorted(
        rng.sample(
            list(by_id),
            min(args.sample_size or len(by_id), len(by_id)),
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    markdown = [
        "# Conversation data review",
        "",
        "Read the complete exchange, then fill `review.csv`. `keep` must be "
        "`yes` or `no`; blank decisions remain excluded.",
        "",
    ]
    rows = []
    for record_id in selected_ids:
        record = by_id[record_id]
        markdown.extend(
            [
                f"## {record_id}",
                "",
                f"Source: `{record.get('source_id', 'unknown')}`",
                "",
            ]
        )
        for message in record.get("messages", []):
            markdown.extend(
                [
                    f"**{str(message.get('role', '')).title()}:** "
                    f"{message.get('content', '')}",
                    "",
                ]
            )
        rows.append(
            {
                "record_id": record_id,
                "keep": "",
                "naturalness_1_5": "",
                "context_1_5": "",
                "correctness_1_5": "",
                "conciseness_1_5": "",
                "flags": "",
                "notes": "",
            }
        )
    (args.output_dir / "review.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "review.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 1,
        "kind": "chat-human-review-pack",
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "seed": args.seed,
        "sample_size": len(selected_ids),
        "record_ids": selected_ids,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} conversations to {args.output_dir}")


def _parse_keep(value: str) -> bool | None:
    normalized = value.strip().casefold()
    if normalized in {"yes", "y", "true", "1", "keep"}:
        return True
    if normalized in {"no", "n", "false", "0", "reject"}:
        return False
    if not normalized:
        return None
    raise ValueError(f"invalid keep decision {value!r}")


def apply_decisions(args: argparse.Namespace) -> None:
    records = _read_jsonl(args.input)
    by_id = {str(record["id"]): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("input contains duplicate record IDs")
    review_manifest_path = args.review_csv.parent / "manifest.json"
    if not review_manifest_path.exists():
        raise ValueError(
            f"review pack manifest is missing: {review_manifest_path}"
        )
    review_manifest = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    if review_manifest.get("kind") != "chat-human-review-pack":
        raise ValueError(f"unexpected review pack kind in {review_manifest_path}")
    actual_input_sha256 = sha256_file(args.input)
    if review_manifest.get("input_sha256") != actual_input_sha256:
        raise ValueError(
            "review pack input hash does not match --input; recreate the review pack"
        )
    with args.review_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    row_ids = [row.get("record_id", "") for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("review CSV contains duplicate record IDs")
    if sorted(row_ids) != sorted(review_manifest.get("record_ids", [])):
        raise ValueError("review CSV record IDs do not match the review pack manifest")
    accepted = []
    undecided = []
    decisions = []
    for row in rows:
        record_id = row.get("record_id", "")
        if record_id not in by_id:
            raise ValueError(f"review references unknown record {record_id!r}")
        keep = _parse_keep(row.get("keep", ""))
        if keep is None:
            undecided.append(record_id)
            continue
        decisions.append({"record_id": record_id, "keep": keep})
        if keep:
            record = dict(by_id[record_id])
            record["human_review"] = {
                "keep": True,
                "naturalness_1_5": row.get("naturalness_1_5"),
                "context_1_5": row.get("context_1_5"),
                "correctness_1_5": row.get("correctness_1_5"),
                "conciseness_1_5": row.get("conciseness_1_5"),
                "flags": row.get("flags"),
                "notes": row.get("notes"),
            }
            accepted.append(record)
    if undecided and args.require_complete:
        raise ValueError(
            f"{len(undecided)} review decisions are blank; first: {undecided[:5]}"
        )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in accepted:
            handle.write(
                json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
            )
    manifest = {
        "schema_version": 1,
        "kind": "human-accepted-chat-data",
        "input_sha256": actual_input_sha256,
        "review_pack_manifest_sha256": sha256_file(review_manifest_path),
        "review_csv_sha256": sha256_file(args.review_csv),
        "accepted": len(accepted),
        "rejected": sum(not decision["keep"] for decision in decisions),
        "undecided_excluded": len(undecided),
        "output_sha256": sha256_file(args.output),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"accepted={len(accepted)} rejected={manifest['rejected']} "
        f"undecided_excluded={len(undecided)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--input", required=True, type=Path)
    create.add_argument("--output_dir", required=True, type=Path)
    create.add_argument("--sample_size", type=int, default=None)
    create.add_argument("--seed", type=int, default=1337)
    create.set_defaults(function=create_pack)
    apply = subparsers.add_parser("apply")
    apply.add_argument("--input", required=True, type=Path)
    apply.add_argument("--review_csv", required=True, type=Path)
    apply.add_argument("--output", required=True, type=Path)
    apply.add_argument("--require_complete", action="store_true")
    apply.set_defaults(function=apply_decisions)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.function(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()

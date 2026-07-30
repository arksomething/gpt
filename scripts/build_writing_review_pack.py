#!/usr/bin/env python3
"""Build human review packs and scrollable text from an acquisition manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.filters import is_prose, shingle_jaccard, wikimedia_is_article


DEFAULT_SAMPLES_ROOT = Path("data/source_samples")
DEFAULT_REVIEW_ROOT = Path("data/review")

NEAR_DUPLICATE_THRESHOLD = 0.7

BOILERPLATE_PATTERNS = {
    "skip_navigation": re.compile(r"\bskip to (?:main )?content\b", re.I),
    "cookie_notice": re.compile(r"\b(?:accept|manage) (?:all )?cookies?\b", re.I),
    "newsletter_prompt": re.compile(r"\b(?:sign up|subscribe) (?:for|to) (?:our|the)\b", re.I),
    "social_share": re.compile(r"\bshare (?:this|on facebook|on twitter)\b", re.I),
    "gutenberg_header": re.compile(r"\bproject gutenberg(?: literary archive)?\b", re.I),
    "digitizer_credit": re.compile(r"\bproduced by .{0,120}(?:proofread|distributed)\b", re.I),
    "publisher_promo": re.compile(r"\bwe are intechopen\b", re.I),
}

SYNTHETIC_MARKERS = {
    "ai_disclosure": re.compile(
        r"\b(?:as an ai language model|generated (?:with|by) (?:an? )?ai|"
        r"ai-generated|large language model generated)\b",
        re.I,
    ),
    "knowledge_cutoff": re.compile(r"\bknowledge cutoff\b", re.I),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _latest_manifest(samples_root: Path) -> Path:
    manifests = sorted(samples_root.glob("*/manifest.json"))
    if not manifests:
        raise SystemExit(f"No acquisition manifests found under {samples_root}")
    return manifests[-1]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    documents = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return documents


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _repeated_line_fraction(text: str) -> float:
    normalized = [
        re.sub(r"\s+", " ", line).strip().lower()
        for line in text.splitlines()
        if len(line.strip()) >= 40
    ]
    if not normalized:
        return 0.0
    counts = Counter(normalized)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(normalized)


def _excerpt(text: str, position: str, length: int) -> str:
    if len(text) <= length:
        return text.strip()
    if position == "beginning":
        start = 0
    elif position == "middle":
        start = max(0, len(text) // 2 - length // 2)
    elif position == "ending":
        start = max(0, len(text) - length)
    else:
        raise ValueError(position)
    excerpt = text[start : start + length]
    if start:
        first_space = excerpt.find(" ")
        if first_space >= 0:
            excerpt = excerpt[first_space + 1 :]
    if start + length < len(text):
        last_space = excerpt.rfind(" ")
        if last_space >= 0:
            excerpt = excerpt[:last_space]
    return excerpt.strip()


def _blockquote(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def _analyze_document(
    document: dict[str, Any],
    max_chars: int,
    earlier_texts: list[str] | None = None,
) -> dict[str, Any]:
    text = document["text"]
    words = re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]
    paragraphs = _paragraphs(text)
    alpha = sum(character.isalpha() for character in text)
    digits = sum(character.isdigit() for character in text)
    non_ascii = sum(ord(character) > 127 for character in text)
    visible = max(1, sum(not character.isspace() for character in text))
    boilerplate = [
        name for name, pattern in BOILERPLATE_PATTERNS.items() if pattern.search(text)
    ]
    synthetic_markers = [
        name for name, pattern in SYNTHETIC_MARKERS.items() if pattern.search(text)
    ]
    repeated_fraction = _repeated_line_fraction(text)
    flags: list[str] = []
    if len(words) < 200:
        flags.append("short")
    if len(paragraphs) < 3 and len(words) >= 500:
        flags.append("weak_paragraph_structure")
    if alpha / visible < 0.55:
        flags.append("low_alpha_fraction")
    if repeated_fraction > 0.10:
        flags.append("repeated_lines")
    if "\ufffd" in text:
        flags.append("replacement_characters")
    if len(boilerplate) >= 2:
        flags.append("boilerplate")
    if synthetic_markers:
        flags.append("synthetic_marker")
    if max_chars > 0 and len(text) >= int(max_chars * 0.99):
        flags.append("sample_truncated")
    if not is_prose(text):
        flags.append("non_prose")
    if document["source_id"] == "common_pile_wikimedia" and not wikimedia_is_article(text):
        flags.append("non_article_namespace")
    if earlier_texts and any(
        shingle_jaccard(text, earlier) >= NEAR_DUPLICATE_THRESHOLD
        for earlier in earlier_texts
    ):
        flags.append("near_duplicate")

    severe_flags = {
        "short",
        "weak_paragraph_structure",
        "low_alpha_fraction",
        "repeated_lines",
        "replacement_characters",
        "boilerplate",
        "synthetic_marker",
        "non_prose",
        "non_article_namespace",
        "near_duplicate",
    }
    return {
        "source_id": document["source_id"],
        "document_id": document["document_id"],
        "url": document.get("url"),
        "chars": len(text),
        "words": len(words),
        "paragraphs": len(paragraphs),
        "sentences": len(sentences),
        "average_sentence_words": round(
            len(words) / max(1, len(sentences)),
            2,
        ),
        "alpha_fraction": round(alpha / visible, 4),
        "digit_fraction": round(digits / visible, 4),
        "non_ascii_fraction": round(non_ascii / max(1, len(text)), 4),
        "url_count": len(re.findall(r"https?://", text)),
        "replacement_character_count": text.count("\ufffd"),
        "repeated_line_fraction": round(repeated_fraction, 4),
        "boilerplate_hits": boilerplate,
        "synthetic_marker_hits": synthetic_markers,
        "flags": flags,
        "automatic_artifact_clean": not bool(severe_flags.intersection(flags)),
    }


def _write_raw_source(
    path: Path,
    source_id: str,
    source_manifest: dict[str, Any],
    documents: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"SOURCE: {source_id}\n")
        handle.write(f"NAME: {source_manifest['name']}\n")
        handle.write(f"LICENSE: {source_manifest['license']}\n")
        handle.write(f"DOCUMENTS: {len(documents)}\n")
        handle.write("=" * 100 + "\n\n")
        for index, document in enumerate(documents, start=1):
            handle.write(f"DOCUMENT {index:03d}/{len(documents):03d}\n")
            handle.write(f"ID: {document['document_id']}\n")
            handle.write(f"URL: {document.get('url') or '(none)'}\n")
            metadata = json.dumps(
                document.get("metadata", {}),
                ensure_ascii=False,
                sort_keys=True,
            )
            handle.write(f"METADATA: {metadata}\n")
            handle.write("-" * 100 + "\n")
            handle.write(document["text"].strip())
            handle.write("\n\n" + "=" * 100 + "\n\n")


def _write_source_pack(
    path: Path,
    source_id: str,
    source_manifest: dict[str, Any],
    documents: list[dict[str, Any]],
    analyses: list[dict[str, Any]],
    excerpt_chars: int,
) -> None:
    clean_count = sum(item["automatic_artifact_clean"] for item in analyses)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Review pack: {source_manifest['name']}\n\n")
        handle.write(f"- Source ID: `{source_id}`\n")
        handle.write(f"- License label: `{source_manifest['license']}`\n")
        handle.write(f"- Redistribution tier: `{source_manifest['redistribution']}`\n")
        handle.write(f"- Documents: {len(documents)}\n")
        handle.write(
            f"- Automatically artifact-clean: {clean_count}/{len(documents)} "
            "(screening aid only)\n\n"
        )
        handle.write(
            "Scores are human judgments: 0 = fail, 1 = mixed, 2 = strong. "
            "Automated flags do not accept or reject a source.\n\n"
        )
        for index, (document, analysis) in enumerate(
            zip(documents, analyses, strict=True),
            start=1,
        ):
            handle.write(f"## Document {index:02d}\n\n")
            handle.write(f"- ID: `{document['document_id']}`\n")
            handle.write(f"- URL: {document.get('url') or '(none)'}\n")
            handle.write(
                f"- Size: {analysis['words']:,} words; "
                f"{analysis['paragraphs']:,} paragraphs; "
                f"{analysis['average_sentence_words']} average words/sentence\n"
            )
            flags = ", ".join(analysis["flags"]) if analysis["flags"] else "none"
            handle.write(f"- Automated flags: `{flags}`\n")
            if analysis["boilerplate_hits"]:
                handle.write(
                    "- Boilerplate matches: "
                    + ", ".join(analysis["boilerplate_hits"])
                    + "\n"
                )
            handle.write("\n### Beginning\n\n")
            handle.write(_blockquote(_excerpt(document["text"], "beginning", excerpt_chars)))
            handle.write("\n\n### Middle\n\n")
            handle.write(_blockquote(_excerpt(document["text"], "middle", excerpt_chars)))
            handle.write("\n\n### Ending\n\n")
            handle.write(_blockquote(_excerpt(document["text"], "ending", excerpt_chars)))
            handle.write("\n\n### Human score\n\n")
            handle.write(
                "- Clarity (0–2):\n"
                "- Structure (0–2):\n"
                "- Factual discipline (0–2):\n"
                "- Artifact cleanliness (0–2):\n"
                "- Voice distinctness (0–2):\n"
                "- Synthetic suspicion (0–2; 0 = none):\n"
                "- Keep (yes/no):\n"
                "- Notes:\n\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--samples-root", type=Path, default=DEFAULT_SAMPLES_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--excerpt-chars", type=int, default=900)
    args = parser.parse_args()

    manifest_path = args.manifest or _latest_manifest(args.samples_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = f"writing-{manifest['run_id']}"
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    packs_dir = output_dir / "packs"
    raw_dir = output_dir / "raw"
    packs_dir.mkdir()
    raw_dir.mkdir()

    triage: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "acquisition_manifest": str(manifest_path),
        "acquisition_manifest_sha256": _sha256(manifest_path.read_bytes()),
        "sources": {},
    }
    score_rows: list[dict[str, Any]] = []
    combined_raw_path = raw_dir / "all_sources.txt"

    with combined_raw_path.open("w", encoding="utf-8") as combined:
        combined.write("HIGH-QUALITY WRITING: ACQUIRED RAW REVIEW TEXT\n")
        combined.write(f"ACQUISITION RUN: {manifest['run_id']}\n")
        combined.write(
            "NOTE: These are capped acquisition samples, not complete upstream corpora.\n"
        )
        combined.write("#" * 100 + "\n\n")

        for source_id, source_manifest in manifest["sources"].items():
            if source_manifest["status"] != "complete":
                continue
            documents_path = Path(source_manifest["documents_path"])
            if _sha256(documents_path.read_bytes()) != source_manifest["documents_sha256"]:
                raise ValueError(f"Hash mismatch for {documents_path}")
            documents = _load_jsonl(documents_path)
            analyses = []
            earlier_texts: list[str] = []
            for document in documents:
                analyses.append(
                    _analyze_document(document, manifest["max_chars"], earlier_texts)
                )
                earlier_texts.append(document["text"])
            clean_count = sum(item["automatic_artifact_clean"] for item in analyses)
            flag_counts = Counter(flag for item in analyses for flag in item["flags"])
            triage["sources"][source_id] = {
                "name": source_manifest["name"],
                "document_count": len(documents),
                "character_count": sum(item["chars"] for item in analyses),
                "word_count": sum(item["words"] for item in analyses),
                "automatic_artifact_clean_count": clean_count,
                "automatic_artifact_clean_fraction": round(
                    clean_count / max(1, len(documents)),
                    4,
                ),
                "flag_counts": dict(sorted(flag_counts.items())),
                "documents": analyses,
            }

            raw_path = raw_dir / f"{source_id}.txt"
            _write_raw_source(raw_path, source_id, source_manifest, documents)
            combined.write(raw_path.read_text(encoding="utf-8"))
            combined.write("\n\n" + "#" * 100 + "\n\n")
            _write_source_pack(
                packs_dir / f"{source_id}.md",
                source_id,
                source_manifest,
                documents,
                analyses,
                args.excerpt_chars,
            )

            for index, analysis in enumerate(analyses, start=1):
                score_rows.append(
                    {
                        "source_id": source_id,
                        "document_index": index,
                        "document_id": analysis["document_id"],
                        "url": analysis["url"] or "",
                        "auto_flags": "|".join(analysis["flags"]),
                        "clarity_0_2": "",
                        "structure_0_2": "",
                        "factual_discipline_0_2": "",
                        "artifacts_0_2": "",
                        "voice_distinctness_0_2": "",
                        "synthetic_suspicion_0_2": "",
                        "keep_yes_no": "",
                        "reviewer_notes": "",
                    }
                )

    triage_path = output_dir / "triage.json"
    triage_path.write_text(
        json.dumps(triage, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    score_path = output_dir / "scores.csv"
    with score_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)

    report_path = output_dir / "REPORT.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# Automated writing-sample triage\n\n")
        handle.write(
            "This is a screening report, not a quality judgment. Automated checks "
            "surface likely extraction problems; they cannot establish factual "
            "accuracy, synthetic origin, licensing, or good writing.\n\n"
        )
        handle.write(f"- Acquisition manifest: `{manifest_path}`\n")
        handle.write(f"- Sources: {len(triage['sources'])}\n")
        handle.write(f"- Documents: {len(score_rows)}\n")
        handle.write(f"- Combined raw text: [`raw/all_sources.txt`](raw/all_sources.txt)\n")
        handle.write(f"- Human score sheet: [`scores.csv`](scores.csv)\n\n")
        handle.write(
            "| Source | Docs | Words | Auto-clean | Flags | Review pack | Raw text |\n"
        )
        handle.write("|---|---:|---:|---:|---|---|---|\n")
        for source_id, summary in triage["sources"].items():
            flags = ", ".join(
                f"{name}={count}" for name, count in summary["flag_counts"].items()
            )
            handle.write(
                f"| {summary['name']} | {summary['document_count']} | "
                f"{summary['word_count']:,} | "
                f"{summary['automatic_artifact_clean_count']}/"
                f"{summary['document_count']} | {flags or 'none'} | "
                f"[pack](packs/{source_id}.md) | [raw](raw/{source_id}.txt) |\n"
            )
        handle.write("\n## Recommended review order\n\n")
        ordered = sorted(
            triage["sources"].items(),
            key=lambda item: (
                item[1]["automatic_artifact_clean_fraction"],
                item[1]["word_count"],
            ),
        )
        for index, (source_id, summary) in enumerate(ordered, start=1):
            handle.write(
                f"{index}. [{summary['name']}](packs/{source_id}.md) — "
                f"{summary['automatic_artifact_clean_count']}/"
                f"{summary['document_count']} automatically artifact-clean\n"
            )
        handle.write(
            "\nStart with flagged sources: they produce the fastest accept, repair, "
            "or reject decisions. Then read the clean sources for actual prose quality.\n"
        )

    output_manifest = {
        "schema_version": 1,
        "created_utc": triage["created_utc"],
        "acquisition_manifest": str(manifest_path),
        "files": {},
    }
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            output_manifest["files"][str(path.relative_to(output_dir))] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path.read_bytes()),
            }
    review_manifest_path = output_dir / "manifest.json"
    review_manifest_path.write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"[report] {report_path}")
    print(f"[raw] {combined_raw_path}")
    print(f"[scores] {score_path}")
    print(
        f"[summary] {len(triage['sources'])} sources, {len(score_rows)} documents, "
        f"{sum(item['word_count'] for item in triage['sources'].values()):,} words"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Acquire capped, source-separated samples for manual writing-quality review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import tarfile
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml
from datasets import load_dataset
from huggingface_hub import HfApi


USER_AGENT = "ark296-gpt-data-review/0.1 (+https://github.com/arksomething/gpt)"
DEFAULT_REGISTRY = Path("data/sources.yaml")
DEFAULT_OUTPUT_ROOT = Path("data/source_samples")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _request(url: str, timeout: int = 60) -> tuple[bytes, dict[str, str], str]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json, application/xml, text/xml, text/html, */*",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return (
            response.read(),
            {key.lower(): value for key, value in response.headers.items()},
            response.geturl(),
        )


def _normalize_text(text: str, max_chars: int) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip()
    return text


class _ReadableHTMLParser(HTMLParser):
    BLOCKED = {"script", "style", "nav", "footer", "header", "noscript", "svg"}
    BREAKS = {
        "address",
        "article",
        "blockquote",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "main",
        "p",
        "pre",
        "section",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._blocked_depth = 0
        self._preferred_depth = 0
        self._body: list[str] = []
        self._preferred: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.BLOCKED:
            self._blocked_depth += 1
        if tag in {"article", "main"}:
            self._preferred_depth += 1
        if tag in self.BREAKS:
            self._append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.BREAKS:
            self._append("\n")
        if tag in {"article", "main"} and self._preferred_depth:
            self._preferred_depth -= 1
        if tag in self.BLOCKED and self._blocked_depth:
            self._blocked_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._blocked_depth:
            self._append(data)

    def _append(self, text: str) -> None:
        if self._blocked_depth:
            return
        self._body.append(text)
        if self._preferred_depth:
            self._preferred.append(text)

    def text(self) -> str:
        preferred = "".join(self._preferred)
        return preferred if len(preferred) >= 500 else "".join(self._body)


def _html_to_text(data: bytes) -> str:
    parser = _ReadableHTMLParser()
    parser.feed(data.decode("utf-8", errors="replace"))
    return parser.text()


def _extract_field(example: dict[str, Any], fields: list[str]) -> str | None:
    for field in fields:
        value = example.get(field)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list):
            joined = "\n\n".join(str(item) for item in value if item)
            if joined.strip():
                return joined
    return None


def _document(
    source_id: str,
    text: str,
    url: str | None,
    license_name: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "document_id": _sha256_text(text),
        "url": url,
        "license": license_name,
        "retrieved_utc": _utc_now(),
        "text": text,
        "metadata": metadata or {},
    }


def _sample_huggingface(
    source_id: str,
    source: dict[str, Any],
    count: int,
    max_chars: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    acquisition = source["acquisition"]
    dataset_id = acquisition["dataset_id"]
    config = acquisition.get("config")
    split = acquisition.get("split", "train")
    fields = acquisition.get("text_fields", ["text"])
    info = HfApi().dataset_info(dataset_id)
    dataset = load_dataset(dataset_id, config, split=split, streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=1_000)

    documents: list[dict[str, Any]] = []
    for example in dataset:
        raw_text = _extract_field(example, fields)
        if not raw_text:
            continue
        text = _normalize_text(raw_text, max_chars)
        if len(text) < 500:
            continue
        metadata = {
            key: value
            for key, value in example.items()
            if key not in fields and isinstance(value, (str, int, float, bool, type(None)))
        }
        url = metadata.get("url") if isinstance(metadata.get("url"), str) else None
        documents.append(_document(source_id, text, url, source["license"], metadata))
        if len(documents) >= count:
            break
    return documents, {"dataset_id": dataset_id, "revision": info.sha, "split": split}


def _sample_sitemap(
    source_id: str,
    source: dict[str, Any],
    count: int,
    max_chars: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    acquisition = source["acquisition"]
    sitemap_data, _, sitemap_final_url = _request(acquisition["sitemap_url"])
    root = ET.fromstring(sitemap_data)
    urls = [
        element.text.strip()
        for element in root.iter()
        if element.tag.endswith("loc") and element.text
    ]
    include_prefix = acquisition.get("include_prefix", "")
    excluded = acquisition.get("exclude_substrings", [])
    urls = [
        url
        for url in urls
        if url.startswith(include_prefix)
        and not any(fragment in url for fragment in excluded)
        and not re.search(r"\.(?:pdf|xml|json|csv|zip|png|jpg|jpeg|gif|svg)$", url, re.I)
    ]
    random.Random(seed).shuffle(urls)

    documents: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for url in urls:
        if len(documents) >= count:
            break
        try:
            body, _, final_url = _request(url)
            text = _normalize_text(_html_to_text(body), max_chars)
            if len(text) < 1_000:
                continue
            documents.append(_document(source_id, text, final_url, source["license"]))
        except Exception as exc:  # keep an auditable sample failure list
            errors.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
            if len(errors) >= 20:
                break
    return documents, {
        "sitemap_url": sitemap_final_url,
        "sitemap_sha256": _sha256_bytes(sitemap_data),
        "candidate_urls": len(urls),
        "errors": errors,
    }


def _sample_crs(
    source_id: str,
    source: dict[str, Any],
    count: int,
    max_chars: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    acquisition = source["acquisition"]
    index_data, _, final_url = _request(acquisition["index_url"])
    rows = list(csv.DictReader(io.StringIO(index_data.decode("utf-8"))))
    random.Random(seed).shuffle(rows)
    base_url = acquisition["base_url"]

    documents: list[dict[str, Any]] = []
    for row in rows:
        html_path = row.get("latestHTML")
        if not html_path:
            continue
        url = urllib.parse.urljoin(base_url, html_path)
        try:
            body, _, final_html_url = _request(url)
            text = _normalize_text(_html_to_text(body), max_chars)
        except Exception:
            continue
        if len(text) < 1_000:
            continue
        metadata = {
            "report_number": row.get("number"),
            "title": row.get("title"),
            "publication_date": row.get("latestPubDate"),
        }
        documents.append(
            _document(source_id, text, final_html_url, source["license"], metadata)
        )
        if len(documents) >= count:
            break
    return documents, {
        "index_url": final_url,
        "index_sha256": _sha256_bytes(index_data),
        "index_rows": len(rows),
    }


def _sample_plos(
    source_id: str,
    source: dict[str, Any],
    count: int,
    max_chars: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    api_url = source["acquisition"]["api_url"]
    probe_query = urllib.parse.urlencode(
        {"q": "*:*", "fq": "doc_type:full", "fl": "id", "rows": 0, "wt": "json"}
    )
    probe_data, _, _ = _request(f"{api_url}?{probe_query}")
    total = int(json.loads(probe_data)["response"]["numFound"])
    start = random.Random(seed).randrange(max(1, total - count * 4))
    query = urllib.parse.urlencode(
        {
            "q": "*:*",
            "fq": "doc_type:full",
            "fl": "id,title_display,abstract,everything,publication_date",
            "rows": count * 4,
            "start": start,
            "wt": "json",
        }
    )
    data, _, final_url = _request(f"{api_url}?{query}")
    rows = json.loads(data)["response"]["docs"]

    documents: list[dict[str, Any]] = []
    for row in rows:
        raw_text = row.get("everything")
        if not isinstance(raw_text, str):
            continue
        text = _normalize_text(raw_text, max_chars)
        if len(text) < 1_000:
            continue
        doi = row.get("id")
        url = f"https://doi.org/{doi}" if doi else None
        metadata = {
            "doi": doi,
            "title": row.get("title_display"),
            "publication_date": row.get("publication_date"),
        }
        documents.append(_document(source_id, text, url, source["license"], metadata))
        if len(documents) >= count:
            break
    return documents, {"api_url": final_url, "total_documents": total, "start": start}


def _sample_standard_ebooks(
    source_id: str,
    source: dict[str, Any],
    count: int,
    max_chars: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    acquisition = source["acquisition"]
    query = urllib.parse.urlencode({"per_page": 100, "type": "public", "sort": "full_name"})
    data, _, final_url = _request(f"{acquisition['api_url']}?{query}")
    repos = json.loads(data)
    repos = [
        repo
        for repo in repos
        if isinstance(repo, dict)
        and repo.get("name")
        and repo.get("default_branch")
        and not repo.get("archived")
    ]
    random.Random(seed).shuffle(repos)

    documents: list[dict[str, Any]] = []
    used_repos: list[dict[str, str]] = []
    for repo in repos:
        if len(documents) >= count:
            break
        full_name = repo["full_name"]
        revision = repo.get("default_branch", "master")
        archive_url = f"https://api.github.com/repos/{full_name}/tarball/{revision}"
        try:
            archive, _, _ = _request(archive_url, timeout=120)
            chunks: list[str] = []
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
                members = sorted(
                    (
                        member
                        for member in tar.getmembers()
                        if "/src/epub/text/" in member.name
                        and member.name.endswith((".xhtml", ".html"))
                        and not any(
                            excluded in member.name
                            for excluded in ("colophon", "imprint", "titlepage", "uncopyright")
                        )
                    ),
                    key=lambda member: member.name,
                )
                for member in members:
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    chunks.append(_html_to_text(extracted.read()))
                    if sum(len(chunk) for chunk in chunks) >= max_chars * 2:
                        break
            text = _normalize_text("\n\n".join(chunks), max_chars)
        except Exception:
            continue
        if len(text) < 1_000:
            continue
        metadata = {
            "repository": full_name,
            "repository_updated_at": repo.get("updated_at"),
            "default_branch": revision,
        }
        documents.append(
            _document(source_id, text, repo.get("html_url"), source["license"], metadata)
        )
        used_repos.append({"repository": full_name, "archive_sha256": _sha256_bytes(archive)})
    return documents, {"api_url": final_url, "repositories": used_repos}


def _capture_policy(source_dir: Path, source: dict[str, Any]) -> list[dict[str, Any]]:
    acquisition = source.get("acquisition", {})
    urls = []
    for key in ("robots_url",):
        if acquisition.get(key):
            urls.append((key, acquisition[key]))
    if source.get("license_url"):
        urls.append(("license", source["license_url"]))

    snapshots: list[dict[str, Any]] = []
    policy_dir = source_dir / "policy"
    for label, url in urls:
        try:
            data, headers, final_url = _request(url)
        except Exception as exc:
            snapshots.append(
                {"label": label, "url": url, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        policy_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".txt"
        content_type = headers.get("content-type", "")
        if "html" in content_type:
            suffix = ".html"
        elif "xml" in content_type:
            suffix = ".xml"
        path = policy_dir / f"{label}{suffix}"
        path.write_bytes(data)
        snapshots.append(
            {
                "label": label,
                "url": final_url,
                "path": str(path),
                "sha256": _sha256_bytes(data),
                "content_type": content_type,
            }
        )
    return snapshots


SAMPLERS = {
    "huggingface": _sample_huggingface,
    "sitemap_html": _sample_sitemap,
    "crs_csv": _sample_crs,
    "plos_api": _sample_plos,
    "github_org_epubs": _sample_standard_ebooks,
}


def _write_jsonl(path: Path, documents: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sources", help="Comma-separated source IDs (default: enabled)")
    parser.add_argument("--docs-per-source", type=int, default=12)
    parser.add_argument("--max-chars", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=20_260_727)
    parser.add_argument("--list", action="store_true", help="List sources and exit")
    args = parser.parse_args()

    if args.docs_per_source <= 0:
        raise SystemExit("--docs-per-source must be positive")
    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
    sources: dict[str, dict[str, Any]] = registry["sources"]

    if args.list:
        for source_id, source in sources.items():
            enabled = bool(source.get("acquisition", {}).get("enabled"))
            print(
                f"{source_id:32} tier={source['tier']} "
                f"status={source['status']:11} enabled={str(enabled).lower()}"
            )
        return

    if args.sources:
        selected_ids = [item.strip() for item in args.sources.split(",") if item.strip()]
        unknown = sorted(set(selected_ids) - set(sources))
        if unknown:
            raise SystemExit(f"Unknown source IDs: {', '.join(unknown)}")
    else:
        selected_ids = [
            source_id
            for source_id, source in sources.items()
            if source.get("acquisition", {}).get("enabled")
        ]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": _utc_now(),
        "registry_path": str(args.registry),
        "registry_sha256": _sha256_bytes(args.registry.read_bytes()),
        "seed": args.seed,
        "docs_per_source": args.docs_per_source,
        "max_chars": args.max_chars,
        "sources": {},
    }

    failures = 0
    for index, source_id in enumerate(selected_ids):
        source = sources[source_id]
        acquisition = source.get("acquisition", {})
        kind = acquisition.get("kind")
        if kind not in SAMPLERS:
            print(f"[skip] {source_id}: unsupported or disabled acquisition kind {kind!r}")
            continue

        source_dir = run_dir / source_id
        source_dir.mkdir(parents=True)
        print(f"[acquire] {source_id} ({kind})", flush=True)
        try:
            documents, acquisition_meta = SAMPLERS[kind](
                source_id,
                source,
                args.docs_per_source,
                args.max_chars,
                args.seed + index,
            )
            if len(documents) < args.docs_per_source:
                raise RuntimeError(
                    f"collected {len(documents)}/{args.docs_per_source} documents"
                )
            documents_path = source_dir / "documents.jsonl"
            _write_jsonl(documents_path, documents)
            policy_snapshots = _capture_policy(source_dir, source)
            entry = {
                "name": source["name"],
                "tier": source["tier"],
                "redistribution": source["redistribution"],
                "license": source["license"],
                "acquisition_kind": kind,
                "status": "complete",
                "document_count": len(documents),
                "character_count": sum(len(doc["text"]) for doc in documents),
                "documents_path": str(documents_path),
                "documents_sha256": _sha256_bytes(documents_path.read_bytes()),
                "policy_snapshots": policy_snapshots,
                "acquisition": acquisition_meta,
            }
            print(
                f"[complete] {source_id}: {len(documents)} docs, "
                f"{entry['character_count']:,} chars",
                flush=True,
            )
        except Exception as exc:
            failures += 1
            entry = {
                "name": source["name"],
                "tier": source["tier"],
                "redistribution": source["redistribution"],
                "license": source["license"],
                "acquisition_kind": kind,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[failed] {source_id}: {entry['error']}", flush=True)
        manifest["sources"][source_id] = entry
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    complete = sum(
        entry["status"] == "complete" for entry in manifest["sources"].values()
    )
    print(f"[manifest] {run_dir / 'manifest.json'}")
    print(f"[summary] {complete} complete, {failures} failed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate provenance-rich synthetic conversations through Fireworks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from scripts.chat_format import validate_messages
from scripts.indexed_shards import sha256_file


SCHEMA_VERSION = 1
FIREWORKS_ENDPOINT = "https://api.fireworks.ai/inference/v1/chat/completions"
DEFAULT_SEEDS = Path("recipes/chat/seeds_v1.jsonl")
DEFAULT_EVAL = Path("evals/conversation/v1.jsonl")
DEFAULT_MODEL = "accounts/fireworks/models/gpt-oss-120b"
DEFAULT_LICENSE_EVIDENCE = (
    "https://huggingface.co/openai/gpt-oss-120b/blob/main/LICENSE"
)

SYSTEM_PROMPT = """You create high-quality training conversations for a small
general conversational assistant. Return only one JSON object with a `messages`
array. Messages must alternate user and assistant, begin with user, and end
with assistant. Write a realistic conversation, not an evaluation item.

Requirements:
- natural human wording and varied sentence rhythms;
- no canned "As an AI" language, excessive headings, or constant praise;
- assistant responses should be correct, direct, context-aware, and no longer
  than useful;
- the user should react naturally to prior assistant messages;
- do not expose hidden reasoning or chain-of-thought;
- do not mention this generation request, datasets, rubrics, or benchmarks;
- do not copy famous benchmark questions.
"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            records.append(value)
    return records


def _normalized_words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9']+", text.casefold()))


def _eval_ngrams(eval_path: Path, width: int = 12) -> set[tuple[str, ...]]:
    ngrams = set()
    for case in _read_jsonl(eval_path):
        texts = [str(case.get("system", ""))]
        texts.extend(str(turn.get("user", "")) for turn in case.get("turns", []))
        for text in texts:
            words = _normalized_words(text)
            ngrams.update(
                tuple(words[index : index + width])
                for index in range(max(0, len(words) - width + 1))
            )
    return ngrams


def _overlaps_eval(messages: list[dict[str, str]], eval_ngrams, width: int = 12) -> bool:
    for message in messages:
        words = _normalized_words(message["content"])
        for index in range(max(0, len(words) - width + 1)):
            if tuple(words[index : index + width]) in eval_ngrams:
                return True
    return False


def _extract_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        stop = candidate.rfind("}")
        if start < 0 or stop <= start:
            raise
        value = json.loads(candidate[start : stop + 1])
    if not isinstance(value, dict):
        raise ValueError("generator response must be a JSON object")
    return value


def _completion(api_key: str, model: str, prompt: str, temperature: float) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 1800,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        FIREWORKS_ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            value = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Fireworks HTTP {exc.code}: {detail}") from exc
    return value["choices"][0]["message"]["content"]


def _prompt_for_seed(seed: dict[str, Any], variation: int) -> str:
    return (
        f"Scenario category: {seed['category']}\n"
        f"Scenario brief: {seed['brief']}\n"
        f"Use between {int(seed['turns_min'])} and {int(seed['turns_max'])} "
        "user-assistant exchanges.\n"
        f"Variation number: {variation}. Choose fresh, specific subject matter."
    )


def generate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and not args.resume:
        raise FileExistsError(
            f"{args.output} exists; use --resume or choose a new immutable output"
        )
    if not args.confirm_spend and not args.dry_run:
        raise ValueError("add --confirm_spend to authorize Fireworks API charges")
    api_key = os.environ.get(args.api_key_env)
    if not api_key and not args.dry_run:
        raise ValueError(f"missing environment variable {args.api_key_env}")

    seeds = _read_jsonl(args.seeds)
    if not seeds:
        raise ValueError("seed recipe is empty")
    frozen_ngrams = _eval_ngrams(args.eval)
    existing = _read_jsonl(args.output) if args.output.exists() else []
    existing_ids = {record.get("id") for record in existing}
    output_count = len(existing)
    rejected = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prompt_template_sha256 = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    rng = random.Random(args.seed)
    work = [
        (seed, variation)
        for seed in seeds
        for variation in range(args.count_per_seed)
    ]
    rng.shuffle(work)
    if args.max_records is not None:
        work = work[: args.max_records]

    if args.dry_run:
        for seed, variation in work[: min(3, len(work))]:
            print(_prompt_for_seed(seed, variation))
            print()
        return {"planned_requests": len(work), "spent": False}

    with args.output.open("a", encoding="utf-8", newline="\n") as handle:
        for seed, variation in work:
            record_id = f"{seed['id']}-{variation:06d}"
            if record_id in existing_ids:
                continue
            prompt = _prompt_for_seed(seed, variation)
            raw = _completion(api_key, args.model, prompt, args.temperature)
            try:
                generated = _extract_json(raw)
                messages = generated.get("messages")
                validate_messages(messages, require_final_assistant=True)
                if any(message["role"] == "system" for message in messages):
                    raise ValueError("generated conversations must not include system turns")
                exchanges = sum(
                    1 for message in messages if message["role"] == "user"
                )
                if not int(seed["turns_min"]) <= exchanges <= int(seed["turns_max"]):
                    raise ValueError(
                        f"generated {exchanges} exchanges outside requested range"
                    )
                if _overlaps_eval(messages, frozen_ngrams):
                    raise ValueError("generated conversation overlaps frozen evaluation")
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                rejected += 1
                print(f"reject {record_id}: {exc}")
                continue
            record = {
                "id": record_id,
                "source_id": args.source_id,
                "source_name": args.source_name,
                "license": args.license,
                "license_evidence": args.license_evidence,
                "synthetic": True,
                "generator": {
                    "provider": "fireworks",
                    "model": args.model,
                    "prompt_template_sha256": prompt_template_sha256,
                    "seed_recipe_id": seed["id"],
                    "variation": variation,
                    "temperature": args.temperature,
                    "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                },
                "messages": messages,
            }
            handle.write(
                json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
            existing_ids.add(record_id)
            output_count += 1
            print(f"accepted {record_id}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "synthetic-chat-generation",
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "record_count": output_count,
        "rejected_this_session": rejected,
        "model": args.model,
        "source_id": args.source_id,
        "license": args.license,
        "license_evidence": args.license_evidence,
        "seed_recipe": {
            "path": str(args.seeds.resolve()),
            "sha256": sha256_file(args.seeds),
        },
        "frozen_eval_decontamination": {
            "path": str(args.eval.resolve()),
            "sha256": sha256_file(args.eval),
            "ngram_width": 12,
        },
        "prompt_template_sha256": prompt_template_sha256,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api_key_env", default="FIREWORKS_API_KEY")
    parser.add_argument("--count_per_seed", type=int, default=10)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--source_id", default="fireworks_gpt_oss_120b_v1")
    parser.add_argument(
        "--source_name",
        default="Fireworks gpt-oss-120b synthetic conversations v1",
    )
    parser.add_argument("--license", default="Apache-2.0")
    parser.add_argument("--license_evidence", default=DEFAULT_LICENSE_EVIDENCE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--confirm_spend", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        manifest = generate(args)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

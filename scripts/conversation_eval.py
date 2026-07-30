#!/usr/bin/env python3
"""Generate, score, judge, and manually review frozen conversation evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import sentencepiece as spm
import torch

from scripts.chat_format import END_MARKER, encode_generation_prompt
from scripts.indexed_shards import sha256_file
from scripts.infer import (
    _load_artifact_manifest,
    _validate_artifacts_or_die,
    load_model,
    validate_tokenizer_model_config,
)


SCHEMA_VERSION = 1
DEFAULT_EVAL = Path("evals/conversation/v1.jsonl")
FIREWORKS_ENDPOINT = "https://api.fireworks.ai/inference/v1/chat/completions"


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            values.append(value)
    return values


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(
                json.dumps(
                    value,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def _validate_eval_cases(cases: list[dict[str, Any]]) -> None:
    ids = set()
    for case in cases:
        case_id = case.get("id")
        turns = case.get("turns")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError(f"invalid or duplicate eval id: {case_id!r}")
        ids.add(case_id)
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"{case_id}: turns must be a non-empty list")
        for index, turn in enumerate(turns):
            if not isinstance(turn.get("user"), str) or not turn["user"].strip():
                raise ValueError(f"{case_id} turn {index}: missing user text")
            if not isinstance(turn.get("rubric"), str) or not turn["rubric"].strip():
                raise ValueError(f"{case_id} turn {index}: missing rubric")
            if not isinstance(turn.get("checks", {}), dict):
                raise ValueError(f"{case_id} turn {index}: checks must be an object")


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _sentence_count(text: str) -> int:
    return len(re.findall(r"[.!?]+(?:\s|$)", text.strip())) or int(bool(text.strip()))


def score_checks(text: str, checks: dict[str, Any]) -> dict[str, Any]:
    lowered = text.casefold()
    results: dict[str, bool] = {}
    if "must_include_all" in checks:
        results["must_include_all"] = all(
            str(value).casefold() in lowered for value in checks["must_include_all"]
        )
    if "must_include_any" in checks:
        results["must_include_any"] = any(
            str(value).casefold() in lowered for value in checks["must_include_any"]
        )
    if "must_not_include" in checks:
        results["must_not_include"] = all(
            str(value).casefold() not in lowered for value in checks["must_not_include"]
        )
    words = _word_count(text)
    if "max_words" in checks:
        results["max_words"] = words <= int(checks["max_words"])
    if "min_words" in checks:
        results["min_words"] = words >= int(checks["min_words"])
    if "exact_words" in checks:
        results["exact_words"] = words == int(checks["exact_words"])
    if "max_sentences" in checks:
        results["max_sentences"] = _sentence_count(text) <= int(
            checks["max_sentences"]
        )
    if checks.get("requires_question"):
        results["requires_question"] = "?" in text
    if "regex" in checks:
        results["regex"] = re.search(str(checks["regex"]), text) is not None
    if "json_type" in checks:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        expected = checks["json_type"]
        results["json_type"] = (
            (expected == "list" and isinstance(parsed, list))
            or (expected == "object" and isinstance(parsed, dict))
        )
        if "json_length" in checks:
            results["json_length"] = (
                isinstance(parsed, (list, dict))
                and len(parsed) == int(checks["json_length"])
            )
    return {
        "passed": all(results.values()) if results else True,
        "checks": results,
        "word_count": words,
        "sentence_count": _sentence_count(text),
    }


@torch.no_grad()
def _generate_turn(
    model,
    tokenizer,
    messages,
    *,
    device,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    repetition_penalty,
) -> str:
    prompt_ids = list(encode_generation_prompt(tokenizer, messages))
    if len(prompt_ids) >= model.config.max_position_embeddings:
        raise ValueError(
            f"conversation prompt has {len(prompt_ids)} tokens, exceeding context"
        )
    input_ids = torch.tensor([prompt_ids], device=device)
    do_sample = temperature > 0
    kwargs = {
        "max_new_tokens": min(
            max_new_tokens,
            model.config.max_position_embeddings - len(prompt_ids),
        ),
        "do_sample": do_sample,
        "pad_token_id": model.config.pad_token_id,
        "eos_token_id": model.config.eos_token_id,
        "repetition_penalty": repetition_penalty,
    }
    if do_sample:
        kwargs.update(
            {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
        )
    output = model.generate(input_ids=input_ids, **kwargs)[0].tolist()
    continuation = output[len(prompt_ids) :]
    eos_id = model.config.eos_token_id
    if eos_id in continuation:
        continuation = continuation[: continuation.index(eos_id)]
    text = tokenizer.decode(continuation).strip()
    return text.split(END_MARKER, 1)[0].strip()


def generate_local(args: argparse.Namespace) -> None:
    cases = _read_jsonl(args.eval)
    _validate_eval_cases(cases)
    manifest, manifest_path = _load_artifact_manifest(args.checkpoint, None)
    if manifest is None:
        raise ValueError("checkpoint is missing artifacts_manifest.json")
    _validate_artifacts_or_die(
        manifest,
        str(args.tokenizer),
        str(args.model_config),
    )
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(args.tokenizer))
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    model, _ = load_model(str(args.checkpoint), str(args.model_config), device)
    validate_tokenizer_model_config(tokenizer, model.config)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    output_records = []
    for case in cases[: args.limit or None]:
        messages = []
        if case.get("system"):
            messages.append({"role": "system", "content": case["system"]})
        turn_results = []
        for turn in case["turns"]:
            messages.append({"role": "user", "content": turn["user"]})
            response = _generate_turn(
                model,
                tokenizer,
                messages,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
            result = {
                "user": turn["user"],
                "assistant": response,
                "rubric": turn["rubric"],
                "deterministic": score_checks(response, turn.get("checks", {})),
            }
            turn_results.append(result)
            messages.append({"role": "assistant", "content": response})
        output_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_id": case["id"],
                "category": case["category"],
                "turns": turn_results,
            }
        )

    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "conversation-eval-run",
        "eval_path": str(args.eval.resolve()),
        "eval_sha256": sha256_file(args.eval),
        "checkpoint": str(args.checkpoint.resolve()),
        "model_sha256": sha256_file(args.checkpoint / "model.pt"),
        "artifacts_manifest_sha256": sha256_file(Path(manifest_path)),
        "model_config_sha256": sha256_file(args.model_config),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "generation": {
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
        },
        "case_count": len(output_records),
    }
    run_manifest["run_id"] = _canonical_hash(run_manifest)[:16]
    _write_jsonl(args.output, output_records)
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    passed = sum(
        turn["deterministic"]["passed"]
        for record in output_records
        for turn in record["turns"]
    )
    total = sum(len(record["turns"]) for record in output_records)
    print(f"wrote {args.output}; deterministic checks {passed}/{total} passed")


def score_file(args: argparse.Namespace) -> None:
    records = _read_jsonl(args.input)
    passed = 0
    total = 0
    by_category: dict[str, list[bool]] = {}
    for record in records:
        category = record.get("category", "unknown")
        for turn in record.get("turns", []):
            result = turn.get("deterministic") or {}
            value = bool(result.get("passed"))
            by_category.setdefault(category, []).append(value)
            total += 1
            passed += int(value)
    summary = {
        "input": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "passed": passed,
        "total": total,
        "pass_rate": passed / total if total else None,
        "categories": {
            category: {
                "passed": sum(values),
                "total": len(values),
                "pass_rate": sum(values) / len(values),
            }
            for category, values in sorted(by_category.items())
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _records_by_id(path: Path) -> dict[str, dict[str, Any]]:
    records = _read_jsonl(path)
    result = {str(record["case_id"]): record for record in records}
    if len(result) != len(records):
        raise ValueError(f"{path}: duplicate case_id")
    return result


def _format_transcript(turns: list[dict[str, Any]], stop: int) -> str:
    lines = []
    for turn in turns[:stop]:
        lines.extend(
            [
                f"User: {turn['user']}",
                f"Assistant: {turn['assistant']}",
            ]
        )
    return "\n\n".join(lines)


def build_review(args: argparse.Namespace) -> None:
    left = _records_by_id(args.left)
    right = _records_by_id(args.right)
    if set(left) != set(right):
        raise ValueError("left and right outputs contain different case IDs")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    markdown = [
        "# Blind conversational evaluation",
        "",
        "For each turn, mark `A`, `B`, `tie`, or `both_bad` in `reviews.csv`.",
        "",
    ]
    for case_id in sorted(left):
        left_turns = left[case_id]["turns"]
        right_turns = right[case_id]["turns"]
        if len(left_turns) != len(right_turns):
            raise ValueError(f"{case_id}: turn count mismatch")
        swap = int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 2 == 1
        markdown.extend([f"## {case_id}", ""])
        for index, (left_turn, right_turn) in enumerate(
            zip(left_turns, right_turns), 1
        ):
            a = right_turn if swap else left_turn
            b = left_turn if swap else right_turn
            a_turns = right_turns if swap else left_turns
            b_turns = left_turns if swap else right_turns
            markdown.extend(
                [
                    f"### Turn {index}",
                    "",
                    f"**Rubric:** {a['rubric']}",
                    "",
                    "**Conversation A:**",
                    "",
                    _format_transcript(a_turns, index),
                    "",
                    "**Conversation B:**",
                    "",
                    _format_transcript(b_turns, index),
                    "",
                ]
            )
            rows.append(
                {
                    "case_id": case_id,
                    "turn": index,
                    "winner": "",
                    "reason": "",
                    "_a_source": "right" if swap else "left",
                    "_b_source": "left" if swap else "right",
                }
            )
    (args.output_dir / "review.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "reviews.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "left": {"path": str(args.left.resolve()), "sha256": sha256_file(args.left)},
        "right": {
            "path": str(args.right.resolve()),
            "sha256": sha256_file(args.right),
        },
        "blinding": "sha256(case_id) parity",
        "comparison_count": len(rows),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote blind review pack to {args.output_dir}")


def _fireworks_completion(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 300,
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
        with urllib.request.urlopen(request, timeout=120) as response:
            value = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Fireworks HTTP {exc.code}: {detail}") from exc
    return value["choices"][0]["message"]["content"]


def _parse_judgment(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
    value = json.loads(candidate)
    if value.get("winner") not in {"A", "B", "tie", "both_bad"}:
        raise ValueError(f"invalid judge winner: {value.get('winner')!r}")
    return value


def judge_fireworks(args: argparse.Namespace) -> None:
    if not args.confirm_spend:
        raise ValueError("add --confirm_spend to authorize Fireworks API charges")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(f"missing environment variable {args.api_key_env}")
    left = _records_by_id(args.left)
    right = _records_by_id(args.right)
    if set(left) != set(right):
        raise ValueError("left and right outputs contain different case IDs")
    judgments = []
    comparisons = []
    for case_id in sorted(left):
        if len(left[case_id]["turns"]) != len(right[case_id]["turns"]):
            raise ValueError(f"{case_id}: turn count mismatch")
        for index, (left_turn, right_turn) in enumerate(
            zip(left[case_id]["turns"], right[case_id]["turns"]), 1
        ):
            comparisons.append(
                (
                    case_id,
                    index,
                    left_turn,
                    right_turn,
                    left[case_id]["turns"],
                    right[case_id]["turns"],
                )
            )
    if args.max_comparisons is not None:
        comparisons = comparisons[: args.max_comparisons]

    system = (
        "You are a strict pairwise evaluator of conversational assistants. "
        "Judge correctness, context use, instruction following, naturalness, "
        "honesty, and concision. Do not reward verbosity. Return only JSON with "
        'keys winner ("A", "B", "tie", or "both_bad") and reason.'
    )
    for (
        case_id,
        index,
        left_turn,
        right_turn,
        left_turns,
        right_turns,
    ) in comparisons:
        order_results = []
        for swapped in (False, True):
            a_turns = right_turns if swapped else left_turns
            b_turns = left_turns if swapped else right_turns
            prompt = (
                f"Instance rubric for turn {index}:\n{left_turn['rubric']}\n\n"
                "Judge the assistant response at the final shown turn while "
                "using all earlier turns as context.\n\n"
                f"Conversation A:\n{_format_transcript(a_turns, index)}\n\n"
                f"Conversation B:\n{_format_transcript(b_turns, index)}"
            )
            raw = _fireworks_completion(
                api_key,
                args.model,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            parsed = _parse_judgment(raw)
            winner = parsed["winner"]
            if swapped:
                winner = {"A": "B", "B": "A"}.get(winner, winner)
            order_results.append(
                {
                    "swapped": swapped,
                    "normalized_winner": winner,
                    "reason": parsed.get("reason"),
                    "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                }
            )
        final_winner = (
            order_results[0]["normalized_winner"]
            if order_results[0]["normalized_winner"]
            == order_results[1]["normalized_winner"]
            else "position_unstable"
        )
        judgments.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_id": case_id,
                "turn": index,
                "judge_model": args.model,
                "winner": final_winner,
                "orders": order_results,
            }
        )
        _write_jsonl(args.output, judgments)
        judge_manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "conversation-pairwise-judge-run",
            "left": {
                "path": str(args.left.resolve()),
                "sha256": sha256_file(args.left),
            },
            "right": {
                "path": str(args.right.resolve()),
                "sha256": sha256_file(args.right),
            },
            "judge_model": args.model,
            "position_swapped": True,
            "prompt_version": 1,
            "requested_comparisons": len(comparisons),
            "completed_comparisons": len(judgments),
            "output_sha256": sha256_file(args.output),
        }
        args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
            json.dumps(judge_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"{case_id} turn={index} winner={final_winner}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-local")
    generate.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--model_config", type=Path, required=True)
    generate.add_argument("--tokenizer", type=Path, default=Path("tokenizer/spm.model"))
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--device", default="auto")
    generate.add_argument("--seed", type=int, default=1337)
    generate.add_argument("--max_new_tokens", type=int, default=160)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--top_p", type=float, default=0.95)
    generate.add_argument("--top_k", type=int, default=50)
    generate.add_argument("--repetition_penalty", type=float, default=1.1)
    generate.add_argument("--limit", type=int, default=None)
    generate.set_defaults(function=generate_local)

    score = subparsers.add_parser("score")
    score.add_argument("--input", type=Path, required=True)
    score.set_defaults(function=score_file)

    review = subparsers.add_parser("build-review")
    review.add_argument("--left", type=Path, required=True)
    review.add_argument("--right", type=Path, required=True)
    review.add_argument("--output_dir", type=Path, required=True)
    review.set_defaults(function=build_review)

    judge = subparsers.add_parser("judge-fireworks")
    judge.add_argument("--left", type=Path, required=True)
    judge.add_argument("--right", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument(
        "--model",
        default="accounts/fireworks/models/qwen3-235b-a22b-thinking-2507",
    )
    judge.add_argument("--api_key_env", default="FIREWORKS_API_KEY")
    judge.add_argument("--max_comparisons", type=int, default=None)
    judge.add_argument("--confirm_spend", action="store_true")
    judge.set_defaults(function=judge_fireworks)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.function(args)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fail-closed validation for a paid training probe without training a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import sentencepiece as spm
import yaml

from scripts.benchmark_throughput import workload_fingerprint
from scripts.experiment_registry import _canonical_sha256
from scripts.indexed_shards import IndexedShardReader, ShardFormatError, sha256_file
from scripts.validate_model_ladder import validate_config


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _git_dirty(root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _check_experiment_plan(
    plan_path: Path | None,
    *,
    model_path: Path,
    train_path: Path,
    readers: dict[str, IndexedShardReader],
) -> tuple[list[Check], dict[str, Any] | None]:
    if plan_path is None:
        return [
            Check(
                "experiment_plan",
                "BLOCK",
                "no immutable experiment plan supplied; create a >=3-seed group",
            )
        ], None
    try:
        record = json.loads(plan_path.read_text(encoding="utf-8"))
        expected = record.get("plan_sha256")
        unsigned = dict(record)
        unsigned.pop("plan_sha256", None)
        if not expected or _canonical_sha256(unsigned) != expected:
            raise ValueError("plan self-hash mismatch")
        seeds = record.get("seeds") or []
        if len(set(seeds)) < 3:
            raise ValueError("fewer than three distinct seeds")
        if record.get("model", {}).get("config_sha256") != sha256_file(model_path):
            raise ValueError("model config hash differs from plan")
        if (
            record.get("training", {}).get("base_config_sha256")
            != sha256_file(train_path)
        ):
            raise ValueError("training config hash differs from plan")
        runs = record.get("runs") or []
        if len(runs) != len(seeds):
            raise ValueError("run count does not match seed count")
        for split in ("train", "validation"):
            reader = readers.get(split)
            planned = (record.get("data", {}).get(split) or {}).get(
                "corpus_sha256"
            )
            if reader is not None and planned != reader.manifest["corpus_sha256"]:
                raise ValueError(f"{split} corpus differs from experiment plan")
        return [
            Check(
                "experiment_plan",
                "PASS",
                f"{len(seeds)} seeds; group {record.get('group_id')}",
            )
        ], record
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        return [Check("experiment_plan", "BLOCK", str(exc))], None


def run_preflight(
    model_path: Path,
    train_path: Path,
    *,
    experiment_plan: Path | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    model_path = model_path.resolve()
    train_path = train_path.resolve()
    checks: list[Check] = []

    try:
        model_result = validate_config(model_path)
        if not model_result["within_tolerance"]:
            raise ValueError("parameter count outside ladder tolerance")
        checks.append(
            Check(
                "model",
                "PASS",
                f"{model_result['total_parameters']:,} parameters on meta device",
            )
        )
    except Exception as exc:
        model_result = None
        checks.append(Check("model", "BLOCK", str(exc)))

    try:
        train_cfg = _load_yaml(train_path)
        data_cfg = train_cfg["data"]
        training_cfg = train_cfg["training"]
        indexed_cfg = data_cfg["indexed"]
        if not indexed_cfg.get("enabled"):
            raise ValueError("paid probes must use indexed data")
        checks.append(Check("training_config", "PASS", "indexed probe recipe loaded"))
    except Exception as exc:
        train_cfg = {}
        data_cfg = {}
        training_cfg = {}
        indexed_cfg = {}
        checks.append(Check("training_config", "BLOCK", str(exc)))

    tokenizer_path: Path | None = None
    tokenizer_sha: str | None = None
    try:
        tokenizer_value = (train_cfg.get("data_prep") or {}).get("tokenizer_model")
        if not tokenizer_value:
            raise ValueError("data_prep.tokenizer_model is required")
        tokenizer_path = _resolve(root, tokenizer_value).resolve()
        tokenizer_sha = sha256_file(tokenizer_path)
        processor = spm.SentencePieceProcessor()
        if not processor.load(str(tokenizer_path)):
            raise ValueError("SentencePiece rejected tokenizer")
        model_vocab = (
            _load_yaml(model_path).get("model", {}).get("vocab_size")
        )
        if processor.vocab_size() != model_vocab:
            raise ValueError(
                f"tokenizer vocab {processor.vocab_size()} != model vocab {model_vocab}"
            )
        checks.append(
            Check("tokenizer", "PASS", f"{tokenizer_path} ({tokenizer_sha[:12]})")
        )
    except Exception as exc:
        checks.append(Check("tokenizer", "BLOCK", str(exc)))

    readers: dict[str, IndexedShardReader] = {}
    for split, key in (("train", "train_dir"), ("validation", "val_dir")):
        try:
            value = indexed_cfg.get(key)
            if not value:
                raise ValueError(f"data.indexed.{key} is required")
            corpus_path = _resolve(root, value).resolve()
            reader = IndexedShardReader(
                corpus_path,
                expected_tokenizer_sha256=tokenizer_sha,
                expected_recipe_sha256=indexed_cfg.get("recipe_sha256"),
                verify_hashes=True,
            )
            explicit_tokenizer = indexed_cfg.get("tokenizer_sha256")
            if (
                explicit_tokenizer
                and reader.manifest["tokenizer_sha256"] != explicit_tokenizer
            ):
                raise ShardFormatError("explicit tokenizer fingerprint mismatch")
            readers[split] = reader
            checks.append(
                Check(
                    f"{split}_corpus",
                    "PASS",
                    f"{reader.manifest['document_count']:,} docs, "
                    f"{reader.manifest['token_count']:,} tokens, all hashes verified",
                )
            )
        except Exception as exc:
            checks.append(Check(f"{split}_corpus", "BLOCK", str(exc)))

    if len(readers) == 2:
        train_manifest = readers["train"].manifest
        val_manifest = readers["validation"].manifest
        if train_manifest["tokenizer_sha256"] != val_manifest["tokenizer_sha256"]:
            checks.append(
                Check("split_fingerprints", "BLOCK", "tokenizer hashes differ")
            )
        elif train_manifest["recipe_sha256"] != val_manifest["recipe_sha256"]:
            checks.append(Check("split_fingerprints", "BLOCK", "recipe hashes differ"))
        else:
            checks.append(
                Check(
                    "split_fingerprints",
                    "PASS",
                    "train and validation tokenizer/recipe hashes agree",
                )
            )

        val_hashes = {doc.content_sha256 for doc in readers["validation"].documents}
        overlap = [
            doc.content_sha256
            for doc in readers["train"].documents
            if doc.content_sha256 in val_hashes
        ]
        checks.append(
            Check(
                "split_overlap",
                "BLOCK" if overlap else "PASS",
                (
                    f"{len(overlap)} train documents overlap validation"
                    if overlap
                    else "zero exact content-hash overlap"
                ),
            )
        )

    plan_checks, experiment_record = _check_experiment_plan(
        experiment_plan.resolve() if experiment_plan else None,
        model_path=model_path,
        train_path=train_path,
        readers=readers,
    )
    checks.extend(plan_checks)

    source_eval = ((train_cfg.get("checks") or {}).get("source_eval") or {})
    checks.append(
        Check(
            "source_eval",
            "PASS" if source_eval.get("enabled") and source_eval.get("batches", 0) > 0 else "BLOCK",
            (
                f"{source_eval.get('batches')} batches per source"
                if source_eval.get("enabled") and source_eval.get("batches", 0) > 0
                else "checks.source_eval must be enabled with positive batches"
            ),
        )
    )

    observability = train_cfg.get("observability") or {}
    observability_ready = (
        observability.get("enabled")
        and int(observability.get("heartbeat_interval_steps", 0)) > 0
        and bool(observability.get("state_file"))
        and bool(observability.get("events_file"))
        and bool(observability.get("metrics_file"))
    )
    checks.append(
        Check(
            "observability",
            "PASS" if observability_ready else "BLOCK",
            (
                f"heartbeat every {observability.get('heartbeat_interval_steps')} "
                "steps with durable state/events/metrics"
                if observability_ready
                else "enable durable state, events, metrics, and heartbeat"
            ),
        )
    )

    budget = train_cfg.get("budget") or {}
    throughput: dict[str, Any] | None = None
    tokens_per_step = 0
    planned_tokens = 0
    try:
        tokens_per_step = (
            int(data_cfg["block_size"])
            * int(training_cfg["micro_batch_size"])
            * int(training_cfg["grad_accum_steps"])
        )
        planned_tokens = tokens_per_step * int(training_cfg["max_steps"])
        target_tokens = int(budget.get("target_tokens", 0))
        if target_tokens <= 0:
            raise ValueError("budget.target_tokens must be positive")
        if planned_tokens < target_tokens or planned_tokens - target_tokens >= tokens_per_step:
            raise ValueError(
                f"recipe schedules {planned_tokens:,} tokens but target is "
                f"{target_tokens:,}"
            )
        checks.append(
            Check(
                "token_schedule",
                "PASS",
                f"{planned_tokens:,} tokens ({tokens_per_step:,}/step)",
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        checks.append(Check("token_schedule", "BLOCK", str(exc)))

    try:
        hourly_rate = float(budget.get("hourly_rate", 0))
        max_cost = float(budget.get("max_cost", 0))
        target_tokens = int(budget.get("target_tokens", 0))
        if min(hourly_rate, max_cost, target_tokens) <= 0:
            raise ValueError("target_tokens, hourly_rate, and max_cost must be positive")
        checks.append(
            Check(
                "budget_authorization",
                "PASS",
                f"${max_cost:.2f} cap at ${hourly_rate:.2f}/hour",
            )
        )
    except (TypeError, ValueError) as exc:
        checks.append(Check("budget_authorization", "BLOCK", str(exc)))

    try:
        throughput_value = budget.get("throughput_path")
        if not throughput_value:
            raise ValueError("budget.throughput_path is required")
        throughput_path = _resolve(root, throughput_value).resolve()
        throughput = json.loads(throughput_path.read_text(encoding="utf-8"))
        expected_workload = workload_fingerprint(
            sha256_file(model_path), data_cfg, training_cfg
        )
        if throughput.get("schema_version") != 2:
            raise ValueError("throughput result lacks current schema")
        if throughput.get("workload_sha256") != expected_workload:
            raise ValueError("throughput benchmark does not match probe workload")
        rate = float(throughput["tokens_per_sec"])
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("tokens_per_sec must be finite and positive")
        target_tokens = int(budget.get("target_tokens", 0))
        seed_count = (
            len(set(experiment_record.get("seeds") or []))
            if experiment_record
            else 1
        )
        estimated_hours = target_tokens / rate / 3600
        estimated_cost = (
            estimated_hours * float(budget.get("hourly_rate", 0)) * seed_count
        )
        if estimated_cost > float(budget.get("max_cost", 0)):
            raise ValueError(
                f"estimated group cost ${estimated_cost:.2f} exceeds configured cost cap"
            )
        checks.append(
            Check(
                "throughput_and_cost",
                "PASS",
                f"{rate:,.0f} tok/s; {estimated_hours:.2f}h/run; "
                f"{seed_count} seed(s); estimated group ${estimated_cost:.2f}",
            )
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        checks.append(Check("throughput_and_cost", "BLOCK", str(exc)))

    upload = train_cfg.get("checkpoint_upload") or {}
    upload_ready = (
        upload.get("enabled")
        and bool(upload.get("repo_id"))
        and upload.get("verify_upload", True)
    )
    checks.append(
        Check(
            "checkpoint_upload",
            "PASS" if upload_ready else "BLOCK",
            (
                f"verified uploads to {upload.get('repo_id')}"
                if upload_ready
                else "enable verified checkpoint upload and set repo_id"
            ),
        )
    )

    dirty = _git_dirty(root)
    checks.append(
        Check(
            "git_state",
            "WARN" if dirty is not False else "PASS",
            "worktree is dirty; plan records exact hashes" if dirty else "worktree clean",
        )
    )

    serialized = [asdict(check) for check in checks]
    blockers = sum(check.status == "BLOCK" for check in checks)
    warnings = sum(check.status == "WARN" for check in checks)
    bindings = {
        "model_config_sha256": (
            sha256_file(model_path) if model_path.exists() else None
        ),
        "train_config_sha256": (
            sha256_file(train_path) if train_path.exists() else None
        ),
        "experiment_plan_sha256": (
            (experiment_record or {}).get("plan_sha256")
        ),
        "train_corpus_sha256": (
            readers.get("train").manifest["corpus_sha256"]
            if readers.get("train")
            else None
        ),
        "validation_corpus_sha256": (
            readers.get("validation").manifest["corpus_sha256"]
            if readers.get("validation")
            else None
        ),
        "throughput_file_sha256": (
            sha256_file(_resolve(root, budget["throughput_path"]).resolve())
            if throughput is not None
            else None
        ),
    }
    return {
        "schema_version": 1,
        "ready": blockers == 0,
        "model_config": str(model_path),
        "train_config": str(train_path),
        "blocker_count": blockers,
        "warning_count": warnings,
        "checks": serialized,
        "bindings": bindings,
        "training_started": False,
    }


def write_preflight_receipt(report: dict[str, Any], plan_path: Path) -> Path:
    if not report.get("ready"):
        raise ValueError("cannot record a failed preflight")
    record = json.loads(plan_path.read_text(encoding="utf-8"))
    receipt = {
        "schema_version": 1,
        "group_id": record["group_id"],
        "plan_sha256": record["plan_sha256"],
        "bindings": report["bindings"],
        "training_started": False,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    path = plan_path.parent / "preflight_receipt.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != receipt:
            raise FileExistsError(
                f"refusing to replace a different preflight receipt: {path}"
            )
        return path
    with path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--experiment-plan", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_preflight(
        args.model_config,
        args.train_config,
        experiment_plan=args.experiment_plan,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for check in report["checks"]:
            print(f"[{check['status']:<5}] {check['name']}: {check['detail']}")
        print(
            f"\nREADY={str(report['ready']).lower()} "
            f"({report['blocker_count']} blockers, "
            f"{report['warning_count']} warnings; no training started)"
        )
    if report["ready"] and args.experiment_plan:
        receipt_path = write_preflight_receipt(
            report, args.experiment_plan.resolve()
        )
        if not args.json:
            print(f"Recorded execution receipt: {receipt_path}")
    raise SystemExit(0 if report["ready"] else 2)


if __name__ == "__main__":
    main()

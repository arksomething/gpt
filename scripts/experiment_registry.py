#!/usr/bin/env python3
"""Plan, inspect, annotate, and explicitly execute reproducible probe groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from scripts.run_observer import inspect_run
from scripts.validate_model_ladder import validate_config


SCHEMA_VERSION = 1
DEFAULT_SEEDS = (1337, 2027, 4099)
ROOT = Path(__file__).resolve().parents[1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_state() -> dict[str, Any]:
    status = _git_value("status", "--porcelain")
    return {
        "commit": _git_value("rev-parse", "HEAD"),
        "branch": _git_value("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def _indexed_split_info(corpus_dir: str | None) -> dict[str, Any] | None:
    if not corpus_dir:
        return None
    path = Path(corpus_dir)
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return {
            "path": str(path.resolve()),
            "available": False,
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "path": str(path.resolve()),
        "available": True,
        "manifest_sha256": _sha256_file(manifest_path),
        "corpus_sha256": manifest.get("corpus_sha256"),
        "recipe_sha256": manifest.get("recipe_sha256"),
        "tokenizer_sha256": manifest.get("tokenizer_sha256"),
        "document_count": manifest.get("document_count"),
        "token_count": manifest.get("token_count"),
        "sources": [
            source.get("source_id") for source in manifest.get("sources", [])
        ],
    }


def _data_info(train_config: dict[str, Any]) -> dict[str, Any]:
    data = train_config.get("data") or {}
    indexed = data.get("indexed") or {}
    if indexed.get("enabled", False):
        return {
            "mode": "indexed",
            "train": _indexed_split_info(indexed.get("train_dir")),
            "validation": _indexed_split_info(indexed.get("val_dir")),
            "source_weights": indexed.get("source_weights"),
            "validation_source_weights": indexed.get(
                "validation_source_weights"
            ),
        }
    return {
        "mode": "flat",
        "train_bin": data.get("train_bin"),
        "val_bin": data.get("val_bin"),
    }


def _parse_seeds(values: Iterable[str] | None) -> tuple[int, ...]:
    if not values:
        return DEFAULT_SEEDS
    seeds: list[int] = []
    for value in values:
        for item in value.split(","):
            seed = int(item.strip())
            if seed not in seeds:
                seeds.append(seed)
    if len(seeds) < 3:
        raise ValueError("probe groups require at least three distinct seeds")
    return tuple(seeds)


def plan_probe_group(
    *,
    name: str,
    model_config_path: Path,
    train_config_path: Path,
    registry_dir: Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    allow_missing_data: bool = False,
) -> tuple[Path, dict[str, Any]]:
    model_config_path = model_config_path.resolve()
    train_config_path = train_config_path.resolve()
    model_result = validate_config(model_config_path)
    base_train_config = _load_yaml(train_config_path)
    data_info = _data_info(base_train_config)
    if data_info["mode"] == "indexed" and not allow_missing_data:
        unavailable = [
            split
            for split in ("train", "validation")
            if not (data_info.get(split) or {}).get("available")
        ]
        if unavailable:
            raise FileNotFoundError(
                "indexed data missing for: " + ", ".join(unavailable)
            )

    created_at = _utc_now()
    identity = {
        "name": name,
        "created_at": created_at,
        "model_config_sha256": _sha256_file(model_config_path),
        "train_config_sha256": _sha256_file(train_config_path),
        "seeds": seeds,
        "data": data_info,
    }
    group_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + _canonical_sha256(identity)[:12]
    )
    group_dir = registry_dir.resolve() / group_id
    configs_dir = group_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=False)

    runs = []
    budget = base_train_config.get("budget") or {}
    group_max_cost = float(budget.get("max_cost", 0.0) or 0.0)
    for seed in seeds:
        run_id = f"{group_id}-seed-{seed}"
        derived = json.loads(json.dumps(base_train_config))
        derived.setdefault("training", {})["seed"] = seed
        derived["training"]["output_dir"] = str(
            (ROOT / "runs" / "probes" / group_id / f"seed-{seed}").resolve()
        )
        derived.setdefault("experiment", {})
        derived["experiment"].update(
            {
                "group_id": group_id,
                "run_id": run_id,
                "seed": seed,
            }
        )
        if group_max_cost > 0:
            derived.setdefault("budget", {})["run_max_cost"] = (
                group_max_cost / len(seeds)
            )
        derived_path = configs_dir / f"train-seed-{seed}.yaml"
        derived_path.write_text(
            yaml.safe_dump(derived, sort_keys=False),
            encoding="utf-8",
        )
        command = [
            "uv",
            "run",
            "train",
            "--model_config",
            str(model_config_path),
            "--train_config",
            str(derived_path),
        ]
        runs.append(
            {
                "run_id": run_id,
                "seed": seed,
                "train_config": str(derived_path),
                "train_config_sha256": _sha256_file(derived_path),
                "output_dir": derived["training"]["output_dir"],
                "command": command,
                "status": "planned",
            }
        )

    training = base_train_config.get("training") or {}
    data_cfg = base_train_config.get("data") or {}
    block_size = int(data_cfg.get("block_size", 0))
    tokens_per_step_single_process = (
        int(training.get("micro_batch_size", 0))
        * int(training.get("grad_accum_steps", 0))
        * block_size
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "group_id": group_id,
        "name": name,
        "created_at": created_at,
        "status": "planned",
        "git": _git_state(),
        "model": {
            **model_result,
            "config_sha256": _sha256_file(model_config_path),
        },
        "training": {
            "base_config": str(train_config_path),
            "base_config_sha256": _sha256_file(train_config_path),
            "max_steps": training.get("max_steps"),
            "tokens_per_step_single_process": tokens_per_step_single_process,
            "planned_tokens_single_process": (
                tokens_per_step_single_process * int(training.get("max_steps", 0))
            ),
        },
        "data": data_info,
        "seeds": list(seeds),
        "runs": runs,
        "execution_guard": {
            "requires_execute_flag": True,
            "required_confirmation": group_id,
        },
    }
    record["plan_sha256"] = _canonical_sha256(record)
    plan_path = group_dir / "experiment.json"
    with plan_path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    (group_dir / "events").mkdir()
    return plan_path, record


def load_group(registry_dir: Path, group_id: str) -> tuple[Path, dict[str, Any]]:
    group_dir = registry_dir.resolve() / group_id
    path = group_dir / "experiment.json"
    if not path.exists():
        raise FileNotFoundError(f"unknown experiment group: {group_id}")
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = record.pop("plan_sha256")
    actual = _canonical_sha256(record)
    record["plan_sha256"] = expected
    if actual != expected:
        raise ValueError(f"immutable experiment plan was modified: {path}")
    return group_dir, record


def append_event(
    registry_dir: Path,
    group_id: str,
    *,
    event_type: str,
    run_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> Path:
    group_dir, record = load_group(registry_dir, group_id)
    if run_id is not None and run_id not in {
        run["run_id"] for run in record["runs"]
    }:
        raise ValueError(f"run does not belong to group: {run_id}")
    event = {
        "schema_version": SCHEMA_VERSION,
        "event_id": str(uuid.uuid4()),
        "created_at": _utc_now(),
        "type": event_type,
        "group_id": group_id,
        "run_id": run_id,
        "payload": payload or {},
    }
    event["event_sha256"] = _canonical_sha256(event)
    events_dir = group_dir / "events"
    filename = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        + "-"
        + event["event_id"]
        + ".json"
    )
    path = events_dir / filename
    with path.open("x", encoding="utf-8") as handle:
        json.dump(event, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def execute_group(
    registry_dir: Path,
    group_id: str,
    confirmation: str,
) -> None:
    group_dir, record = load_group(registry_dir, group_id)
    if confirmation != group_id:
        raise SystemExit(
            "Execution confirmation mismatch. Pass the exact group ID to "
            "--confirm to authorize training."
        )
    receipt_path = group_dir / "preflight_receipt.json"
    if not receipt_path.exists():
        raise SystemExit(
            "No passing preflight receipt. Run probe-preflight against this "
            "experiment plan before execution."
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected_receipt_sha = receipt.pop("receipt_sha256")
        actual_receipt_sha = _canonical_sha256(receipt)
        receipt["receipt_sha256"] = expected_receipt_sha
        if actual_receipt_sha != expected_receipt_sha:
            raise ValueError("receipt self-hash mismatch")
        if receipt.get("group_id") != group_id:
            raise ValueError("receipt group mismatch")
        if receipt.get("plan_sha256") != record["plan_sha256"]:
            raise ValueError("receipt plan mismatch")
        bindings = receipt.get("bindings") or {}
        model_path = Path(record["model"]["path"])
        train_path = Path(record["training"]["base_config"])
        if _sha256_file(model_path) != bindings.get("model_config_sha256"):
            raise ValueError("model config changed after preflight")
        if _sha256_file(train_path) != bindings.get("train_config_sha256"):
            raise ValueError("training config changed after preflight")
        for split, binding_name in (
            ("train", "train_corpus_sha256"),
            ("validation", "validation_corpus_sha256"),
        ):
            manifest_path = Path(record["data"][split]["path"]) / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("corpus_sha256") != bindings.get(binding_name):
                raise ValueError(f"{split} corpus changed after preflight")
        base_cfg = _load_yaml(train_path)
        throughput_path = Path((base_cfg.get("budget") or {})["throughput_path"])
        if not throughput_path.is_absolute():
            throughput_path = ROOT / throughput_path
        if _sha256_file(throughput_path) != bindings.get(
            "throughput_file_sha256"
        ):
            raise ValueError("throughput result changed after preflight")
        for run in record["runs"]:
            if _sha256_file(Path(run["train_config"])) != run["train_config_sha256"]:
                raise ValueError(
                    f"derived training config changed: {run['train_config']}"
                )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid or stale preflight receipt: {exc}") from exc
    for run in record["runs"]:
        append_event(
            registry_dir,
            group_id,
            event_type="run_started",
            run_id=run["run_id"],
            payload={"command": run["command"]},
        )
        try:
            subprocess.run(run["command"], cwd=ROOT, check=True)
        except BaseException as exc:
            append_event(
                registry_dir,
                group_id,
                event_type="run_failed",
                run_id=run["run_id"],
                payload={"error": repr(exc)},
            )
            raise
        append_event(
            registry_dir,
            group_id,
            event_type="run_completed",
            run_id=run["run_id"],
        )


def group_status(
    registry_dir: Path,
    group_id: str,
    *,
    stale_after_seconds: float = 900,
) -> dict[str, Any]:
    _, record = load_group(registry_dir, group_id)
    runs = []
    for run in record["runs"]:
        report = inspect_run(
            Path(run["output_dir"]),
            stale_after_seconds=stale_after_seconds,
        )
        report["run_id"] = run["run_id"]
        report["seed"] = run["seed"]
        runs.append(report)
    observed_cost = sum(
        float(run.get("estimated_cost") or 0.0) for run in runs
    )
    statuses = {run["status"] for run in runs}
    if statuses == {"completed"}:
        status = "completed"
    elif "stale" in statuses:
        status = "stale"
    elif statuses & {"failed", "budget_stopped", "signal_stopped", "invalid"}:
        status = "attention"
    elif "running" in statuses:
        status = "running"
    else:
        status = "planned"
    return {
        "schema_version": SCHEMA_VERSION,
        "group_id": group_id,
        "name": record["name"],
        "status": status,
        "observed_cost": observed_cost,
        "group_max_cost": float(
            (_load_yaml(Path(record["training"]["base_config"])).get("budget") or {}).get(
                "max_cost", 0.0
            )
            or 0.0
        ),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=ROOT / "runs" / "experiments",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--name", required=True)
    plan_parser.add_argument("--model-config", type=Path, required=True)
    plan_parser.add_argument("--train-config", type=Path, required=True)
    plan_parser.add_argument("--seeds", nargs="*")
    plan_parser.add_argument("--allow-missing-data", action="store_true")

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("group_id")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("group_id")
    status_parser.add_argument("--stale-after-seconds", type=float, default=900)
    status_parser.add_argument("--json", action="store_true")

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--limit", type=int, default=20)

    event_parser = subparsers.add_parser("event")
    event_parser.add_argument("group_id")
    event_parser.add_argument("--type", required=True)
    event_parser.add_argument("--run-id")
    event_parser.add_argument("--payload-json", default="{}")

    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("group_id")
    execute_parser.add_argument("--confirm", required=True)

    args = parser.parse_args()
    if args.command == "plan":
        path, record = plan_probe_group(
            name=args.name,
            model_config_path=args.model_config,
            train_config_path=args.train_config,
            registry_dir=args.registry_dir,
            seeds=_parse_seeds(args.seeds),
            allow_missing_data=args.allow_missing_data,
        )
        print(f"Planned {record['group_id']}: {path}")
        for run in record["runs"]:
            print("  " + " ".join(run["command"]))
    elif args.command == "show":
        _, record = load_group(args.registry_dir, args.group_id)
        print(json.dumps(record, indent=2, sort_keys=True))
    elif args.command == "status":
        status = group_status(
            args.registry_dir,
            args.group_id,
            stale_after_seconds=args.stale_after_seconds,
        )
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(
                f"{status['group_id']}  {status['status']}  "
                f"${status['observed_cost']:.2f}/"
                f"${status['group_max_cost']:.2f}"
            )
            for run in status["runs"]:
                progress = (
                    f"{run.get('step')}/{run.get('max_steps')}"
                    if run.get("step") is not None
                    else "-"
                )
                print(
                    f"  seed {run['seed']:<10} {run['status']:<12} "
                    f"step {progress:<16} "
                    f"${float(run.get('estimated_cost') or 0.0):.2f}"
                )
    elif args.command == "list":
        groups = sorted(
            args.registry_dir.glob("*/experiment.json"),
            reverse=True,
        )[: args.limit]
        for path in groups:
            record = json.loads(path.read_text(encoding="utf-8"))
            print(
                f"{record['group_id']}  {record['name']}  "
                f"{len(record['runs'])} runs"
            )
    elif args.command == "event":
        path = append_event(
            args.registry_dir,
            args.group_id,
            event_type=args.type,
            run_id=args.run_id,
            payload=json.loads(args.payload_json),
        )
        print(path)
    elif args.command == "execute":
        execute_group(args.registry_dir, args.group_id, args.confirm)


if __name__ == "__main__":
    main()

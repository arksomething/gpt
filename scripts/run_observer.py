#!/usr/bin/env python3
"""Durable local observability and health inspection for training runs."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import socket
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
TERMINAL_STATUSES = {"completed", "failed", "budget_stopped", "signal_stopped"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


class RunObserver:
    def __init__(
        self,
        cfg: Mapping[str, Any],
        output_dir: str | os.PathLike[str],
        *,
        is_main_process: bool,
        budget: Mapping[str, Any] | None = None,
    ) -> None:
        self.enabled = bool(cfg.get("enabled", False)) and is_main_process
        self.output_dir = Path(output_dir)
        self.state_path = self.output_dir / str(
            cfg.get("state_file", "run_state.json")
        )
        self.events_path = self.output_dir / str(
            cfg.get("events_file", "events.jsonl")
        )
        self.metrics_path = self.output_dir / str(
            cfg.get("metrics_file", "metrics.jsonl")
        )
        self.heartbeat_interval_steps = max(
            1, int(cfg.get("heartbeat_interval_steps", 10))
        )
        self.budget = dict(budget or {})
        self.run_max_cost = float(
            self.budget.get(
                "run_max_cost",
                self.budget.get("max_cost", 0.0),
            )
            or 0.0
        )
        self.hourly_rate = float(self.budget.get("hourly_rate", 0.0) or 0.0)
        fractions = cfg.get("budget_alarm_fractions", [0.5, 0.8, 0.95])
        self.alarm_fractions = tuple(
            sorted(
                {
                    float(value)
                    for value in fractions
                    if 0 < float(value) < 1
                }
            )
        )
        self._emitted_alarms: set[float] = set()
        self._session_started = time.monotonic()
        self._prior_elapsed = 0.0
        self._state: dict[str, Any] = {}
        self._terminal = False
        if self.enabled:
            self._load_prior_elapsed()
            atexit.register(self._atexit)

    def _load_prior_elapsed(self) -> None:
        if not self.state_path.exists():
            return
        try:
            previous = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._prior_elapsed = max(
                0.0, float(previous.get("elapsed_seconds", 0.0))
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            self._prior_elapsed = 0.0

    def _elapsed(self) -> float:
        return self._prior_elapsed + (time.monotonic() - self._session_started)

    def _estimated_cost(self) -> float:
        return self._elapsed() / 3600.0 * self.hourly_rate

    def _write_state(self, **updates: Any) -> None:
        if not self.enabled:
            return
        self._state.update(updates)
        self._state.update(
            {
                "schema_version": SCHEMA_VERSION,
                "updated_at": utc_now(),
                "heartbeat_unix": time.time(),
                "elapsed_seconds": self._elapsed(),
                "estimated_cost": self._estimated_cost(),
                "hourly_rate": self.hourly_rate,
                "run_max_cost": self.run_max_cost,
            }
        )
        atomic_write_json(self.state_path, self._state)

    def event(self, event_type: str, **payload: Any) -> None:
        if not self.enabled:
            return
        append_jsonl(
            self.events_path,
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": utc_now(),
                "type": event_type,
                "payload": payload,
            },
        )

    def start(
        self,
        *,
        max_steps: int,
        completed_steps: int,
        tokens_processed: int,
        compatibility_fingerprint: str,
        experiment: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        self._state = {
            "status": "running",
            "started_at": utc_now(),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "step": completed_steps,
            "max_steps": max_steps,
            "tokens_processed": tokens_processed,
            "compatibility_fingerprint": compatibility_fingerprint,
            "experiment": dict(experiment or {}),
            "last_checkpoint": None,
            "last_eval": None,
            "stop_reason": None,
        }
        self._write_state()
        self.event(
            "run_started",
            step=completed_steps,
            max_steps=max_steps,
            resumed=completed_steps > 0,
        )

    def heartbeat(
        self,
        *,
        step: int,
        tokens_processed: int,
        force: bool = False,
    ) -> bool:
        if not self.enabled:
            return False
        if not force and step % self.heartbeat_interval_steps != 0:
            return self.budget_exceeded()
        self._write_state(step=step, tokens_processed=tokens_processed)
        cost = self._estimated_cost()
        if self.run_max_cost > 0:
            fraction = cost / self.run_max_cost
            for threshold in self.alarm_fractions:
                if fraction >= threshold and threshold not in self._emitted_alarms:
                    self._emitted_alarms.add(threshold)
                    self.event(
                        "budget_alarm",
                        threshold=threshold,
                        estimated_cost=cost,
                        run_max_cost=self.run_max_cost,
                        step=step,
                    )
        return self.budget_exceeded()

    def budget_exceeded(self) -> bool:
        return (
            self.enabled
            and self.run_max_cost > 0
            and self._estimated_cost() >= self.run_max_cost
        )

    def metrics(self, step: int, metrics: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        append_jsonl(
            self.metrics_path,
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": utc_now(),
                "step": step,
                "metrics": dict(metrics),
            },
        )

    def evaluation(self, step: int, loss: float, perplexity: float) -> None:
        if not self.enabled:
            return
        self._write_state(
            step=step,
            last_eval={
                "step": step,
                "loss": float(loss),
                "perplexity": float(perplexity),
                "created_at": utc_now(),
            },
        )
        self.event("evaluation_completed", step=step, loss=loss, perplexity=perplexity)

    def checkpoint(
        self, step: int, name: str, *, uploaded: bool, local_path: str | None
    ) -> None:
        if not self.enabled:
            return
        checkpoint = {
            "step": step,
            "name": name,
            "uploaded": bool(uploaded),
            "local_path": local_path,
            "created_at": utc_now(),
        }
        self._write_state(step=step, last_checkpoint=checkpoint)
        self.event("checkpoint_completed", **checkpoint)

    def terminal(self, status: str, *, reason: str | None = None) -> None:
        if not self.enabled or self._terminal:
            return
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"invalid terminal status: {status}")
        self._terminal = True
        self._write_state(status=status, stop_reason=reason, finished_at=utc_now())
        self.event("run_terminal", status=status, reason=reason)

    def _atexit(self) -> None:
        if self.enabled and not self._terminal:
            self.terminal(
                "failed",
                reason="process exited without recording a normal terminal state",
            )


def inspect_run(path: Path, stale_after_seconds: float = 900) -> dict[str, Any]:
    output_dir = path if path.is_dir() else path.parent
    state_path = path if path.is_file() else output_dir / "run_state.json"
    if not state_path.exists():
        return {
            "output_dir": str(output_dir.resolve()),
            "status": "missing",
            "healthy": False,
            "detail": "run_state.json not found",
        }
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        heartbeat_age = max(0.0, time.time() - float(state["heartbeat_unix"]))
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {
            "output_dir": str(output_dir.resolve()),
            "status": "invalid",
            "healthy": False,
            "detail": str(exc),
        }
    status = state.get("status", "unknown")
    stale = status == "running" and heartbeat_age > stale_after_seconds
    return {
        "output_dir": str(output_dir.resolve()),
        "status": "stale" if stale else status,
        "healthy": not stale and status in {"running", "completed"},
        "heartbeat_age_seconds": heartbeat_age,
        "step": state.get("step"),
        "max_steps": state.get("max_steps"),
        "tokens_processed": state.get("tokens_processed"),
        "estimated_cost": state.get("estimated_cost"),
        "run_max_cost": state.get("run_max_cost"),
        "last_eval": state.get("last_eval"),
        "last_checkpoint": state.get("last_checkpoint"),
        "stop_reason": state.get("stop_reason"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--stale-after-seconds", type=float, default=900)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-healthy", action="store_true")
    args = parser.parse_args()
    reports = [
        inspect_run(path, args.stale_after_seconds) for path in args.paths
    ]
    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        for report in reports:
            progress = ""
            if report.get("step") is not None:
                progress = f" step={report['step']}/{report.get('max_steps')}"
            cost = ""
            if report.get("estimated_cost") is not None:
                cost = (
                    f" cost=${report['estimated_cost']:.2f}"
                    f"/${report.get('run_max_cost', 0):.2f}"
                )
            print(
                f"{report['status']:<12} {report['output_dir']}"
                f"{progress}{cost}"
            )
            if report.get("stop_reason"):
                print(f"  reason: {report['stop_reason']}")
    if args.require_healthy and not all(report["healthy"] for report in reports):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

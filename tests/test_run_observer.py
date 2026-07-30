import json
import tempfile
import time
import unittest
from pathlib import Path

from scripts.run_observer import RunObserver, inspect_run


class RunObserverTests(unittest.TestCase):
    def test_writes_state_metrics_events_and_terminal_status(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            observer = RunObserver(
                {"enabled": True, "heartbeat_interval_steps": 1},
                output,
                is_main_process=True,
                budget={"hourly_rate": 2.0, "run_max_cost": 10.0},
            )
            observer.start(
                max_steps=10,
                completed_steps=0,
                tokens_processed=0,
                compatibility_fingerprint="a" * 64,
                experiment={"group_id": "fixture"},
            )
            observer.metrics(1, {"train/loss": 3.0})
            observer.heartbeat(step=1, tokens_processed=100, force=True)
            observer.evaluation(1, 2.5, 12.18)
            observer.checkpoint(
                1, "step_0000001", uploaded=True, local_path="step_0000001"
            )
            observer.terminal("completed")

            state = json.loads(
                (output / "run_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["status"], "completed")
            self.assertEqual(state["tokens_processed"], 100)
            self.assertEqual(state["last_checkpoint"]["step"], 1)
            self.assertTrue((output / "metrics.jsonl").exists())
            events = [
                json.loads(line)
                for line in (output / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(events[0]["type"], "run_started")
            self.assertEqual(events[-1]["type"], "run_terminal")
            self.assertTrue(inspect_run(output)["healthy"])

    def test_runtime_budget_and_stale_heartbeat_are_detected(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            observer = RunObserver(
                {
                    "enabled": True,
                    "heartbeat_interval_steps": 1,
                    "budget_alarm_fractions": [0.5],
                },
                output,
                is_main_process=True,
                budget={"hourly_rate": 3600.0, "run_max_cost": 1.0},
            )
            observer.start(
                max_steps=10,
                completed_steps=0,
                tokens_processed=0,
                compatibility_fingerprint="b" * 64,
            )
            observer._session_started -= 2.0
            self.assertTrue(
                observer.heartbeat(step=1, tokens_processed=100, force=True)
            )
            observer.terminal("budget_stopped", reason="fixture")

            state_path = output / "run_state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["status"] = "running"
            state["heartbeat_unix"] = time.time() - 100
            state_path.write_text(json.dumps(state), encoding="utf-8")
            report = inspect_run(output, stale_after_seconds=10)
            self.assertEqual(report["status"], "stale")
            self.assertFalse(report["healthy"])


if __name__ == "__main__":
    unittest.main()

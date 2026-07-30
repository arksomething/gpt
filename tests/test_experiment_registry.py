import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.experiment_registry import (
    append_event,
    execute_group,
    group_status,
    load_group,
    plan_probe_group,
)


ROOT = Path(__file__).parents[1]


class ExperimentRegistryTests(unittest.TestCase):
    def test_three_seed_plan_is_immutable_and_event_log_is_append_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_config = root / "train.yaml"
            train_config.write_text(
                yaml.safe_dump(
                    {
                        "data": {
                            "block_size": 128,
                            "indexed": {
                                "enabled": True,
                                "train_dir": str(root / "missing-train"),
                                "val_dir": str(root / "missing-validation"),
                                "source_weights": {"writing": 1.0},
                            },
                        },
                        "training": {
                            "micro_batch_size": 2,
                            "grad_accum_steps": 4,
                            "max_steps": 10,
                            "output_dir": "replaced",
                            "seed": 0,
                        },
                        "budget": {"max_cost": 9.0, "hourly_rate": 1.0},
                    }
                ),
                encoding="utf-8",
            )
            plan_path, record = plan_probe_group(
                name="fixture",
                model_config_path=ROOT / "configs" / "model_25m.yaml",
                train_config_path=train_config,
                registry_dir=root / "registry",
                seeds=(11, 22, 33),
                allow_missing_data=True,
            )

            self.assertTrue(plan_path.exists())
            self.assertEqual(record["seeds"], [11, 22, 33])
            self.assertEqual(len(record["runs"]), 3)
            self.assertEqual(
                record["training"]["planned_tokens_single_process"],
                2 * 4 * 128 * 10,
            )
            status = group_status(root / "registry", record["group_id"])
            self.assertEqual(status["status"], "planned")
            self.assertEqual(status["group_max_cost"], 9.0)
            self.assertEqual(len(status["runs"]), 3)
            for run, seed in zip(record["runs"], (11, 22, 33), strict=True):
                derived = yaml.safe_load(
                    Path(run["train_config"]).read_text(encoding="utf-8")
                )
                self.assertEqual(derived["training"]["seed"], seed)
                self.assertIn(record["group_id"], derived["training"]["output_dir"])
                self.assertEqual(derived["budget"]["run_max_cost"], 3.0)

            event_path = append_event(
                root / "registry",
                record["group_id"],
                event_type="preflight_passed",
                payload={"tests": 1},
            )
            self.assertTrue(event_path.exists())
            self.assertEqual(
                json.loads(event_path.read_text())["type"],
                "preflight_passed",
            )
            _, reloaded = load_group(root / "registry", record["group_id"])
            self.assertEqual(reloaded["plan_sha256"], record["plan_sha256"])

    def test_modified_plan_and_wrong_execution_confirmation_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_config = root / "train.yaml"
            train_config.write_text(
                yaml.safe_dump(
                    {
                        "data": {"block_size": 8},
                        "training": {
                            "micro_batch_size": 1,
                            "grad_accum_steps": 1,
                            "max_steps": 1,
                            "output_dir": "unused",
                        },
                    }
                ),
                encoding="utf-8",
            )
            plan_path, record = plan_probe_group(
                name="guard",
                model_config_path=ROOT / "configs" / "model_25m.yaml",
                train_config_path=train_config,
                registry_dir=root / "registry",
                seeds=(1, 2, 3),
                allow_missing_data=True,
            )
            with self.assertRaisesRegex(SystemExit, "confirmation mismatch"):
                execute_group(
                    root / "registry",
                    record["group_id"],
                    confirmation="wrong",
                )
            with self.assertRaisesRegex(SystemExit, "No passing preflight receipt"):
                execute_group(
                    root / "registry",
                    record["group_id"],
                    confirmation=record["group_id"],
                )

            modified = json.loads(plan_path.read_text())
            modified["name"] = "tampered"
            plan_path.write_text(json.dumps(modified), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "was modified"):
                load_group(root / "registry", record["group_id"])


if __name__ == "__main__":
    unittest.main()

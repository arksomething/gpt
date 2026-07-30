import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.benchmark_throughput import workload_fingerprint
from scripts.experiment_registry import _canonical_sha256
from scripts.indexed_shards import IndexedShardWriter, Source, content_hash, sha256_file
from scripts.preflight_probe import ROOT, run_preflight


class ProbePreflightTests(unittest.TestCase):
    def _corpus(
        self,
        path: Path,
        *,
        tokenizer_sha: str,
        text: str,
    ) -> None:
        with IndexedShardWriter(
            path,
            sources=[Source("writing", "Reviewed writing", "test")],
            tokenizer_sha256=tokenizer_sha,
            recipe_sha256="b" * 64,
            target_shard_tokens=100,
        ) as writer:
            writer.add_document(
                [1, 10, 2],
                source_id="writing",
                content_sha256=content_hash(text),
            )

    def _fixture(self, temporary: Path) -> tuple[Path, Path, Path]:
        model_path = ROOT / "configs" / "model_25m.yaml"
        tokenizer_path = ROOT / "tokenizer" / "spm.model"
        tokenizer_sha = sha256_file(tokenizer_path)
        train_dir = temporary / "train"
        val_dir = temporary / "validation"
        self._corpus(train_dir, tokenizer_sha=tokenizer_sha, text="train")
        self._corpus(val_dir, tokenizer_sha=tokenizer_sha, text="validation")

        train_cfg = {
            "data": {
                "block_size": 32,
                "indexed": {
                    "enabled": True,
                    "train_dir": str(train_dir),
                    "val_dir": str(val_dir),
                    "verify_hashes": True,
                },
            },
            "data_prep": {"tokenizer_model": str(tokenizer_path)},
            "training": {
                "seed": 1337,
                "micro_batch_size": 1,
                "grad_accum_steps": 1,
                "max_steps": 10,
                "precision": "bf16",
                "gradient_checkpointing": False,
            },
            "checks": {"source_eval": {"enabled": True, "batches": 2}},
            "observability": {
                "enabled": True,
                "heartbeat_interval_steps": 1,
                "state_file": "run_state.json",
                "events_file": "events.jsonl",
                "metrics_file": "metrics.jsonl",
            },
            "budget": {
                "target_tokens": 320,
                "hourly_rate": 2.0,
                "max_cost": 1.0,
                "throughput_path": str(temporary / "throughput.json"),
            },
            "checkpoint_upload": {
                "enabled": True,
                "repo_id": "test/checkpoints",
                "verify_upload": True,
            },
        }
        train_path = temporary / "train.yaml"
        train_path.write_text(yaml.safe_dump(train_cfg), encoding="utf-8")

        throughput = {
            "schema_version": 2,
            "workload_sha256": workload_fingerprint(
                sha256_file(model_path), train_cfg["data"], train_cfg["training"]
            ),
            "tokens_per_sec": 1000,
        }
        (temporary / "throughput.json").write_text(
            json.dumps(throughput), encoding="utf-8"
        )

        seeds = [1337, 2027, 4099]
        plan = {
            "group_id": "fixture-group",
            "seeds": seeds,
            "model": {"config_sha256": sha256_file(model_path)},
            "training": {"base_config_sha256": sha256_file(train_path)},
            "data": {
                "train": {
                    "corpus_sha256": json.loads(
                        (train_dir / "manifest.json").read_text(encoding="utf-8")
                    )["corpus_sha256"]
                },
                "validation": {
                    "corpus_sha256": json.loads(
                        (val_dir / "manifest.json").read_text(encoding="utf-8")
                    )["corpus_sha256"]
                },
            },
            "runs": [{"run_id": str(seed)} for seed in seeds],
        }
        plan["plan_sha256"] = _canonical_sha256(plan)
        plan_path = temporary / "experiment.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        return model_path, train_path, plan_path

    def test_complete_synthetic_preflight_passes_without_training(self):
        with tempfile.TemporaryDirectory() as raw:
            model_path, train_path, plan_path = self._fixture(Path(raw))
            report = run_preflight(
                model_path,
                train_path,
                experiment_plan=plan_path,
                root=ROOT,
            )
            self.assertTrue(report["ready"], report)
            self.assertEqual(report["blocker_count"], 0)
            self.assertFalse(report["training_started"])

    def test_overlap_and_stale_throughput_block_probe(self):
        with tempfile.TemporaryDirectory() as raw:
            temporary = Path(raw)
            model_path, train_path, plan_path = self._fixture(temporary)
            validation_index = temporary / "validation" / "documents.jsonl"
            # Integrity validation must detect even direct post-build tampering.
            validation_index.write_text(
                validation_index.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            throughput_path = temporary / "throughput.json"
            throughput = json.loads(throughput_path.read_text(encoding="utf-8"))
            throughput["workload_sha256"] = "0" * 64
            throughput_path.write_text(json.dumps(throughput), encoding="utf-8")

            report = run_preflight(
                model_path,
                train_path,
                experiment_plan=plan_path,
                root=ROOT,
            )
            blockers = {
                item["name"] for item in report["checks"] if item["status"] == "BLOCK"
            }
            self.assertIn("validation_corpus", blockers)
            self.assertIn("throughput_and_cost", blockers)
            self.assertFalse(report["ready"])


if __name__ == "__main__":
    unittest.main()

"""Gate 0 tests for the training math that is supported by the current pipeline."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import LlamaConfig, LlamaForCausalLM

from scripts.indexed_shards import IndexedShardWriter, Source, content_hash
from scripts.train import (
    RuntimeControl,
    create_data_loader,
    evaluate_indexed_by_source,
    get_batch,
    get_model_batch,
    load_training_progress,
    save_training_progress,
    set_seed,
)


def _tiny_model(seed: int = 123) -> LlamaForCausalLM:
    set_seed(seed)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=16,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    return LlamaForCausalLM(config)


def _optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )


def _train_steps(seed: int, steps: int = 4) -> tuple[dict[str, torch.Tensor], list[float]]:
    model = _tiny_model(seed)
    optimizer = _optimizer(model)
    data = np.arange(128, dtype=np.uint16) % model.config.vocab_size
    rng = np.random.default_rng(seed)
    losses = []

    model.train()
    for _ in range(steps):
        input_ids, labels = get_batch(
            data,
            batch_size=2,
            block_size=8,
            rng=rng,
            device=torch.device("cpu"),
        )
        output = model(input_ids=input_ids, labels=labels)
        optimizer.zero_grad()
        output.loss.backward()
        optimizer.step()
        losses.append(output.loss.item())

    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    return state, losses


class TrainingCorrectnessTests(unittest.TestCase):
    def test_indexed_loader_runs_a_packed_training_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "validation"):
                with IndexedShardWriter(
                    root / split,
                    sources=[Source("writing", "Writing")],
                    tokenizer_sha256="a" * 64,
                    recipe_sha256="b" * 64,
                    token_dtype="uint16",
                    target_shard_tokens=64,
                ) as writer:
                    writer.add_document(
                        [1, 3, 5, 7],
                        source_id="writing",
                        content_sha256=content_hash(f"{split}-one"),
                    )
                    writer.add_document(
                        [2, 4, 6, 8, 10],
                        source_id="writing",
                        content_sha256=content_hash(f"{split}-two"),
                    )

            loader = create_data_loader(
                {
                    "block_size": 8,
                    "indexed": {
                        "enabled": True,
                        "train_dir": str(root / "train"),
                        "val_dir": str(root / "validation"),
                        "tokenizer_sha256": "a" * 64,
                        "recipe_sha256": "b" * 64,
                    },
                },
                batch_size=2,
                block_size=8,
                device=torch.device("cpu"),
                seed=7,
            )
            input_ids, labels, model_kwargs = get_model_batch(
                loader["train"],
                loader["mode"],
                batch_size=2,
                block_size=8,
                rng=np.random.default_rng(7),
                device=torch.device("cpu"),
            )

            self.assertEqual(input_ids.shape, (2, 8))
            self.assertEqual(labels.shape, (2, 8))
            self.assertFalse(model_kwargs["use_cache"])
            self.assertTrue(torch.any(model_kwargs["position_ids"][:, 1:] == 0))
            self.assertTrue(torch.any(labels[:, 1:] == -100))

            model = _tiny_model(seed=7)
            loss = model(
                input_ids=input_ids,
                labels=labels,
                **model_kwargs,
            ).loss
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(
                any(parameter.grad is not None for parameter in model.parameters())
            )
            source_results = evaluate_indexed_by_source(
                model,
                loader["val"],
                np.random.default_rng(99),
                batches=2,
                accelerator=SimpleNamespace(
                    gather=lambda value: value,
                    num_processes=1,
                ),
            )
            self.assertEqual(set(source_results), {"writing"})
            self.assertEqual(source_results["writing"]["tokens"], 2 * 2 * 8)
            self.assertTrue(np.isfinite(source_results["writing"]["loss"]))

    def test_indexed_training_cli_writes_fingerprinted_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer_hash = "a" * 64
            recipe_hash = "b" * 64
            corpus_hashes = {}
            for split in ("train", "validation"):
                with IndexedShardWriter(
                    root / split,
                    sources=[Source("fixture", "Fixture")],
                    tokenizer_sha256=tokenizer_hash,
                    recipe_sha256=recipe_hash,
                    token_dtype="uint16",
                ) as writer:
                    writer.add_document(
                        list(range(1, 17)),
                        source_id="fixture",
                        content_sha256=content_hash(split),
                    )
                corpus_hashes[split] = json.loads(
                    (root / split / "manifest.json").read_text(encoding="utf-8")
                )["corpus_sha256"]

            model_config = root / "model.yaml"
            train_config = root / "train.yaml"
            output = root / "output"
            model_config.write_text(
                yaml.safe_dump(
                    {
                        "model": {
                            "vocab_size": 32,
                            "hidden_size": 16,
                            "intermediate_size": 32,
                            "num_hidden_layers": 1,
                            "num_attention_heads": 2,
                            "num_key_value_heads": 1,
                            "max_position_embeddings": 16,
                            "rms_norm_eps": 1e-5,
                            "rope_theta": 10000.0,
                            "hidden_act": "silu",
                            "attention_bias": False,
                            "mlp_bias": False,
                            "tie_word_embeddings": False,
                            "pad_token_id": 0,
                            "bos_token_id": 1,
                            "eos_token_id": 2,
                        }
                    }
                ),
                encoding="utf-8",
            )
            train_config.write_text(
                yaml.safe_dump(
                    {
                        "data": {
                            "block_size": 8,
                            "indexed": {
                                "enabled": True,
                                "train_dir": str(root / "train"),
                                "val_dir": str(root / "validation"),
                                "tokenizer_sha256": tokenizer_hash,
                                "recipe_sha256": recipe_hash,
                            },
                        },
                        "training": {
                            "seed": 17,
                            "micro_batch_size": 2,
                            "grad_accum_steps": 1,
                            "learning_rate": 1e-3,
                            "weight_decay": 0.0,
                            "betas": [0.9, 0.95],
                            "warmup_steps": 0,
                            "max_steps": 1,
                            "eval_interval": 100,
                            "log_interval": 1,
                            "save_interval": 100,
                            "output_dir": str(output),
                            "precision": "no",
                            "max_grad_norm": 0.0,
                            "gradient_checkpointing": False,
                            "allow_tf32": False,
                            "checkpoint_limit": 1,
                        },
                        "checkpoint_slots": {"best": 0, "good": []},
                        "checkpoint_upload": {
                            "enabled": False,
                            "local_checkpoint_mode": "persistent",
                            "keep_local_final": True,
                        },
                        "logging": {"enabled": False, "log_file": None},
                        "observability": {
                            "enabled": True,
                            "heartbeat_interval_steps": 1,
                        },
                        "runtime_control": {"enabled": False},
                        "budget": {},
                        "checks": {},
                    }
                ),
                encoding="utf-8",
            )

            training_result = subprocess.run(
                [
                    sys.executable,
                    "scripts/train.py",
                    "--model_config",
                    str(model_config),
                    "--train_config",
                    str(train_config),
                ],
                cwd=Path(__file__).parents[1],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                training_result.returncode,
                0,
                training_result.stdout + "\n" + training_result.stderr,
            )

            artifact = json.loads(
                (output / "artifacts_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                artifact["data"]["indexed"]["train"]["corpus_sha256"],
                corpus_hashes["train"],
            )
            self.assertEqual(
                artifact["data"]["indexed"]["validation"]["corpus_sha256"],
                corpus_hashes["validation"],
            )
            self.assertTrue((output / "final" / "model.pt").exists())
            self.assertTrue(
                (output / "final" / "training_progress_rank_00000.json").exists()
            )
            run_state = json.loads(
                (output / "run_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(run_state["status"], "completed")
            self.assertEqual(run_state["step"], 1)
            self.assertTrue((output / "metrics.jsonl").exists())
            self.assertTrue((output / "events.jsonl").exists())

    def test_batch_labels_are_aligned_and_hf_applies_the_only_shift(self) -> None:
        data = np.arange(24, dtype=np.uint16)
        input_ids, labels = get_batch(
            data,
            batch_size=2,
            block_size=6,
            rng=np.random.default_rng(7),
            device=torch.device("cpu"),
        )
        self.assertTrue(torch.equal(labels, input_ids))

        model = _tiny_model()
        model.eval()
        with torch.no_grad():
            output = model(input_ids=input_ids, labels=labels)
            manual_loss = F.cross_entropy(
                output.logits[:, :-1, :].contiguous().view(-1, model.config.vocab_size),
                input_ids[:, 1:].contiguous().view(-1),
            )
            double_shifted_loss = F.cross_entropy(
                output.logits[:, :-2, :].contiguous().view(-1, model.config.vocab_size),
                input_ids[:, 2:].contiguous().view(-1),
            )

        torch.testing.assert_close(output.loss, manual_loss, rtol=0, atol=1e-7)
        self.assertGreater(abs(output.loss.item() - double_shifted_loss.item()), 1e-4)

    def test_gradient_accumulation_matches_a_single_larger_batch(self) -> None:
        base = _tiny_model()
        full_batch_model = copy.deepcopy(base)
        accumulated_model = copy.deepcopy(base)
        full_optimizer = _optimizer(full_batch_model)
        accumulated_optimizer = _optimizer(accumulated_model)
        input_ids = torch.tensor(
            [
                [1, 3, 5, 7, 9, 11, 13, 2],
                [1, 4, 6, 8, 10, 12, 14, 2],
            ],
            dtype=torch.long,
        )

        full_optimizer.zero_grad()
        full_batch_model(input_ids=input_ids, labels=input_ids).loss.backward()
        full_optimizer.step()

        accumulated_optimizer.zero_grad()
        for microbatch in input_ids.chunk(2):
            loss = accumulated_model(
                input_ids=microbatch,
                labels=microbatch,
            ).loss
            (loss / 2).backward()
        accumulated_optimizer.step()

        for full_parameter, accumulated_parameter in zip(
            full_batch_model.parameters(),
            accumulated_model.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(
                full_parameter,
                accumulated_parameter,
                rtol=1e-5,
                atol=1e-7,
            )

    def test_tiny_training_is_deterministic_for_a_fixed_seed(self) -> None:
        first_state, first_losses = _train_steps(seed=2026)
        second_state, second_losses = _train_steps(seed=2026)

        self.assertEqual(first_losses, second_losses)
        self.assertEqual(first_state.keys(), second_state.keys())
        for name in first_state:
            torch.testing.assert_close(
                first_state[name],
                second_state[name],
                rtol=0,
                atol=0,
                msg=lambda message, name=name: f"{name}: {message}",
            )

    def test_tiny_model_can_intentionally_overfit_one_batch(self) -> None:
        model = _tiny_model(seed=33)
        optimizer = _optimizer(model)
        input_ids = torch.tensor(
            [
                [1, 3, 5, 7, 9, 11, 13, 2],
                [1, 4, 6, 8, 10, 12, 14, 2],
            ],
            dtype=torch.long,
        )

        model.train()
        with torch.no_grad():
            initial_loss = model(input_ids=input_ids, labels=input_ids).loss.item()
        for _ in range(80):
            optimizer.zero_grad()
            loss = model(input_ids=input_ids, labels=input_ids).loss
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            final_loss = model(input_ids=input_ids, labels=input_ids).loss.item()

        self.assertLess(final_loss, initial_loss * 0.7)

    def test_reset_position_ids_isolate_packed_documents(self) -> None:
        model = _tiny_model(seed=91)
        model.eval()
        original = torch.tensor(
            [[1, 3, 5, 7, 2, 4, 6, 8]],
            dtype=torch.long,
        )
        changed_first_document = torch.tensor(
            [[9, 11, 13, 15, 2, 4, 6, 8]],
            dtype=torch.long,
        )
        packed_positions = torch.tensor(
            [[0, 1, 2, 3, 0, 1, 2, 3]],
            dtype=torch.long,
        )

        with torch.no_grad():
            original_logits = model(
                input_ids=original,
                position_ids=packed_positions,
                use_cache=False,
            ).logits
            changed_logits = model(
                input_ids=changed_first_document,
                position_ids=packed_positions,
                use_cache=False,
            ).logits
            unmasked_changed_logits = model(
                input_ids=changed_first_document,
            ).logits

        torch.testing.assert_close(
            original_logits[:, 4:, :],
            changed_logits[:, 4:, :],
            rtol=0,
            atol=0,
        )
        self.assertGreater(
            torch.max(
                torch.abs(
                    original_logits[:, 4:, :] - unmasked_changed_logits[:, 4:, :]
                )
            ).item(),
            1e-6,
        )

    def test_hf_save_reload_preserves_logits_and_loss(self) -> None:
        model = _tiny_model()
        model.eval()
        input_ids = torch.tensor([[1, 7, 8, 9, 2]], dtype=torch.long)
        with torch.no_grad():
            before = model(input_ids=input_ids, labels=input_ids)

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            restored = LlamaForCausalLM.from_pretrained(directory)
            restored.eval()
            with torch.no_grad():
                after = restored(input_ids=input_ids, labels=input_ids)

        torch.testing.assert_close(before.logits, after.logits, rtol=0, atol=0)
        torch.testing.assert_close(before.loss, after.loss, rtol=0, atol=0)

    def test_training_progress_restores_the_exact_next_batches(self) -> None:
        data = np.arange(257, dtype=np.uint16)
        uninterrupted_rng = np.random.default_rng(917)

        for _ in range(3):
            get_batch(
                data,
                batch_size=3,
                block_size=7,
                rng=uninterrupted_rng,
                device=torch.device("cpu"),
            )

        with tempfile.TemporaryDirectory() as directory:
            save_training_progress(
                directory,
                completed_steps=3,
                batch_rng=uninterrupted_rng,
                process_index=2,
            )
            expected_batches = [
                get_batch(
                    data,
                    batch_size=3,
                    block_size=7,
                    rng=uninterrupted_rng,
                    device=torch.device("cpu"),
                )
                for _ in range(2)
            ]

            resumed_rng = np.random.default_rng(123456)
            completed_steps = load_training_progress(
                directory,
                batch_rng=resumed_rng,
                process_index=2,
            )
            resumed_batches = [
                get_batch(
                    data,
                    batch_size=3,
                    block_size=7,
                    rng=resumed_rng,
                    device=torch.device("cpu"),
                )
                for _ in range(2)
            ]

        self.assertEqual(completed_steps, 3)
        for expected, resumed in zip(expected_batches, resumed_batches, strict=True):
            torch.testing.assert_close(expected[0], resumed[0], rtol=0, atol=0)
            torch.testing.assert_close(expected[1], resumed[1], rtol=0, atol=0)

    def test_training_progress_is_rank_specific_and_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rng = np.random.default_rng(4)
            save_training_progress(
                directory,
                completed_steps=8,
                batch_rng=rng,
                process_index=0,
            )
            with self.assertRaisesRegex(
                SystemExit,
                "lacks exact-resume training progress",
            ):
                load_training_progress(
                    directory,
                    batch_rng=np.random.default_rng(4),
                    process_index=1,
                )

    def test_training_progress_rejects_incompatible_run_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            save_training_progress(
                directory,
                completed_steps=2,
                batch_rng=np.random.default_rng(8),
                compatibility_fingerprint="a" * 64,
            )
            with self.assertRaisesRegex(
                SystemExit,
                "compatibility fingerprint mismatch",
            ):
                load_training_progress(
                    directory,
                    batch_rng=np.random.default_rng(8),
                    expected_compatibility_fingerprint="b" * 64,
                )

    def test_runtime_command_cursor_prevents_replay_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands = root / "commands.jsonl"
            commands.write_text(
                json.dumps({"cmd": "stop_training", "save": True}) + "\n",
                encoding="utf-8",
            )
            config = {
                "enabled": True,
                "poll_interval_steps": 1,
                "command_path": str(commands),
            }
            first = RuntimeControl(config, str(root))
            _, first_commands = first.poll(1)
            self.assertEqual(len(first_commands), 1)

            restarted = RuntimeControl(config, str(root))
            _, replayed = restarted.poll(1)
            self.assertEqual(replayed, [])

            with commands.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"cmd": "save_now"}) + "\n")
            _, new_commands = restarted.poll(2)
            self.assertEqual(len(new_commands), 1)
            self.assertEqual(new_commands[0]["cmd"], "save_now")

    def test_interrupted_resume_matches_uninterrupted_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_bin = root / "train.bin"
            val_bin = root / "val.bin"
            model_config = root / "model.yaml"
            uninterrupted_config = root / "uninterrupted.yaml"
            interrupted_config = root / "interrupted.yaml"
            command_path = root / "commands.jsonl"
            uninterrupted_output = root / "uninterrupted"
            interrupted_output = root / "interrupted"

            tokens = (np.arange(256, dtype=np.uint16) % 32).astype(np.uint16)
            tokens.tofile(train_bin)
            tokens[::-1].tofile(val_bin)
            model_config.write_text(
                yaml.safe_dump(
                    {
                        "model": {
                            "vocab_size": 32,
                            "hidden_size": 16,
                            "intermediate_size": 32,
                            "num_hidden_layers": 1,
                            "num_attention_heads": 2,
                            "num_key_value_heads": 1,
                            "max_position_embeddings": 16,
                            "rms_norm_eps": 1e-5,
                            "rope_theta": 10000.0,
                            "hidden_act": "silu",
                            "attention_bias": False,
                            "mlp_bias": False,
                            "tie_word_embeddings": False,
                            "pad_token_id": 0,
                            "bos_token_id": 1,
                            "eos_token_id": 2,
                        }
                    }
                ),
                encoding="utf-8",
            )

            def write_train_config(path: Path, output: Path, runtime_enabled: bool) -> None:
                path.write_text(
                    yaml.safe_dump(
                        {
                            "data": {
                                "train_bin": str(train_bin),
                                "val_bin": str(val_bin),
                                "block_size": 8,
                                "dtype": "uint16",
                                "streaming": {"enabled": False},
                            },
                            "training": {
                                "seed": 77,
                                "micro_batch_size": 2,
                                "grad_accum_steps": 1,
                                "learning_rate": 1e-3,
                                "weight_decay": 0.0,
                                "betas": [0.9, 0.95],
                                "warmup_steps": 0,
                                "max_steps": 2,
                                "eval_interval": 100,
                                "log_interval": 1,
                                "save_interval": 1,
                                "output_dir": str(output),
                                "precision": "no",
                                "max_grad_norm": 0.0,
                                "gradient_checkpointing": False,
                                "allow_tf32": False,
                                "checkpoint_limit": 4,
                            },
                            "checkpoint_slots": {"best": 0, "good": []},
                            "checkpoint_upload": {
                                "enabled": False,
                                "local_checkpoint_mode": "persistent",
                                "keep_local_final": True,
                            },
                            "logging": {"enabled": False, "log_file": None},
                            "runtime_control": {
                                "enabled": runtime_enabled,
                                "poll_interval_steps": 1,
                                "command_path": str(command_path),
                            },
                            "budget": {},
                            "checks": {},
                        }
                    ),
                    encoding="utf-8",
                )

            write_train_config(uninterrupted_config, uninterrupted_output, False)
            write_train_config(interrupted_config, interrupted_output, True)

            def run_training(
                config: Path, *extra: str, expect_success: bool = True
            ) -> None:
                result = subprocess.run(
                    [
                        sys.executable,
                        "scripts/train.py",
                        "--model_config",
                        str(model_config),
                        "--train_config",
                        str(config),
                        *extra,
                    ],
                    cwd=Path(__file__).parents[1],
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                    capture_output=True,
                    text=True,
                )
                if expect_success:
                    self.assertEqual(
                        result.returncode,
                        0,
                        result.stdout + "\n" + result.stderr,
                    )
                else:
                    self.assertNotEqual(result.returncode, 0)

            run_training(uninterrupted_config)
            command_path.write_text(
                json.dumps({"cmd": "stop_training", "save": True}) + "\n",
                encoding="utf-8",
            )
            run_training(interrupted_config, expect_success=False)
            command_path.unlink()
            run_training(
                interrupted_config,
                "--resume_from",
                str(interrupted_output / "step_0000001"),
            )

            uninterrupted_state = torch.load(
                uninterrupted_output / "final" / "model.pt",
                map_location="cpu",
                weights_only=True,
            )
            resumed_state = torch.load(
                interrupted_output / "final" / "model.pt",
                map_location="cpu",
                weights_only=True,
            )

        self.assertEqual(uninterrupted_state.keys(), resumed_state.keys())
        for name in uninterrupted_state:
            torch.testing.assert_close(
                uninterrupted_state[name],
                resumed_state[name],
                rtol=0,
                atol=0,
                msg=lambda message, name=name: f"{name}: {message}",
            )


if __name__ == "__main__":
    unittest.main()

import json
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from scripts.chat_format import (
    IGNORE_INDEX,
    ChatFormatError,
    encode_conversation,
    encode_generation_prompt,
)
from scripts.conversation_eval import build_review, score_checks
from scripts.indexed_shards import IndexedCorpusSampler, IndexedShardReader
from scripts.prepare_chat_data import prepare_chat_corpus
from scripts.train import (
    _load_initial_weights,
    load_training_progress,
    save_training_progress,
)
from scripts.generate_chat_data import _extract_json, _overlaps_eval
from scripts.chat_review import apply_decisions, create_pack


def _read_json_records(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class _FixtureTokenizer:
    def bos_id(self):
        return 1

    def encode(self, text, out_type=int):
        return [3 + (ord(char) % 23) for char in text]


class ChatFormatTests(unittest.TestCase):
    def test_only_assistant_content_and_end_marker_are_supervised(self):
        tokenizer = _FixtureTokenizer()
        encoded = encode_conversation(
            tokenizer,
            [
                {"role": "system", "content": "Be direct."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "One word?"},
                {"role": "assistant", "content": "Okay"},
            ],
        )
        self.assertEqual(len(encoded.input_ids), len(encoded.labels))
        self.assertEqual(len(encoded.assistant_spans), 2)
        for index, label in enumerate(encoded.labels):
            supervised = any(start <= index < stop for start, stop in encoded.assistant_spans)
            self.assertEqual(label != IGNORE_INDEX, supervised)
            if supervised:
                self.assertEqual(label, encoded.input_ids[index])

    def test_generation_requires_trailing_user(self):
        tokenizer = _FixtureTokenizer()
        prompt = encode_generation_prompt(
            tokenizer,
            [{"role": "user", "content": "Hello"}],
        )
        self.assertGreater(len(prompt), 1)
        with self.assertRaises(ChatFormatError):
            encode_generation_prompt(
                tokenizer,
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ],
            )

    def test_deterministic_conversation_checks(self):
        result = score_checks(
            '["oatmeal", "toast", "fruit"]',
            {"json_type": "list", "json_length": 3, "max_words": 8},
        )
        self.assertTrue(result["passed"])
        failed = score_checks(
            "You should try to relax.",
            {"must_not_include": ["you should"], "requires_question": True},
        )
        self.assertFalse(failed["passed"])

    def test_synthetic_response_parsing_and_eval_overlap(self):
        value = _extract_json(
            '```json\n{"messages":[{"role":"user","content":"Hi"},'
            '{"role":"assistant","content":"Hello"}]}\n```'
        )
        self.assertEqual(len(value["messages"]), 2)
        frozen = {tuple("one two three four".split())}
        self.assertTrue(
            _overlaps_eval(
                [{"role": "user", "content": "zero one two three four five"}],
                frozen,
                width=4,
            )
        )


class ChatCorpusTests(unittest.TestCase):
    @staticmethod
    def _train_tokenizer(root: Path) -> Path:
        corpus = root / "tokenizer.txt"
        corpus.write_text(
            "\n".join(
                [
                    "Hello there this is tokenizer training text.",
                    "Please answer clearly and conversationally.",
                    "A follow up question has enough ordinary words.",
                ]
                * 20
            ),
            encoding="utf-8",
        )
        prefix = root / "spm"
        spm.SentencePieceTrainer.train(
            input=str(corpus),
            model_prefix=str(prefix),
            vocab_size=64,
            model_type="bpe",
            bos_id=1,
            eos_id=2,
            pad_id=3,
            unk_id=0,
        )
        return prefix.with_suffix(".model")

    def test_prepared_corpus_masks_user_tokens_in_sampled_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer_path = self._train_tokenizer(root)
            input_path = root / "chat.jsonl"
            records = []
            for index in range(80):
                records.append(
                    {
                        "id": f"row-{index}",
                        "source_id": "synthetic",
                        "source_name": "Synthetic fixture",
                        "license": "Apache-2.0",
                        "synthetic": True,
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Please answer fixture question number {index}.",
                            },
                            {
                                "role": "assistant",
                                "content": f"This is fixture answer number {index}.",
                            },
                        ],
                    }
                )
            input_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            output = root / "prepared"
            manifest = prepare_chat_corpus(
                input_path=input_path,
                output_dir=output,
                tokenizer_path=tokenizer_path,
                validation_fraction=0.2,
                target_shard_tokens=10_000,
                max_tokens=256,
                require_human_keep=False,
            )
            self.assertGreater(manifest["counts"]["train_documents"], 0)
            self.assertGreater(manifest["counts"]["validation_documents"], 0)

            reader = IndexedShardReader(output / "train")
            self.assertTrue(
                all(document.metadata.get("supervision_spans") for document in reader.documents)
            )
            sampler = IndexedCorpusSampler(reader)
            batch = sampler.sample_batch(
                batch_size=2,
                block_size=128,
                rng=np.random.default_rng(7),
            )
            self.assertTrue(np.any(batch.labels == IGNORE_INDEX))
            self.assertTrue(np.any(batch.labels != IGNORE_INDEX))
            supervised = batch.labels != IGNORE_INDEX
            self.assertTrue(np.array_equal(batch.labels[supervised], batch.input_ids[supervised]))

    def test_blind_review_pack_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.jsonl"
            right = root / "right.jsonl"
            base = {
                "schema_version": 1,
                "case_id": "case-one",
                "category": "fixture",
                "turns": [
                    {
                        "user": "Remember my preferred color is green.",
                        "assistant": "",
                        "rubric": "Be natural.",
                        "deterministic": {"passed": True},
                    },
                    {
                        "user": "What color did I say?",
                        "assistant": "",
                        "rubric": "Use the earlier context.",
                        "deterministic": {"passed": True},
                    },
                ],
            }
            left_record = json.loads(json.dumps(base))
            right_record = json.loads(json.dumps(base))
            left_record["turns"][0]["assistant"] = "Left response"
            right_record["turns"][0]["assistant"] = "Right response"
            left_record["turns"][1]["assistant"] = "Green."
            right_record["turns"][1]["assistant"] = "I do not know."
            left.write_text(json.dumps(left_record) + "\n", encoding="utf-8")
            right.write_text(json.dumps(right_record) + "\n", encoding="utf-8")
            args = type(
                "Args",
                (),
                {"left": left, "right": right, "output_dir": root / "review"},
            )()
            build_review(args)
            self.assertTrue((root / "review" / "review.md").exists())
            rows = (root / "review" / "reviews.csv").read_text(encoding="utf-8")
            self.assertIn("_a_source", rows)
            self.assertIn("case-one", rows)
            review = (root / "review" / "review.md").read_text(encoding="utf-8")
            self.assertGreaterEqual(
                review.count("Remember my preferred color is green."),
                2,
            )

    def test_chat_data_review_exports_only_explicit_keeps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "id": f"record-{index}",
                            "source_id": "fixture",
                            "messages": [
                                {"role": "user", "content": f"Question {index}"},
                                {"role": "assistant", "content": f"Answer {index}"},
                            ],
                        }
                    )
                    for index in range(2)
                )
                + "\n",
                encoding="utf-8",
            )
            pack = root / "pack"
            create_pack(
                type(
                    "Args",
                    (),
                    {
                        "input": source,
                        "output_dir": pack,
                        "sample_size": None,
                        "seed": 1,
                    },
                )()
            )
            with (pack / "review.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["keep"] = "yes"
            rows[1]["keep"] = "no"
            with (pack / "review.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            accepted = root / "accepted.jsonl"
            apply_decisions(
                type(
                    "Args",
                    (),
                    {
                        "input": source,
                        "review_csv": pack / "review.csv",
                        "output": accepted,
                        "require_complete": True,
                    },
                )()
            )
            exported = _read_json_records(accepted)
            self.assertEqual(len(exported), 1)
            self.assertTrue(exported[0]["human_review"]["keep"])

            source.write_text(
                source.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "input hash"):
                apply_decisions(
                    type(
                        "Args",
                        (),
                        {
                            "input": source,
                            "review_csv": pack / "review.csv",
                            "output": root / "should-not-exist.jsonl",
                            "require_complete": True,
                        },
                    )()
                )

    def test_preparation_can_require_explicit_human_keeps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer_path = self._train_tokenizer(root)
            input_path = root / "chat.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "id": "unreviewed",
                        "source_id": "synthetic",
                        "messages": [
                            {"role": "user", "content": "Please answer this."},
                            {"role": "assistant", "content": "Here is an answer."},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not an explicit human keep"):
                prepare_chat_corpus(
                    input_path=input_path,
                    output_dir=root / "prepared",
                    tokenizer_path=tokenizer_path,
                    validation_fraction=0.5,
                    target_shard_tokens=10_000,
                    max_tokens=256,
                    require_human_keep=True,
                )

    def test_initialization_checkpoint_is_fingerprint_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "base" / "final"
            checkpoint.mkdir(parents=True)
            config_path = root / "model.yaml"
            tokenizer_path = root / "tokenizer.model"
            config_path.write_text("model: fixture\n", encoding="utf-8")
            tokenizer_path.write_bytes(b"tokenizer fixture")
            config = LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
            )
            source = LlamaForCausalLM(config)
            torch.save(source.state_dict(), checkpoint / "model.pt")
            file_hash = lambda path: __import__("hashlib").sha256(
                Path(path).read_bytes()
            ).hexdigest()
            (root / "base" / "artifacts_manifest.json").write_text(
                json.dumps(
                    {
                        "model_config": {"sha256": file_hash(config_path)},
                        "tokenizer": {"sha256": file_hash(tokenizer_path)},
                        "compatibility_fingerprint": "parent",
                    }
                ),
                encoding="utf-8",
            )
            target = LlamaForCausalLM(config)
            lineage = _load_initial_weights(
                target,
                checkpoint,
                model_config_path=config_path,
                tokenizer_path=tokenizer_path,
            )
            self.assertEqual(lineage["parent_compatibility_fingerprint"], "parent")
            for expected, actual in zip(source.parameters(), target.parameters()):
                self.assertTrue(torch.equal(expected, actual))

            tokenizer_path.write_bytes(b"changed tokenizer")
            with self.assertRaises(SystemExit):
                _load_initial_weights(
                    target,
                    checkpoint,
                    model_config_path=config_path,
                    tokenizer_path=tokenizer_path,
                )

    def test_supervised_token_counter_survives_exact_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            original = np.random.default_rng(9)
            save_training_progress(
                directory,
                12,
                original,
                compatibility_fingerprint="fixture",
                counters={"supervised_tokens": 345},
            )
            restored = np.random.default_rng(1)
            counters = {}
            steps = load_training_progress(
                directory,
                restored,
                expected_compatibility_fingerprint="fixture",
                counters=counters,
            )
            self.assertEqual(steps, 12)
            self.assertEqual(counters["supervised_tokens"], 345)
            self.assertEqual(original.integers(0, 1000), restored.integers(0, 1000))


if __name__ == "__main__":
    unittest.main()

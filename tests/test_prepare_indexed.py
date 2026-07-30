import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import sentencepiece as spm
import yaml

from scripts.indexed_shards import IndexedShardReader, Source, content_hash
from scripts.prepare_data import (
    FilterConfig,
    PreparedDocument,
    partition_documents,
    preparation_recipe_sha256,
    process_and_tokenize,
    main as prepare_main,
    write_documents_to_indexed,
    write_tokens_to_memmap,
)


class FakeTokenizer:
    def unk_id(self):
        return 0

    def encode(self, text, out_type=int):
        return [1 + (ord(character) % 29) for character in text if not character.isspace()]


class Interrupted:
    interrupted = True


class MutableInterrupt:
    interrupted = False


class IndexedPreparationTests(unittest.TestCase):
    def test_filter_tokenize_retains_source_and_content_hash(self):
        text = "A clear paragraph with enough ordinary alphabetic prose."
        stats = self._stats()
        documents = list(
            process_and_tokenize(
                iter([("fineweb", text)]),
                self._filter_config(),
                FakeTokenizer(),
                deduplicator=None,
                stats=stats,
                shuffle_buffer_size=1,
                seed=12,
            )
        )

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].source_id, "fineweb")
        self.assertEqual(documents[0].content_sha256, content_hash(text))
        self.assertEqual(documents[0].metadata["chunk_characters"], len(text))
        self.assertTrue(documents[0].tokens)

    def test_canonical_source_metadata_survives_preparation(self):
        text = "A manually accepted essay with clear analytical prose."
        stats = self._stats()
        stats["essays"] = stats["fineweb"].__class__()
        documents = list(
            process_and_tokenize(
                iter(
                    [
                        (
                            "essays",
                            text,
                            {
                                "canonical_document_id": content_hash(text),
                                "license": "CC-BY-4.0",
                                "review": {"keep_yes_no": "yes"},
                            },
                        )
                    ]
                ),
                self._filter_config(),
                FakeTokenizer(),
                deduplicator=None,
                stats=stats,
                shuffle_buffer_size=1,
                seed=12,
            )
        )

        self.assertEqual(documents[0].source_id, "essays")
        self.assertEqual(documents[0].metadata["license"], "CC-BY-4.0")
        self.assertEqual(
            documents[0].metadata["canonical_document_id"],
            content_hash(text),
        )

    def test_prepared_documents_remain_compatible_with_flat_writer(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "train.bin"
            checkpoint = Path(temporary) / "checkpoint.json"
            document = PreparedDocument(
                tokens=[3, 4, 5],
                source_id="fineweb",
                content_sha256=content_hash("flat"),
                metadata={},
            )
            count, interrupted = write_tokens_to_memmap(
                iter([document]),
                str(output),
                target_tokens=4,
                dtype=np.uint16,
                eos_token_id=2,
                checkpoint_path=str(checkpoint),
                checkpoint_interval=3600,
                log_interval=3600,
            )

            self.assertEqual(count, 4)
            self.assertFalse(interrupted)
            self.assertEqual(
                np.fromfile(output, dtype=np.uint16).tolist(),
                [3, 4, 5, 2],
            )

    def test_flat_resume_verifies_replayed_token_contents(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "train.bin"
            checkpoint = Path(temporary) / "checkpoint.json"
            np.asarray([3, 4, 5, 2], dtype=np.uint16).tofile(output)

            with self.assertRaisesRegex(
                SystemExit,
                "regenerated tokens differ",
            ):
                write_tokens_to_memmap(
                    iter([[3, 9, 5]]),
                    str(output),
                    target_tokens=8,
                    dtype=np.uint16,
                    eos_token_id=2,
                    checkpoint_path=str(checkpoint),
                    checkpoint_interval=3600,
                    log_interval=3600,
                    resume_existing_tokens=4,
                )

            self.assertEqual(
                np.fromfile(output, dtype=np.uint16).tolist(),
                [3, 4, 5, 2],
            )

    def test_indexed_writer_publishes_provenance_and_eos(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "train"
            documents = [
                PreparedDocument(
                    tokens=[3, 4, 5],
                    source_id="fineweb",
                    content_sha256=content_hash("one"),
                    metadata={"ordinal": 1},
                ),
                PreparedDocument(
                    tokens=[6, 7],
                    source_id="fineweb",
                    content_sha256=content_hash("two"),
                    metadata={"ordinal": 2},
                ),
            ]
            count, interrupted, fingerprint = write_documents_to_indexed(
                iter(documents),
                str(output),
                target_tokens=7,
                token_dtype="uint16",
                eos_token_id=2,
                tokenizer_sha256="a" * 64,
                recipe_sha256="b" * 64,
                sources=[Source("fineweb", "FineWeb")],
                target_shard_tokens=5,
                log_interval=3600,
            )

            reader = IndexedShardReader(output)
            self.assertEqual(count, 7)
            self.assertFalse(interrupted)
            self.assertEqual(fingerprint, reader.manifest["corpus_sha256"])
            self.assertEqual(reader.read_tokens(0).tolist(), [3, 4, 5, 2])
            self.assertEqual(reader.read_tokens(1).tolist(), [6, 7, 2])
            self.assertEqual(reader.documents[1].metadata["ordinal"], 2)

    def test_interrupted_indexed_writer_publishes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "train"
            result = write_documents_to_indexed(
                iter(
                    [
                        PreparedDocument(
                            tokens=[3, 4],
                            source_id="fineweb",
                            content_sha256=content_hash("interrupted"),
                            metadata={},
                        )
                    ]
                ),
                str(output),
                target_tokens=3,
                token_dtype="uint16",
                eos_token_id=2,
                tokenizer_sha256="a" * 64,
                recipe_sha256="b" * 64,
                sources=[Source("fineweb", "FineWeb")],
                target_shard_tokens=10,
                interrupt_handler=Interrupted(),
            )

            self.assertEqual(result, (0, True, None))
            self.assertFalse(output.exists())

    def test_indexed_writer_resumes_verified_committed_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "train"
            documents = [
                PreparedDocument(
                    tokens=[3, 4],
                    source_id="fineweb",
                    content_sha256=content_hash("one"),
                    metadata={},
                ),
                PreparedDocument(
                    tokens=[5, 6],
                    source_id="fineweb",
                    content_sha256=content_hash("two"),
                    metadata={},
                ),
            ]
            interrupt = MutableInterrupt()

            def interrupted_documents():
                yield documents[0]
                interrupt.interrupted = True
                yield documents[1]

            first = write_documents_to_indexed(
                interrupted_documents(),
                str(output),
                target_tokens=6,
                token_dtype="uint16",
                eos_token_id=2,
                tokenizer_sha256="a" * 64,
                recipe_sha256="b" * 64,
                sources=[Source("fineweb", "FineWeb")],
                target_shard_tokens=10,
                interrupt_handler=interrupt,
            )
            self.assertEqual(first, (3, True, None))
            self.assertFalse(output.exists())
            self.assertTrue(output.with_name("train.staging").exists())

            resumed = write_documents_to_indexed(
                iter(documents),
                str(output),
                target_tokens=6,
                token_dtype="uint16",
                eos_token_id=2,
                tokenizer_sha256="a" * 64,
                recipe_sha256="b" * 64,
                sources=[Source("fineweb", "FineWeb")],
                target_shard_tokens=10,
                resume=True,
            )
            self.assertEqual(resumed[:2], (6, False))
            reader = IndexedShardReader(output)
            self.assertEqual(reader.read_tokens(0).tolist(), [3, 4, 2])
            self.assertEqual(reader.read_tokens(1).tolist(), [5, 6, 2])

    def test_recipe_fingerprint_is_deterministic_and_input_sensitive(self):
        config = self._filter_config()
        first = preparation_recipe_sha256(
            filter_cfg=config,
            tokenizer_sha256="a" * 64,
            c4_weight=0.0,
            wiki_weight=0.0,
            fineweb_weight=1.0,
            shuffle_buffer=8,
            seed=3,
            max_unk_ratio=0.01,
            validation_fraction=0.1,
        )
        second = preparation_recipe_sha256(
            filter_cfg=config,
            tokenizer_sha256="a" * 64,
            c4_weight=0.0,
            wiki_weight=0.0,
            fineweb_weight=1.0,
            shuffle_buffer=8,
            seed=3,
            max_unk_ratio=0.01,
            validation_fraction=0.1,
        )
        changed = preparation_recipe_sha256(
            filter_cfg=config,
            tokenizer_sha256="a" * 64,
            c4_weight=0.0,
            wiki_weight=0.0,
            fineweb_weight=1.0,
            shuffle_buffer=8,
            seed=4,
            max_unk_ratio=0.01,
            validation_fraction=0.1,
        )

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_content_hash_partition_is_disjoint_and_complete(self):
        documents = [
            ("fineweb", f"document number {index} with stable text")
            for index in range(500)
        ]
        train = list(
            partition_documents(
                iter(documents),
                split="train",
                validation_fraction=0.2,
                seed=19,
            )
        )
        validation = list(
            partition_documents(
                iter(reversed(documents)),
                split="validation",
                validation_fraction=0.2,
                seed=19,
            )
        )
        train_text = {text for _, text in train}
        validation_text = {text for _, text in validation}

        self.assertFalse(train_text & validation_text)
        self.assertEqual(train_text | validation_text, {text for _, text in documents})
        self.assertGreater(len(validation_text), 70)
        self.assertLess(len(validation_text), 130)

    def test_prepare_cli_builds_disjoint_indexed_splits_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer_prefix = root / "tokenizer"
            tokenizer_corpus = [
                (
                    "This is a clear synthetic fixture paragraph number "
                    f"{index} with ordinary words for tokenizer training."
                )
                for index in range(300)
            ]
            spm.SentencePieceTrainer.train(
                sentence_iterator=iter(tokenizer_corpus),
                model_prefix=str(tokenizer_prefix),
                vocab_size=96,
                model_type="bpe",
                bos_id=1,
                eos_id=2,
                pad_id=3,
                unk_id=0,
                minloglevel=2,
            )
            output = root / "indexed"
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "data_prep": {
                            "tokenizer_model": str(tokenizer_prefix) + ".model",
                            "out_dir": str(output),
                            "output_format": "indexed",
                            "target_shard_tokens": 128,
                            "train_tokens": 400,
                            "val_tokens": 200,
                            "validation_fraction": 0.3,
                            "c4_weight": 0.0,
                            "wiki_weight": 0.0,
                            "fineweb_weight": 1.0,
                            "shuffle_buffer": 1,
                            "seed": 41,
                            "log_interval": 3600,
                            "tokenizer_workers": 1,
                            "tokenizer_prefetch": 1,
                            "min_chars": 1,
                            "max_chars": 10_000,
                            "min_alpha_ratio": 0.1,
                            "max_repeated_chars": 20,
                            "max_weird_ratio": 1.0,
                            "max_short_line_ratio": 1.0,
                            "max_caps_ratio": 1.0,
                            "dedup_enabled": False,
                            "max_chunk_chars": 10_000,
                            "min_chunk_chars": 1,
                        }
                    }
                ),
                encoding="utf-8",
            )

            def fixture_stream(seed=42):
                del seed
                yield from tokenizer_corpus

            argv = [
                "prepare-data",
                "--config",
                str(config_path),
                "--overwrite",
                "--no_log",
            ]
            with (
                patch("scripts.prepare_data.stream_fineweb", fixture_stream),
                patch.object(sys, "argv", argv),
            ):
                prepare_main()

            train_reader = IndexedShardReader(output / "train")
            validation_reader = IndexedShardReader(output / "validation")
            train_hashes = {
                document.content_sha256 for document in train_reader.documents
            }
            validation_hashes = {
                document.content_sha256
                for document in validation_reader.documents
            }
            metadata = json.loads(
                (output / "data_meta.json").read_text(encoding="utf-8")
            )

            self.assertFalse(train_hashes & validation_hashes)
            self.assertEqual(metadata["output_format"], "indexed")
            self.assertEqual(
                metadata["indexed"]["train_corpus_sha256"],
                train_reader.manifest["corpus_sha256"],
            )
            self.assertEqual(
                metadata["indexed"]["validation_corpus_sha256"],
                validation_reader.manifest["corpus_sha256"],
            )

    @staticmethod
    def _filter_config():
        return FilterConfig(
            min_chars=1,
            max_chars=10_000,
            min_alpha_ratio=0.1,
            max_repeated_chars=20,
            max_weird_ratio=1.0,
            max_short_line_ratio=1.0,
            max_caps_ratio=1.0,
            max_chunk_chars=10_000,
            min_chunk_chars=1,
            dedup_enabled=False,
        )

    @staticmethod
    def _stats():
        from scripts.filters import FilterStats

        return {
            "c4": FilterStats(),
            "wiki": FilterStats(),
            "fineweb": FilterStats(),
            "dedup_rejects": 0,
        }


if __name__ == "__main__":
    unittest.main()

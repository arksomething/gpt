import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.indexed_shards import (
    IndexedCorpusSampler,
    IndexedShardReader,
    IndexedShardWriter,
    ResumableIndexedShardWriter,
    ShardFormatError,
    Source,
    content_hash,
)


TOKENIZER_HASH = "a" * 64
RECIPE_HASH = "b" * 64


class IndexedShardTests(unittest.TestCase):
    def build_corpus(self, root: Path, *, dtype: str = "uint16") -> Path:
        output = root / "corpus"
        with IndexedShardWriter(
            output,
            sources=[
                Source("web", "Filtered web", "ODC-By-1.0"),
                Source("books", "Open books", "public-domain"),
            ],
            tokenizer_sha256=TOKENIZER_HASH,
            recipe_sha256=RECIPE_HASH,
            token_dtype=dtype,
            target_shard_tokens=5,
            metadata={"run_id": "fixture"},
        ) as writer:
            writer.add_document(
                [1, 2, 3],
                source_id="web",
                content_sha256=content_hash("first"),
                quality_score=0.8,
                metadata={"url": "https://example.test/1"},
            )
            writer.add_document(
                [4, 5, 6, 7],
                source_id="books",
                content_sha256=content_hash("second"),
                metadata={"title": "Second"},
            )
        return output

    def test_round_trip_preserves_boundaries_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            corpus = self.build_corpus(Path(temporary))
            reader = IndexedShardReader(
                corpus,
                expected_tokenizer_sha256=TOKENIZER_HASH,
                expected_recipe_sha256=RECIPE_HASH,
            )

            self.assertEqual(reader.manifest["document_count"], 2)
            self.assertEqual(reader.manifest["token_count"], 7)
            self.assertEqual(len(reader.manifest["shards"]), 2)
            self.assertEqual(reader.documents[0].source_id, "web")
            self.assertEqual(reader.documents[1].metadata["title"], "Second")
            self.assertEqual(reader.read_tokens(0).tolist(), [1, 2, 3])
            self.assertEqual(reader.read_tokens(1).tolist(), [4, 5, 6, 7])

    def test_packed_sampler_is_deterministic_and_marks_document_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            corpus = self.build_corpus(Path(temporary))
            reader = IndexedShardReader(corpus)
            sampler = IndexedCorpusSampler(
                reader,
                source_weights={"web": 1.0},
            )

            first = sampler.sample_batch(
                batch_size=2,
                block_size=8,
                rng=np.random.default_rng(42),
            )
            second = sampler.sample_batch(
                batch_size=2,
                block_size=8,
                rng=np.random.default_rng(42),
            )

            np.testing.assert_array_equal(first.input_ids, second.input_ids)
            np.testing.assert_array_equal(first.labels, second.labels)
            np.testing.assert_array_equal(first.position_ids, second.position_ids)
            self.assertEqual(first.segments, second.segments)
            self.assertEqual(first.input_ids.shape, (2, 8))
            self.assertTrue(all(segment.source_id == "web" for segment in first.segments))
            for segment in first.segments:
                row = segment.batch_index
                start = segment.packed_start
                stop = start + segment.token_count
                self.assertEqual(first.labels[row, start], -100)
                np.testing.assert_array_equal(
                    first.position_ids[row, start:stop],
                    np.arange(segment.token_count),
                )

    def test_packed_sampler_rejects_invalid_source_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            reader = IndexedShardReader(self.build_corpus(Path(temporary)))
            with self.assertRaisesRegex(ShardFormatError, "unknown or empty"):
                IndexedCorpusSampler(reader, source_weights={"missing": 1.0})
            with self.assertRaisesRegex(ShardFormatError, "positive total"):
                IndexedCorpusSampler(
                    reader,
                    source_weights={"web": 0.0, "books": 0.0},
                )

    def test_refuses_fingerprint_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            corpus = self.build_corpus(Path(temporary))
            with self.assertRaisesRegex(
                ShardFormatError, "tokenizer fingerprint mismatch"
            ):
                IndexedShardReader(
                    corpus, expected_tokenizer_sha256="c" * 64
                )

    def test_detects_corrupt_token_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            corpus = self.build_corpus(Path(temporary))
            shard = corpus / "tokens-00000.bin"
            payload = bytearray(shard.read_bytes())
            payload[0] ^= 0xFF
            shard.write_bytes(payload)

            with self.assertRaisesRegex(ShardFormatError, "SHA-256 mismatch"):
                IndexedShardReader(corpus)

    def test_detects_manifest_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            corpus = self.build_corpus(Path(temporary))
            manifest_path = corpus / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["token_count"] += 1
            manifest_path.write_text(json.dumps(manifest))

            with self.assertRaisesRegex(
                ShardFormatError, "manifest fingerprint mismatch"
            ):
                IndexedShardReader(corpus)

    def test_detects_index_corruption_even_when_file_hash_checks_are_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            corpus = self.build_corpus(Path(temporary))
            index_path = corpus / "documents.jsonl"
            records = [
                json.loads(line) for line in index_path.read_text().splitlines()
            ]
            records[1]["token_start"] = 1
            index_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n"
            )

            with self.assertRaisesRegex(ShardFormatError, "non-contiguous"):
                IndexedShardReader(corpus, verify_hashes=False)

    def test_writer_rejects_unknown_source_and_out_of_range_token(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "corpus"
            writer = IndexedShardWriter(
                output,
                sources=[Source("web", "Web")],
                tokenizer_sha256=TOKENIZER_HASH,
                recipe_sha256=RECIPE_HASH,
                token_dtype="uint16",
            )
            with self.assertRaisesRegex(ShardFormatError, "unknown source"):
                writer.add_document(
                    [1],
                    source_id="missing",
                    content_sha256=content_hash("bad source"),
                )
            with self.assertRaisesRegex(ShardFormatError, "between 0 and 65535"):
                writer.add_document(
                    [65536],
                    source_id="web",
                    content_sha256=content_hash("bad token"),
                )
            with self.assertRaisesRegex(ShardFormatError, "integers only"):
                writer.add_document(
                    [1.5],
                    source_id="web",
                    content_sha256=content_hash("bad token type"),
                )
            writer.abort()
            self.assertFalse(output.exists())

    def test_writer_is_transactional_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "corpus"
            with self.assertRaises(RuntimeError):
                with IndexedShardWriter(
                    output,
                    sources=[Source("web", "Web")],
                    tokenizer_sha256=TOKENIZER_HASH,
                    recipe_sha256=RECIPE_HASH,
                ):
                    raise RuntimeError("stop")
            self.assertFalse(output.exists())

            self.build_corpus(root)
            with self.assertRaises(FileExistsError):
                IndexedShardWriter(
                    output,
                    sources=[Source("web", "Web")],
                    tokenizer_sha256=TOKENIZER_HASH,
                    recipe_sha256=RECIPE_HASH,
                )

    def test_resumable_writer_appends_without_copying_committed_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "corpus"
            sources = [Source("web", "Web")]
            writer = ResumableIndexedShardWriter(
                output,
                sources=sources,
                tokenizer_sha256=TOKENIZER_HASH,
                recipe_sha256=RECIPE_HASH,
                token_dtype="uint16",
                target_shard_tokens=4,
            )
            writer.add_document(
                [1, 2, 3],
                source_id="web",
                content_sha256=content_hash("one"),
            )
            writer.suspend()

            staging = output.with_name("corpus.staging")
            committed_shard = staging / "tokens-00000.bin"
            committed_hash = content_hash(committed_shard.read_bytes())
            resumed = ResumableIndexedShardWriter(
                output,
                sources=sources,
                tokenizer_sha256=TOKENIZER_HASH,
                recipe_sha256=RECIPE_HASH,
                token_dtype="uint16",
                target_shard_tokens=4,
                resume=True,
            )
            self.assertEqual(len(resumed.existing_documents), 1)
            resumed.add_document(
                [4, 5],
                source_id="web",
                content_sha256=content_hash("two"),
            )
            resumed.close()

            reader = IndexedShardReader(output)
            self.assertEqual(reader.read_tokens(0).tolist(), [1, 2, 3])
            self.assertEqual(reader.read_tokens(1).tolist(), [4, 5])
            self.assertEqual(
                content_hash((output / "tokens-00000.bin").read_bytes()),
                committed_hash,
            )
            self.assertFalse(staging.exists())

    def test_resumable_writer_discards_uncommitted_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "corpus"
            arguments = {
                "sources": [Source("web", "Web")],
                "tokenizer_sha256": TOKENIZER_HASH,
                "recipe_sha256": RECIPE_HASH,
                "token_dtype": "uint16",
            }
            writer = ResumableIndexedShardWriter(output, **arguments)
            writer.add_document(
                [1, 2],
                source_id="web",
                content_sha256=content_hash("committed"),
            )
            writer.checkpoint()
            writer.add_document(
                [9, 9],
                source_id="web",
                content_sha256=content_hash("tail"),
            )
            writer.discard_uncommitted()

            resumed = ResumableIndexedShardWriter(
                output,
                resume=True,
                **arguments,
            )
            self.assertEqual(len(resumed.existing_documents), 1)
            resumed.close()
            reader = IndexedShardReader(output)
            self.assertEqual(len(reader.documents), 1)
            self.assertEqual(reader.read_tokens(0).tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()

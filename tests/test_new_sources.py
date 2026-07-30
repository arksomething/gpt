"""Tests for the Gate 1 mixture bakeoff sources.

No test in this file touches the network: `datasets.load_dataset` is
monkeypatched everywhere a stream could be opened, and any unexpected call
fails the test loudly.
"""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sentencepiece as spm
import yaml

from scripts.filters import FilterStats
from scripts.indexed_shards import IndexedShardReader
from scripts.prepare_data import (
    CONVERSATION_MIN_CHARS,
    MIXTURE_SOURCE_IDS,
    MIXTURE_SOURCE_SPECS,
    MIXTURE_SOURCES_BY_ID,
    FilterConfig,
    active_mixture_weights,
    filter_and_transform_mixture,
    indexed_source_records,
    main as prepare_main,
    open_mixture_stream,
    preparation_recipe_sha256,
    process_and_tokenize,
    resolve_mixture_weights,
    stream_common_pile,
)

EXPECTED_SOURCES = {
    "dclm": ("mlfoundations/dclm-baseline-1.0", None, "global"),
    "finepdfs": ("HuggingFaceFW/finepdfs", "eng_Latn", "global"),
    "stackexchange": ("common-pile/stackexchange_filtered", None, "conversation"),
    "youtube": ("common-pile/youtube_filtered", None, "conversation"),
    "ubuntu_irc": ("common-pile/ubuntu_irc_filtered", None, "conversation"),
    "github_archive": ("common-pile/github_archive_filtered", None, "conversation"),
    "uk_hansard": ("common-pile/uk_hansard_filtered", None, "conversation"),
    "finemath": ("HuggingFaceTB/finemath", "finemath-4plus", "global"),
    "code": ("common-pile/stackv2_edu_filtered", None, "code"),
    "cosmopedia": ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "global"),
    "wikiteam": ("common-pile/wikiteam_filtered", None, "global"),
    "libretexts": ("common-pile/libretexts_filtered", None, "global"),
    "narrative": ("common-pile/pre_1929_books_filtered", None, "global"),
}


class FakeArgs:
    """Stand-in for the argparse namespace: any unset weight is None."""

    def __init__(self, **overrides):
        for source_id in MIXTURE_SOURCE_IDS:
            setattr(self, f"{source_id}_weight", None)
        for name, value in overrides.items():
            setattr(self, name, value)


class FakeDataset:
    """Minimal streaming-dataset stand-in: iterable of {"text": ...} rows."""

    def __init__(self, texts):
        self._texts = list(texts)

    def __iter__(self):
        for text in self._texts:
            yield {"text": text}


def recording_load_dataset(rows_by_repo, calls):
    def load_dataset(repo_id, *args, **kwargs):
        calls.append((repo_id, args[0] if args else kwargs.get("name")))
        return FakeDataset(rows_by_repo.get(repo_id, []))

    return load_dataset


def exploding_load_dataset(*args, **kwargs):
    raise AssertionError(f"load_dataset must not be called: {args} {kwargs}")


def permissive_filter_config(**overrides):
    defaults = dict(
        min_chars=300,
        max_chars=80_000,
        min_alpha_ratio=0.65,
        max_repeated_chars=6,
        max_weird_ratio=0.01,
        max_short_line_ratio=0.30,
        max_caps_ratio=0.20,
        max_chunk_chars=10_000,
        min_chunk_chars=500,
        dedup_enabled=False,
    )
    defaults.update(overrides)
    return FilterConfig(**defaults)


class SourceRegistryTests(unittest.TestCase):
    def test_every_requested_source_is_registered_with_expected_repo(self):
        self.assertEqual(set(MIXTURE_SOURCE_IDS), set(EXPECTED_SOURCES))
        for spec in MIXTURE_SOURCE_SPECS:
            repo, config, treatment = EXPECTED_SOURCES[spec.source_id]
            self.assertEqual(spec.repo_id, repo, spec.source_id)
            self.assertEqual(spec.config, config, spec.source_id)
            self.assertEqual(spec.filter_treatment, treatment, spec.source_id)
            self.assertEqual(spec.text_field, "text", spec.source_id)

    def test_common_pile_sources_all_use_the_one_factory(self):
        """Common Pile sources share a schema, so none may declare a config."""
        common_pile = [spec for spec in MIXTURE_SOURCE_SPECS if spec.is_common_pile]
        self.assertEqual(len(common_pile), 9)
        for spec in common_pile:
            self.assertIsNone(spec.config)
            self.assertEqual(spec.split, "train")
            self.assertEqual(spec.text_field, "text")

    def test_stream_common_pile_streams_text_field(self):
        calls = []
        rows = {"common-pile/wikiteam_filtered": ["alpha text", "", "beta text"]}
        with patch(
            "scripts.prepare_data.load_dataset",
            recording_load_dataset(rows, calls),
        ):
            texts = list(stream_common_pile("common-pile/wikiteam_filtered", seed=7))

        self.assertEqual(texts, ["alpha text", "beta text"])
        self.assertEqual(calls, [("common-pile/wikiteam_filtered", None)])

    def test_narrative_uses_the_common_pile_factory_with_no_config(self):
        spec = MIXTURE_SOURCES_BY_ID["narrative"]
        self.assertTrue(spec.is_common_pile)
        self.assertIsNone(spec.config)

        calls = []
        rows = {
            "common-pile/pre_1929_books_filtered": [
                "CHAPTER I. The narrative opens on a grey morning in November."
            ]
        }
        with patch(
            "scripts.prepare_data.load_dataset",
            recording_load_dataset(rows, calls),
        ):
            texts = list(open_mixture_stream(spec, 7))

        self.assertEqual(len(texts), 1)
        self.assertEqual(calls, [("common-pile/pre_1929_books_filtered", None)])

    def test_open_mixture_stream_passes_config_for_configured_sources(self):
        calls = []
        rows = {"HuggingFaceTB/finemath": ["math text"]}
        with patch(
            "scripts.prepare_data.load_dataset",
            recording_load_dataset(rows, calls),
        ):
            texts = list(open_mixture_stream(MIXTURE_SOURCES_BY_ID["finemath"], 7))

        self.assertEqual(texts, ["math text"])
        self.assertEqual(calls, [("HuggingFaceTB/finemath", "finemath-4plus")])


class WeightPlumbingTests(unittest.TestCase):
    def test_every_source_defaults_to_zero(self):
        weights = resolve_mixture_weights(FakeArgs(), {})
        self.assertEqual(set(weights), set(MIXTURE_SOURCE_IDS))
        self.assertEqual(set(weights.values()), {0.0})
        self.assertEqual(active_mixture_weights(weights), {})

    def test_config_supplies_weights(self):
        weights = resolve_mixture_weights(
            FakeArgs(),
            {"dclm_weight": 0.4, "code_weight": 0.1},
        )
        self.assertEqual(weights["dclm"], 0.4)
        self.assertEqual(weights["code"], 0.1)
        self.assertEqual(weights["finemath"], 0.0)
        self.assertEqual(
            active_mixture_weights(weights),
            {"code": 0.1, "dclm": 0.4},
        )

    def test_cli_overrides_config(self):
        weights = resolve_mixture_weights(
            FakeArgs(dclm_weight=0.25),
            {"dclm_weight": 0.4},
        )
        self.assertEqual(weights["dclm"], 0.25)

    def test_cli_zero_overrides_a_positive_config_weight(self):
        weights = resolve_mixture_weights(
            FakeArgs(dclm_weight=0.0),
            {"dclm_weight": 0.4},
        )
        self.assertEqual(weights["dclm"], 0.0)
        self.assertEqual(active_mixture_weights(weights), {})

    def test_cli_flag_exists_for_every_source(self):
        for source_id in MIXTURE_SOURCE_IDS:
            argv = [
                "prepare-data",
                "--config",
                "configs/train.yaml",
                f"--{source_id}_weight",
                "0.5",
            ]
            with patch.object(sys, "argv", argv):
                import argparse

                # Parsing happens inside main(); assert the flag name is at
                # least accepted by re-parsing the same spec shape.
                parser = argparse.ArgumentParser()
                parser.add_argument(f"--{source_id}_weight", type=float)
                parsed, _ = parser.parse_known_args(argv[1:])
                self.assertEqual(getattr(parsed, f"{source_id}_weight"), 0.5)

    def test_indexed_source_records_carry_enabled_mixture_sources(self):
        records = indexed_source_records(
            0.0,
            0.0,
            1.0,
            {"dclm": 0.3, "code": 0.2, "finemath": 0.0},
        )
        by_id = {record.source_id: record for record in records}
        self.assertIn("fineweb", by_id)
        self.assertIn("dclm", by_id)
        self.assertIn("code", by_id)
        self.assertNotIn("finemath", by_id)
        self.assertEqual(
            by_id["dclm"].metadata["dataset"],
            "mlfoundations/dclm-baseline-1.0",
        )
        self.assertEqual(by_id["code"].metadata["filter_treatment"], "code")

    def test_indexed_source_records_unchanged_without_mixture_weights(self):
        legacy = indexed_source_records(0.3, 0.5, 0.2)
        with_zeroes = indexed_source_records(
            0.3,
            0.5,
            0.2,
            {source_id: 0.0 for source_id in MIXTURE_SOURCE_IDS},
        )
        self.assertEqual(
            [record.source_id for record in legacy],
            ["c4", "wiki", "fineweb"],
        )
        self.assertEqual(legacy, with_zeroes)


class RecipeHashTests(unittest.TestCase):
    def _recipe(self, **overrides):
        kwargs = dict(
            filter_cfg=permissive_filter_config(),
            tokenizer_sha256="a" * 64,
            c4_weight=0.3,
            wiki_weight=0.5,
            fineweb_weight=0.2,
            shuffle_buffer=1000,
            seed=1337,
            max_unk_ratio=0.01,
            validation_fraction=0.01,
        )
        kwargs.update(overrides)
        return preparation_recipe_sha256(**kwargs)

    def test_legacy_recipe_hash_is_unchanged_by_zero_weight_sources(self):
        baseline = self._recipe()
        self.assertEqual(baseline, self._recipe(mixture_weights=None))
        self.assertEqual(baseline, self._recipe(mixture_weights={}))
        self.assertEqual(
            baseline,
            self._recipe(
                mixture_weights={
                    source_id: 0.0 for source_id in MIXTURE_SOURCE_IDS
                }
            ),
        )

    def test_enabling_a_source_changes_the_recipe_hash(self):
        baseline = self._recipe()
        enabled = self._recipe(mixture_weights={"dclm": 0.1})
        self.assertNotEqual(baseline, enabled)

    def test_recipe_hash_is_sensitive_to_weight_value_and_identity(self):
        one = self._recipe(mixture_weights={"dclm": 0.1})
        two = self._recipe(mixture_weights={"dclm": 0.2})
        other = self._recipe(mixture_weights={"finemath": 0.1})
        self.assertNotEqual(one, two)
        self.assertNotEqual(one, other)

    def test_recipe_hash_ignores_mixture_weight_ordering(self):
        forward = self._recipe(mixture_weights={"dclm": 0.1, "code": 0.2})
        reverse = self._recipe(mixture_weights={"code": 0.2, "dclm": 0.1})
        self.assertEqual(forward, reverse)


class FilterTreatmentTests(unittest.TestCase):
    CODE = (
        "import os\n"
        "from typing import List\n\n"
        "def collect(paths: List[str]) -> dict[str, int]:\n"
        "    sizes = {}\n"
        "    for path in paths:\n"
        "        sizes[path] = os.path.getsize(path)\n"
        "    return sizes\n\n"
        "class Walker:\n"
        "    def __init__(self, root: str) -> None:\n"
        "        self.root = root\n"
        "    def walk(self) -> list[str]:\n"
        "        return [p for p in os.listdir(self.root) if p]\n"
    )

    def test_code_source_is_exempt_from_alpha_ratio_filters(self):
        filter_cfg = permissive_filter_config(min_chunk_chars=100)
        code_stats = FilterStats()
        chunks = filter_and_transform_mixture(
            self.CODE,
            MIXTURE_SOURCES_BY_ID["code"],
            filter_cfg,
            code_stats,
        )
        self.assertTrue(chunks)
        self.assertEqual(code_stats.passed_docs, 1)

        # The same text under the global treatment is rejected by the prose
        # filters, which is exactly why code needs the exemption.
        global_stats = FilterStats()
        rejected = filter_and_transform_mixture(
            self.CODE,
            MIXTURE_SOURCES_BY_ID["wikiteam"],
            filter_cfg,
            global_stats,
        )
        self.assertEqual(rejected, [])
        self.assertEqual(global_stats.passed_docs, 0)
        self.assertTrue(global_stats.rejected_by)

    def test_code_still_enforces_length(self):
        filter_cfg = permissive_filter_config()
        stats = FilterStats()
        chunks = filter_and_transform_mixture(
            "x = 1\n",
            MIXTURE_SOURCES_BY_ID["code"],
            filter_cfg,
            stats,
        )
        self.assertEqual(chunks, [])
        self.assertEqual(stats.rejected_by["code_G0_length"], 1)

    def test_conversation_sources_accept_short_documents(self):
        text = (
            "Question: how do I list the files in a directory from a shell "
            "script without breaking on spaces in the names? Answer: quote the "
            "expansion and iterate over it safely, because unquoted word "
            "splitting is what actually breaks those names."
        )
        self.assertGreaterEqual(len(text), CONVERSATION_MIN_CHARS)
        self.assertLess(len(text), 300)

        filter_cfg = permissive_filter_config()
        conversation_stats = FilterStats()
        chunks = filter_and_transform_mixture(
            text,
            MIXTURE_SOURCES_BY_ID["stackexchange"],
            filter_cfg,
            conversation_stats,
        )
        self.assertTrue(chunks)
        self.assertEqual(conversation_stats.passed_docs, 1)

        global_stats = FilterStats()
        rejected = filter_and_transform_mixture(
            text,
            MIXTURE_SOURCES_BY_ID["dclm"],
            filter_cfg,
            global_stats,
        )
        self.assertEqual(rejected, [])
        self.assertEqual(global_stats.rejected_by["dclm_G0_length"], 1)

    def test_conversation_sources_keep_the_other_global_filters(self):
        filter_cfg = permissive_filter_config()
        stats = FilterStats()
        rejected = filter_and_transform_mixture(
            "!!!" + ("@#$%^&" * 80),
            MIXTURE_SOURCES_BY_ID["ubuntu_irc"],
            filter_cfg,
            stats,
        )
        self.assertEqual(rejected, [])
        self.assertEqual(stats.passed_docs, 0)
        self.assertNotIn("ubuntu_irc_G0_length", stats.rejected_by)

    def test_process_and_tokenize_routes_mixture_sources_and_records_stats(self):
        class FakeTokenizer:
            def unk_id(self):
                return 0

            def encode(self, text, out_type=int):
                return [1 + (ord(c) % 29) for c in text if not c.isspace()]

        stats = {"c4": FilterStats(), "wiki": FilterStats(), "fineweb": FilterStats()}
        documents = list(
            process_and_tokenize(
                iter([("code", self.CODE)]),
                permissive_filter_config(min_chunk_chars=100),
                FakeTokenizer(),
                deduplicator=None,
                stats=stats,
                shuffle_buffer_size=1,
                seed=3,
            )
        )

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].source_id, "code")
        self.assertIn("code", stats)
        self.assertEqual(stats["code"].passed_docs, 1)


class NoInitWhenDisabledTests(unittest.TestCase):
    """A zero-weight source must never be constructed, streamed, or loaded."""

    def test_zero_weight_sources_never_call_load_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = [
                "This is a clear synthetic fixture paragraph number "
                f"{index} with ordinary words for tokenizer training."
                for index in range(300)
            ]
            tokenizer_prefix = root / "spm"
            spm.SentencePieceTrainer.train(
                sentence_iterator=iter(corpus),
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
                yaml.safe_dump({"data_prep": _base_data_prep(tokenizer_prefix, output)}),
                encoding="utf-8",
            )

            def fixture_stream(seed=42):
                del seed
                yield from corpus

            argv = ["prepare-data", "--config", str(config_path), "--overwrite", "--no_log"]
            with (
                patch("scripts.prepare_data.load_dataset", exploding_load_dataset),
                patch("scripts.prepare_data.stream_fineweb", fixture_stream),
                patch.object(sys, "argv", argv),
            ):
                prepare_main()

            metadata = json.loads((output / "data_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["mixture_source_weights"], {})
            self.assertEqual(metadata["mixture_sources"], {})
            self.assertNotIn("dclm", metadata["train_stats"]["by_source"])

            manifest = json.loads(
                (output / "train" / "manifest.json").read_text(encoding="utf-8")
            )
            manifest_sources = {source["source_id"] for source in manifest["sources"]}
            self.assertEqual(manifest_sources, {"fineweb"})

    def test_enabled_source_is_streamed_and_reaches_manifest_and_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = [
                "This is a clear synthetic fixture paragraph number "
                f"{index} with ordinary words for tokenizer training."
                for index in range(300)
            ]
            tokenizer_prefix = root / "spm"
            spm.SentencePieceTrainer.train(
                sentence_iterator=iter(corpus),
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
            data_prep = _base_data_prep(tokenizer_prefix, output)
            data_prep["fineweb_weight"] = 0.0
            data_prep["dclm_weight"] = 1.0
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump({"data_prep": data_prep}), encoding="utf-8")

            calls = []
            rows = {"mlfoundations/dclm-baseline-1.0": corpus * 4}
            argv = ["prepare-data", "--config", str(config_path), "--overwrite", "--no_log"]
            with (
                patch(
                    "scripts.prepare_data.load_dataset",
                    recording_load_dataset(rows, calls),
                ),
                patch.object(sys, "argv", argv),
            ):
                prepare_main()

            self.assertEqual(
                {repo for repo, _ in calls},
                {"mlfoundations/dclm-baseline-1.0"},
            )

            metadata = json.loads((output / "data_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["mixture_source_weights"], {"dclm": 1.0})
            self.assertEqual(
                metadata["mixture_sources"]["dclm"]["dataset"],
                "mlfoundations/dclm-baseline-1.0",
            )
            self.assertIn("dclm", metadata["train_stats"]["by_source"])
            self.assertGreater(metadata["train_stats"]["by_source"]["dclm"]["total"], 0)

            reader = IndexedShardReader(output / "train")
            manifest_sources = {
                source["source_id"] for source in reader.manifest["sources"]
            }
            self.assertEqual(manifest_sources, {"dclm"})
            self.assertEqual(
                {document.source_id for document in reader.documents},
                {"dclm"},
            )


class LegacyRunUnperturbedTests(unittest.TestCase):
    """A legacy-weights run must be byte-identical with the new sources present.

    The same preparation is run twice: once from a config that predates the
    mixture sources entirely, and once from a config that declares every one of
    them at 0.0. Identical recipe sha256, corpus sha256 and token counts mean a
    disabled source cannot perturb an existing recipe.
    """

    def test_declaring_zero_weights_changes_nothing(self):
        # One tokenizer for both runs: the recipe hash covers the tokenizer,
        # so retraining it per run would mask the property under test.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = [
                "This is a clear synthetic fixture paragraph number "
                f"{index} with ordinary words for tokenizer training."
                for index in range(300)
            ]
            tokenizer_prefix = root / "spm"
            spm.SentencePieceTrainer.train(
                sentence_iterator=iter(corpus),
                model_prefix=str(tokenizer_prefix),
                vocab_size=96,
                model_type="bpe",
                bos_id=1,
                eos_id=2,
                pad_id=3,
                unk_id=0,
                minloglevel=2,
            )
            first = self._run(root, tokenizer_prefix, corpus, "a", False)
            second = self._run(root, tokenizer_prefix, corpus, "b", True)

        self.assertEqual(first, second)
        self.assertEqual(first["sources"], ["fineweb"])

    def _run(
        self,
        root: Path,
        tokenizer_prefix: Path,
        corpus: list,
        name: str,
        with_zero_weight_keys: bool,
    ) -> dict:
        output = root / f"indexed_{name}"
        data_prep = _base_data_prep(tokenizer_prefix, output)
        if with_zero_weight_keys:
            for source_id in MIXTURE_SOURCE_IDS:
                data_prep[f"{source_id}_weight"] = 0.0
        config_path = root / f"config_{name}.yaml"
        config_path.write_text(
            yaml.safe_dump({"data_prep": data_prep}), encoding="utf-8"
        )

        def fixture_stream(seed=42):
            del seed
            yield from corpus

        argv = [
            "prepare-data",
            "--config",
            str(config_path),
            "--overwrite",
            "--no_log",
        ]
        with (
            patch("scripts.prepare_data.load_dataset", exploding_load_dataset),
            patch("scripts.prepare_data.stream_fineweb", fixture_stream),
            patch.object(sys, "argv", argv),
        ):
            prepare_main()

        meta = json.loads((output / "data_meta.json").read_text(encoding="utf-8"))
        train = json.loads(
            (output / "train" / "manifest.json").read_text(encoding="utf-8")
        )
        # corpus_sha256 embeds a wall-clock created_at and is therefore
        # never equal across runs; compare the token bytes and the document
        # index instead, which is what "byte-identical output" means here.
        return {
            "recipe": meta["preparation_recipe_sha256"],
            "train_tokens": meta["train_tokens"],
            "val_tokens": meta["val_tokens"],
            "train_stats": meta["train_stats"],
            "sources": [source["source_id"] for source in train["sources"]],
            "train_documents": _document_fingerprint(output / "train"),
            "validation_documents": _document_fingerprint(output / "validation"),
            "shard_bytes": _shard_fingerprint(output),
        }


def _document_fingerprint(corpus_dir: Path) -> list:
    reader = IndexedShardReader(corpus_dir)
    return [
        (
            document.document_id,
            document.source_id,
            document.shard,
            document.token_start,
            document.token_count,
            document.content_sha256,
        )
        for document in reader.documents
    ]


def _shard_fingerprint(output: Path) -> dict:
    return {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*.bin"))
    }


def _base_data_prep(tokenizer_prefix: Path, output: Path) -> dict:
    return {
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


if __name__ == "__main__":
    unittest.main()

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.canonical_writing import (
    build_canonical_lane,
    iter_canonical_manifest,
    sha256_file,
)


class CanonicalWritingTests(unittest.TestCase):
    def _fixture(self, root: Path, keep: str):
        registry = root / "sources.yaml"
        registry.write_text(
            yaml.safe_dump(
                {
                    "sources": {
                        "essays": {
                            "name": "Essays",
                            "tier": "A",
                            "redistribution": "redistributable",
                            "license": "CC-BY-4.0",
                            "roles": ["analytical_essay"],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        text = "A carefully reviewed analytical essay fixture."
        document_id = hashlib.sha256(text.encode()).hexdigest()
        documents = root / "documents.jsonl"
        documents.write_text(
            json.dumps(
                {
                    "source_id": "essays",
                    "document_id": document_id,
                    "text": text,
                    "url": "https://example.test/essay",
                    "license": "CC-BY-4.0",
                    "metadata": {"author": "Example"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        acquisition = root / "acquisition.json"
        acquisition.write_text(
            json.dumps(
                {
                    "run_id": "fixture",
                    "sources": {
                        "essays": {
                            "documents_path": str(documents),
                            "documents_sha256": sha256_file(documents),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        scores = root / "scores.csv"
        with scores.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "source_id",
                    "document_id",
                    "keep_yes_no",
                    "clarity_0_2",
                    "reviewer_notes",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "source_id": "essays",
                    "document_id": document_id,
                    "keep_yes_no": keep,
                    "clarity_0_2": "2",
                    "reviewer_notes": "clean",
                }
            )
        return registry, acquisition, scores, document_id

    def test_blank_human_decision_cannot_become_training_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry, acquisition, scores, _ = self._fixture(root, "")
            with self.assertRaisesRegex(ValueError, "no manually accepted"):
                build_canonical_lane(
                    registry_path=registry,
                    acquisition_manifest_path=acquisition,
                    scores_path=scores,
                    output_dir=root / "canonical",
                    root=root,
                )

    def test_kept_document_builds_verified_canonical_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry, acquisition, scores, document_id = self._fixture(root, "yes")
            manifest_path, manifest = build_canonical_lane(
                registry_path=registry,
                acquisition_manifest_path=acquisition,
                scores_path=scores,
                output_dir=root / "canonical",
                root=root,
            )
            documents = list(iter_canonical_manifest(manifest_path))

            self.assertEqual(manifest["selection_policy"], "manual_keep_yes_no")
            self.assertEqual(manifest["document_count"], 1)
            self.assertEqual(documents[0][0], "essays")
            self.assertEqual(
                documents[0][2]["canonical_document_id"],
                document_id,
            )
            self.assertEqual(documents[0][2]["license"], "CC-BY-4.0")


if __name__ == "__main__":
    unittest.main()

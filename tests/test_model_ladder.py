import unittest
from pathlib import Path

from scripts.validate_model_ladder import DEFAULT_CONFIGS, validate_config


class ModelLadderTest(unittest.TestCase):
    def test_ladder_configs_construct_on_meta_and_match_targets(self):
        for config_path in DEFAULT_CONFIGS:
            with self.subTest(config_path=Path(config_path).name):
                result = validate_config(config_path)

                self.assertEqual(result["parameter_device"], "meta")
                self.assertTrue(result["within_tolerance"])
                self.assertTrue(result["tie_word_embeddings"])
                self.assertGreater(
                    result["total_parameters"],
                    result["non_embedding_parameters"],
                )
                self.assertEqual(
                    result["num_attention_heads"]
                    % result["num_key_value_heads"],
                    0,
                )
                self.assertEqual(
                    result["hidden_size"] % result["num_attention_heads"],
                    0,
                )
                self.assertEqual(result["max_position_embeddings"], 4096)

    def test_ladder_parameter_counts_increase_monotonically(self):
        results = [validate_config(path) for path in DEFAULT_CONFIGS]
        totals = [result["total_parameters"] for result in results]

        self.assertEqual(totals, sorted(totals))
        self.assertEqual(len(set(totals)), len(totals))


if __name__ == "__main__":
    unittest.main()

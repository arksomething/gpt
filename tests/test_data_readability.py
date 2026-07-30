"""Gate 0: every token .bin must decode to readable text under the project
tokenizer.

Regression guard for the v3 val.bin incident: a --resume kept tokens encoded
by a different tokenizer while metadata claimed otherwise, and nothing noticed
until manual review. The fixture bin preserves 60k tokens of that corrupt
stream as a permanent negative control.
"""

import glob
import os

import pytest

from scripts.data_quality import (
    READABILITY_MEAN_THRESHOLD,
    bin_readability,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER = os.path.join(REPO, "tokenizer", "spm.model")
CORRUPT_FIXTURE = os.path.join(
    REPO, "tests", "fixtures", "cross_tokenizer_corrupt.bin"
)


def data_bins():
    """All token bins under data/, excluding quarantine and fixtures."""
    paths = glob.glob(os.path.join(REPO, "data", "**", "*.bin"), recursive=True)
    return sorted(
        p
        for p in paths
        if "quarantine" not in p and os.path.getsize(p) > 0
    )


@pytest.mark.skipif(not os.path.exists(TOKENIZER), reason="tokenizer missing")
class TestBinReadability:
    def test_known_clean_bin_passes(self):
        path = os.path.join(REPO, "data", "v3", "val_fresh.bin")
        if not os.path.exists(path):
            pytest.skip("val_fresh.bin not present")
        result = bin_readability(path, TOKENIZER)
        assert result["passed"], result

    def test_cross_tokenizer_corruption_is_detected(self):
        result = bin_readability(CORRUPT_FIXTURE, TOKENIZER)
        assert not result["passed"], (
            "corrupt fixture scored as readable; detector is broken: "
            f"{result}"
        )
        assert result["mean"] < READABILITY_MEAN_THRESHOLD

    @pytest.mark.parametrize(
        "bin_path", data_bins(), ids=lambda p: os.path.relpath(p, REPO)
    )
    def test_all_data_bins_readable(self, bin_path):
        result = bin_readability(bin_path, TOKENIZER)
        assert result["passed"], (
            f"{os.path.relpath(bin_path, REPO)} failed readability "
            f"(mean={result['mean']:.3f}, min={result['min']:.3f}); "
            "wrong or changed tokenizer?"
        )

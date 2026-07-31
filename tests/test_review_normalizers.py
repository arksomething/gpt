"""The IRC markup normalizer must rescue clean text without excusing garbage."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sentencepiece as spm  # noqa: E402

from scripts.data_quality import (  # noqa: E402
    READABILITY_MEAN_THRESHOLD,
    text_readability,
    vocab_word_set,
)
from scripts.review_corpus import SCORING_NORMALIZERS, _strip_irc_markup  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER = os.path.join(REPO, "tokenizer", "spm.model")

CLEAN_IRC = " ".join(
    [
        "[14:17] <dholbach> hey there, I have a question about the release "
        "schedule for the next version.",
        "[14:18] <mitya57> the freeze happens next week, so please upload any "
        "remaining changes before then.",
        "[14:19] <dholbach> thanks, that is very helpful. I will prepare the "
        "package tonight and send it.",
        "[14:20] * someone waves at the channel and asks about documentation "
        "for the installer settings.",
    ]
)


class StripIrcMarkupTests(unittest.TestCase):
    def test_removes_timestamps_nicknames_and_actions(self):
        out = _strip_irc_markup(CLEAN_IRC)
        self.assertNotIn("[14:17]", out)
        self.assertNotIn("<dholbach>", out)
        self.assertIn("release", out)

    def test_registered_for_ubuntu_irc(self):
        self.assertIn("ubuntu_irc", SCORING_NORMALIZERS)


@unittest.skipUnless(os.path.exists(TOKENIZER), "tokenizer not present")
class NormalizerSharpnessTests(unittest.TestCase):
    """Stripping markup must not turn the check into a rubber stamp."""

    @classmethod
    def setUpClass(cls):
        sp = spm.SentencePieceProcessor(model_file=TOKENIZER)
        cls.words = vocab_word_set(sp)

    def test_clean_irc_passes_after_normalization(self):
        score = text_readability(_strip_irc_markup(CLEAN_IRC), self.words)
        self.assertIsNotNone(score)
        self.assertGreaterEqual(score, READABILITY_MEAN_THRESHOLD)

    def test_scrambled_irc_still_fails_after_normalization(self):
        # Same IRC scaffolding, non-word payload: this is what cross-tokenizer
        # corruption looks like, and it must not survive normalization.
        garbage = " ".join(
            f"[14:{i:02d}] <nick{i}> zxqv jkkq vbnm qwrt plfh mnbv xzcv"
            for i in range(20)
        )
        score = text_readability(_strip_irc_markup(garbage), self.words)
        if score is not None:
            self.assertLess(score, READABILITY_MEAN_THRESHOLD)


if __name__ == "__main__":
    unittest.main()

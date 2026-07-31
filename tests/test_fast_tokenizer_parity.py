"""The fast tokenizer must encode exactly like the SentencePiece model, or refuse.

Guards a silent, version-dependent failure: LlamaTokenizerFast(vocab_file=...)
builds a tokenizer that encodes the first content token differently, and
transformers 5.x collapses repeated leading whitespace. Neither raises on its
own -- you just train on one tokenization and serve on another.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAST_DIR = os.path.join(REPO, "tokenizer", "fast")
SPM = os.path.join(REPO, "tokenizer", "spm.model")


@unittest.skipUnless(os.path.isdir(FAST_DIR), "fast tokenizer not built")
@unittest.skipUnless(os.path.exists(SPM), "spm model not present")
class FastTokenizerParityTests(unittest.TestCase):
    def test_loader_either_verifies_or_raises(self):
        """It must never hand back a silently-wrong tokenizer.

        Under transformers 4.49 this loads and matches. Under 5.x it raises,
        because leading whitespace is handled differently. Both are acceptable;
        returning a mis-encoding tokenizer is not.
        """
        import sentencepiece as spm

        from scripts.build_fast_tokenizer import (
            PARITY_CANARIES,
            TokenizerParityError,
            load_verified_fast_tokenizer,
        )

        sp = spm.SentencePieceProcessor(model_file=SPM)
        try:
            tok = load_verified_fast_tokenizer(FAST_DIR, SPM)
        except TokenizerParityError as e:
            self.assertIn("SentencePiece", str(e))
            return

        for text in PARITY_CANARIES:
            with self.subTest(text=text):
                self.assertEqual(
                    tok(text, add_special_tokens=False)["input_ids"],
                    sp.encode(text),
                )

    def test_special_token_ids_match_spm(self):
        import sentencepiece as spm
        from transformers import AutoTokenizer

        sp = spm.SentencePieceProcessor(model_file=SPM)
        tok = AutoTokenizer.from_pretrained(FAST_DIR)
        self.assertEqual(tok.bos_token_id, sp.bos_id())
        self.assertEqual(tok.eos_token_id, sp.eos_id())
        self.assertEqual(tok.vocab_size, sp.get_piece_size())


if __name__ == "__main__":
    unittest.main()

"""Build a fast tokenizer whose encoding is identical to the raw SentencePiece.

Every premade RL and eval stack (TRL, vLLM, verifiers) wants a fast tokenizer.
Ours did not have one that was correct, and the failure was silent: constructing

    LlamaTokenizerFast(vocab_file="tokenizer/spm.model")

produces a tokenizer that encodes the *first content token* of nearly every
string differently -- "The" becomes 1839 instead of 350 -- because the leading
SentencePiece space is handled differently. Nothing raises; you simply train on
one tokenization and serve on another. That is the same class of bug as the v3
val.bin and the transformers 5.0 breakage, both of which cost real time.

The correct path is convert_slow_tokenizer.LlamaConverter, which carries over
the Prepend/Replace normalizer that the direct constructor omits. Parity is not
assumed here: it is measured against raw SentencePiece over a corpus sample, and
this script exits non-zero if a single document disagrees.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

EDGE_CASES = [
    "",
    " ",
    "  leading spaces",
    "trailing spaces  ",
    "The capital of France is Paris.",
    "def f(x):\n    return x * 2",
    "<|user|> hi <|assistant|> hello <|end|>",
    "emoji 🙂 and accents café naïve",
    "[16:35] <nick> irc style line",
    "Multi\nline\ntext\twith\ttabs",
    "1234567890 !@#$%^&*()",
    "a" * 500,
]


# Canaries chosen to cover the two ways this has actually broken: the leading
# SentencePiece space (which the direct LlamaTokenizerFast constructor gets
# wrong) and repeated leading whitespace (which transformers 5.x collapses).
PARITY_CANARIES = [
    "The capital of France is Paris.",
    " Once upon a time in a small town",
    "  leading spaces",
    "def f(x):\n    return x * 2",
]


class TokenizerParityError(RuntimeError):
    """The fast tokenizer does not encode like the SentencePiece model."""


def load_verified_fast_tokenizer(
    fast_dir: str = "tokenizer/fast", spm_path: str = "tokenizer/spm.model"
):
    """Load the fast tokenizer, refusing to return one that mis-encodes.

    This exists because the failure mode is silent and version-dependent. The
    tokenizer written here is exact under transformers 4.49 (412/412 including
    real corpus text) but diverges under 5.x on any string beginning with
    whitespace -- 24 of 400 real documents, about 6%. Training on one
    tokenization and serving on another is the v3 val.bin failure wearing a
    different hat, so this raises rather than letting it through.
    """
    import sentencepiece as spm
    from transformers import AutoTokenizer

    sp = spm.SentencePieceProcessor(
        model_file=spm_path if os.path.isabs(spm_path) else os.path.join(REPO_ROOT, spm_path)
    )
    tok = AutoTokenizer.from_pretrained(
        fast_dir if os.path.isabs(fast_dir) else os.path.join(REPO_ROOT, fast_dir)
    )
    for text in PARITY_CANARIES:
        expected = sp.encode(text)
        actual = tok(text, add_special_tokens=False)["input_ids"]
        if expected != actual:
            import transformers

            raise TokenizerParityError(
                f"fast tokenizer disagrees with SentencePiece on {text!r} under "
                f"transformers {transformers.__version__}: "
                f"{actual[:8]} != {expected[:8]}. "
                "Use transformers==4.49.0 for anything that tokenizes text."
            )
    return tok


def corpus_samples(corpus_dir: str, n: int, seed: int) -> List[str]:
    """Decode real documents so parity is checked on the actual distribution."""
    import sentencepiece as spm

    from scripts.indexed_shards import IndexedShardReader

    sp = spm.SentencePieceProcessor(model_file=os.path.join(REPO_ROOT, "tokenizer", "spm.model"))
    reader = IndexedShardReader(corpus_dir, verify_hashes=False)
    docs = list(reader.iter_documents())
    random.Random(seed).shuffle(docs)
    out = []
    for doc in docs[:n]:
        text = sp.decode(reader.read_tokens(doc).tolist())
        # Long documents mostly re-test the same merges; a prefix is enough and
        # keeps the check fast enough to run in CI.
        out.append(text[:2000])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spm", default="tokenizer/spm.model")
    ap.add_argument("--out_dir", default="tokenizer/fast")
    ap.add_argument("--corpus_dir", default=None, help="Indexed corpus for real-text parity")
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    import sentencepiece as spm
    from transformers import LlamaTokenizer, LlamaTokenizerFast
    from transformers.convert_slow_tokenizer import LlamaConverter

    spm_path = os.path.join(REPO_ROOT, args.spm)
    sp = spm.SentencePieceProcessor(model_file=spm_path)
    slow = LlamaTokenizer(vocab_file=spm_path, legacy=True)

    backend = LlamaConverter(slow).converted()
    fast = LlamaTokenizerFast(
        tokenizer_object=backend,
        bos_token=slow.bos_token,
        eos_token=slow.eos_token,
        unk_token=slow.unk_token,
        legacy=True,
    )

    texts = list(EDGE_CASES)
    if args.corpus_dir:
        texts += corpus_samples(args.corpus_dir, args.samples, args.seed)
    print(f"Checking parity on {len(texts)} texts...")

    failures = 0
    for text in texts:
        expected = sp.encode(text)
        actual = fast(text, add_special_tokens=False)["input_ids"]
        if expected != actual:
            failures += 1
            if failures <= 3:
                print(f"  MISMATCH {text[:60]!r}")
                print(f"    spm : {expected[:12]}")
                print(f"    fast: {actual[:12]}")

    if failures:
        print(f"\nFAIL: {failures}/{len(texts)} texts disagree with SentencePiece.")
        print("Refusing to write a tokenizer that does not match training.")
        sys.exit(1)

    out_dir = os.path.join(REPO_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    fast.save_pretrained(out_dir)
    print(f"\nPASS: {len(texts)}/{len(texts)} exact matches.")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()

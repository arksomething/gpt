"""Per-source decoded-sample review for an indexed corpus.

The v3 corruption was invisible in every metadata field and obvious the moment
someone decoded the tokens, so nothing trains until a corpus has been read back
through its own tokenizer. This scores each source separately: a single bad
source is diluted below the threshold in a whole-corpus average, but it is
exactly what a mixture experiment would then misattribute to the mixture.

Exit code is non-zero if any source falls below the readability threshold.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from collections import defaultdict
from typing import Callable, Dict, List

import sentencepiece as spm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_quality import (  # noqa: E402
    READABILITY_MEAN_THRESHOLD,
    text_readability,
    vocab_word_set,
)
from scripts.indexed_shards import IndexedShardReader  # noqa: E402


# Some sources carry structural scaffolding that is not prose and never will
# be: IRC timestamps and nicknames are the clear case. The readability score
# counts dictionary words, so that scaffolding drags the score down even when
# every character is exactly right -- ubuntu_irc measures 0.618 raw and 0.746
# once the markup is removed.
#
# The scaffolding is stripped before scoring rather than the threshold being
# lowered for that source. Lowering a threshold blunts the check; removing
# known-good markup leaves it sharp, because tokens that are genuinely
# scrambled still score terribly after stripping.
_IRC_TIMESTAMP = re.compile(r"\[\d{2}:\d{2}(?::\d{2})?\]")
_IRC_NICK = re.compile(r"<[^>\s]{1,20}>")
_IRC_ACTION = re.compile(r"^\s*\*\s+\S+", re.MULTILINE)


def _strip_irc_markup(text: str) -> str:
    return _IRC_ACTION.sub(" ", _IRC_NICK.sub(" ", _IRC_TIMESTAMP.sub(" ", text)))


SCORING_NORMALIZERS: Dict[str, Callable[[str], str]] = {
    "ubuntu_irc": _strip_irc_markup,
}


def review(
    corpus_dir: str,
    tokenizer_model: str,
    per_source: int,
    excerpt_chars: int,
    seed: int,
    verify_hashes: bool,
) -> int:
    sp = spm.SentencePieceProcessor(model_file=tokenizer_model)
    words = vocab_word_set(sp)
    reader = IndexedShardReader(corpus_dir, verify_hashes=verify_hashes)

    by_source: Dict[str, List] = defaultdict(list)
    for doc in reader.iter_documents():
        by_source[doc.source_id].append(doc)

    if not by_source:
        print(f"FAIL: {corpus_dir} contains no documents")
        return 1

    rng = random.Random(seed)
    failures: List[str] = []
    print(f"Corpus: {corpus_dir}")
    print(f"Tokenizer: {tokenizer_model}")
    print(f"Threshold: mean readability >= {READABILITY_MEAN_THRESHOLD}\n")

    for source_id in sorted(by_source):
        docs = by_source[source_id]
        total_tokens = sum(d.token_count for d in docs)
        sample = rng.sample(docs, min(per_source, len(docs)))

        normalizer = SCORING_NORMALIZERS.get(source_id)
        scores: List[float] = []
        excerpt = ""
        for doc in sample:
            text = sp.decode(reader.read_tokens(doc).tolist())
            scored_text = normalizer(text) if normalizer else text
            score = text_readability(scored_text, words)
            if score is not None:
                scores.append(score)
            # The excerpt shows the real decoded text, never the normalized
            # form -- a human reviewing this must see what is actually stored.
            if not excerpt:
                excerpt = " ".join(text.split())[:excerpt_chars]

        header = (
            f"=== {source_id}  "
            f"{len(docs):,} docs / {total_tokens:,} tokens / "
            f"{len(scores)} scored"
            f"{' / markup-normalized' if normalizer else ''} ==="
        )
        if not scores:
            print(header)
            print("  FAIL: no scorable samples\n")
            failures.append(f"{source_id} (no scorable samples)")
            continue

        mean = sum(scores) / len(scores)
        worst = min(scores)
        verdict = "ok" if mean >= READABILITY_MEAN_THRESHOLD else "FAIL"
        if verdict == "FAIL":
            failures.append(f"{source_id} (mean {mean:.3f})")
        print(header)
        print(f"  readability mean={mean:.3f} min={worst:.3f}  [{verdict}]")
        print(f"  sample: {excerpt}\n")

    print("=" * 60)
    if failures:
        print(f"FAILED {len(failures)}/{len(by_source)} sources: {', '.join(failures)}")
        return 1
    print(f"All {len(by_source)} sources readable. Corpus cleared for training.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus_dir", help="Indexed corpus directory (train or validation)")
    ap.add_argument("--tokenizer_model", default="tokenizer/spm.model")
    ap.add_argument("--per_source", type=int, default=40)
    ap.add_argument("--excerpt_chars", type=int, default=220)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no_verify_hashes", action="store_true")
    args = ap.parse_args()

    sys.exit(
        review(
            args.corpus_dir,
            args.tokenizer_model,
            args.per_source,
            args.excerpt_chars,
            args.seed,
            not args.no_verify_hashes,
        )
    )


if __name__ == "__main__":
    main()

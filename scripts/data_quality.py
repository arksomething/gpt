#!/usr/bin/env python3
"""Readability checks for tokenized .bin files.

Decoded samples from a healthy corpus consist mostly of real words drawn from
the tokenizer's own whole-word vocabulary. Token streams encoded with a
different tokenizer (or otherwise scrambled) decode into subword salad whose
5+ letter "words" are mostly fabricated, so their vocabulary hit rate drops
far below that of clean text. Calibrated on data/v3: clean windows score
0.59-0.87, cross-tokenizer windows score 0.33-0.52.
"""

import re
from typing import Optional

import numpy as np
import sentencepiece as spm

# Window mean below this is corrupt; healthy English prose sits well above.
# Individual windows overlap between clean and corrupt corpora (name-dense
# clean pages score as low as ~0.52, corrupt windows as high as ~0.52), so
# only the mean over many windows is a reliable criterion.
READABILITY_MEAN_THRESHOLD = 0.65

_WORD_RE = re.compile(r"[A-Za-z]{5,}")
_SPM_SPACE = "▁"


def vocab_word_set(sp: spm.SentencePieceProcessor, min_len: int = 5) -> set:
    """Whole-word alphabetic pieces from the tokenizer vocabulary."""
    words = set()
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        if piece.startswith(_SPM_SPACE):
            word = piece[1:]
            if word.isalpha() and len(word) >= min_len:
                words.add(word.lower())
    return words


def text_readability(text: str, words: set, min_words: int = 15) -> Optional[float]:
    """Fraction of 5+ letter words present in the tokenizer vocabulary.

    Returns None when the sample has too few words to judge (short or
    non-English windows should not produce confident scores).
    """
    tokens = _WORD_RE.findall(text)
    if len(tokens) < min_words:
        return None
    return sum(1 for t in tokens if t.lower() in words) / len(tokens)


def bin_readability(
    bin_path: str,
    tokenizer_model: str,
    n_windows: int = 25,
    window_tokens: int = 3000,
    seed: int = 0,
    token_dtype: str = "uint16",
) -> dict:
    """Score random decoded windows of a token .bin file.

    Returns {"mean": float, "min": float, "windows": int, "passed": bool}.
    Raises ValueError if the file is too small to produce any scorable window.
    """
    sp = spm.SentencePieceProcessor(model_file=tokenizer_model)
    words = vocab_word_set(sp)
    arr = np.memmap(bin_path, dtype=np.dtype(token_dtype), mode="r")
    if len(arr) < window_tokens:
        window_tokens = max(200, len(arr))
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_windows):
        lo = int(rng.integers(0, max(1, len(arr) - window_tokens)))
        window = arr[lo : lo + window_tokens].tolist()
        score = text_readability(sp.decode(window), words)
        if score is not None:
            scores.append(score)
    if not scores:
        raise ValueError(
            f"{bin_path}: no scorable windows (file too small or non-text)"
        )
    mean = float(np.mean(scores))
    return {
        "mean": mean,
        "min": float(np.min(scores)),
        "windows": len(scores),
        "passed": mean >= READABILITY_MEAN_THRESHOLD,
    }

#!/usr/bin/env python3
"""
Text filtering functions for data preparation.

Implements filtering rules:
- Global (G0-G7): Applied to all sources
- C4-specific (C0-C6): Stricter filters for web content
- Wikipedia-specific (W0-W4): Strip non-prose, keep paragraphs
- Gutenberg-specific (GUT0-GUT5): Strip boilerplate, normalize
"""

import math
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

# Common punctuation for ratio calculations
COMMON_PUNCT = set(".,!?;:'\"-()[]{}…")
# Punctuation that indicates spam when excessive
SPAM_PUNCT = set(".,;:-/|=_\\\"'")


# =============================================================================
# Global Filters (G0-G7)
# =============================================================================


def filter_by_length(text: str, min_chars: int = 300, max_chars: int = 80000) -> bool:
    """G0: Hard size cutoffs. Returns True if text passes."""
    length = len(text)
    return min_chars <= length <= max_chars


def compute_alpha_ratio(text: str) -> float:
    """Compute ratio of alphabetic characters to total characters."""
    if not text:
        return 0.0
    alpha_count = sum(1 for c in text if c.isalpha())
    return alpha_count / len(text)


def filter_by_alpha_ratio(text: str, min_ratio: float = 0.65) -> bool:
    """G1: Character composition filter. Returns True if text passes."""
    return compute_alpha_ratio(text) >= min_ratio


def filter_repeated_chars(text: str, max_repeat: int = 6) -> bool:
    """G2: Drop if any character repeats >= max_repeat times in a row."""
    if max_repeat <= 0:
        return True
    # Build pattern to match any char repeated max_repeat+ times
    pattern = re.compile(r"(.)\1{" + str(max_repeat - 1) + r",}")
    return pattern.search(text) is None


def compute_weird_ratio(text: str) -> float:
    """
    Compute ratio of 'weird' characters.
    Weird = not in [letters, digits, common punctuation, whitespace]
    """
    if not text:
        return 0.0
    allowed = set(string.ascii_letters + string.digits + string.punctuation + string.whitespace)
    weird_count = sum(1 for c in text if c not in allowed)
    return weird_count / len(text)


def filter_weird_symbols(text: str, max_ratio: float = 0.01) -> bool:
    """G3: Drop if weird symbol density > max_ratio."""
    return compute_weird_ratio(text) <= max_ratio


def filter_line_structure(
    text: str,
    max_short_ratio: float = 0.30,
    max_caps_ratio: float = 0.20,
    short_threshold: int = 20,
) -> bool:
    """
    G4: Line structure filter.
    Drop if:
    - >max_short_ratio of lines are very short (<short_threshold chars)
    - OR >max_caps_ratio of lines are ALL CAPS
    """
    lines = text.split("\n")
    if not lines:
        return True

    non_empty_lines = [line for line in lines if line.strip()]
    if not non_empty_lines:
        return True

    total = len(non_empty_lines)

    # Count short lines
    short_count = sum(1 for line in non_empty_lines if len(line.strip()) < short_threshold)
    if short_count / total > max_short_ratio:
        return False

    # Count all-caps lines (only consider lines with at least some letters)
    caps_count = 0
    lines_with_letters = 0
    for line in non_empty_lines:
        stripped = line.strip()
        letters = [c for c in stripped if c.isalpha()]
        if letters:
            lines_with_letters += 1
            if all(c.isupper() for c in letters):
                caps_count += 1

    if lines_with_letters > 0 and caps_count / lines_with_letters > max_caps_ratio:
        return False

    return True


# Default boilerplate phrases to filter
DEFAULT_BOILERPLATE = [
    "cookie policy",
    "privacy policy",
    "terms of service",
    "all rights reserved",
    "subscribe to our newsletter",
    "sign up to continue",
    "log in to continue",
    "enable javascript",
    "accept cookies",
    "we use cookies",
    "javascript is required",
    "please enable cookies",
]


def filter_boilerplate(text: str, blacklist: Optional[List[str]] = None) -> bool:
    """G5: Drop if doc contains boilerplate phrases (case-insensitive)."""
    if blacklist is None:
        blacklist = DEFAULT_BOILERPLATE
    text_lower = text.lower()
    for phrase in blacklist:
        if phrase.lower() in text_lower:
            return False
    return True


def apply_global_filters(
    text: str,
    min_chars: int = 300,
    max_chars: int = 80000,
    min_alpha_ratio: float = 0.65,
    max_repeat: int = 6,
    max_weird_ratio: float = 0.01,
    max_short_line_ratio: float = 0.30,
    max_caps_ratio: float = 0.20,
    boilerplate_blacklist: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """
    Apply all global filters (G0-G5).
    Returns (passed, reason) where reason is empty string if passed.
    """
    if not filter_by_length(text, min_chars, max_chars):
        return False, "G0_length"
    if not filter_by_alpha_ratio(text, min_alpha_ratio):
        return False, "G1_alpha_ratio"
    if not filter_repeated_chars(text, max_repeat):
        return False, "G2_repeated_chars"
    if not filter_weird_symbols(text, max_weird_ratio):
        return False, "G3_weird_symbols"
    if not filter_line_structure(text, max_short_line_ratio, max_caps_ratio):
        return False, "G4_line_structure"
    if not filter_boilerplate(text, boilerplate_blacklist):
        return False, "G5_boilerplate"
    return True, ""


# =============================================================================
# C4-Specific Filters (C0-C6)
# =============================================================================


def filter_c4_alpha(text: str, min_ratio: float = 0.70) -> bool:
    """C1: Stricter alpha ratio for C4."""
    return compute_alpha_ratio(text) >= min_ratio


def compute_punct_ratio(text: str) -> float:
    """Compute ratio of spam punctuation characters."""
    if not text:
        return 0.0
    punct_count = sum(1 for c in text if c in SPAM_PUNCT)
    return punct_count / len(text)


def filter_punctuation_spam(text: str, max_ratio: float = 0.20) -> bool:
    """C2: Drop if punctuation ratio > max_ratio."""
    if compute_punct_ratio(text) > max_ratio:
        return False

    # Also check for repeated punctuation patterns
    spam_patterns = [
        r"/{4,}",  # ////
        r"\|{4,}",  # ||||
        r"={4,}",  # ====
        r"-{4,}",  # ----
        r"'{6,}",  # ''''''
        r'"{6,}',  # """"""
    ]
    for pattern in spam_patterns:
        if re.search(pattern, text):
            return False
    return True


# Web junk keywords that indicate low-quality content
DEFAULT_WEB_JUNK_KEYWORDS = [
    # E-commerce
    "shipping",
    "add to cart",
    "buy now",
    "coupon",
    "free download",
    "click here",
    "affiliate",
    "checkout",
    "price $",
    "order now",
    "limited time",
    "sale ends",
    "discount code",
    # Gambling/betting spam
    "betting tips",
    "free bets",
    "betting predictions",
    "soccer predictions",
    "football predictions",
    "sports betting",
    "bet now",
    "odds today",
    "inplay bet",
    "in-play bet",
    "bookmaker",
    "casino bonus",
    "free spins",
    "slot machine",
    "poker bonus",
    # SEO spam
    "seo services",
    "backlinks",
    "link building",
    "guest post",
    "sponsored post",
    # Crypto/finance spam
    "crypto trading",
    "bitcoin trading",
    "forex trading",
    "make money online",
    "work from home",
    "passive income",
    # Promotional/marketing content
    "startup venture",
    "our company",
    "our team",
    "our services",
    "we offer",
    "we provide",
    "contact us today",
    "get in touch",
    "learn more about our",
    "schedule a consultation",
    "free consultation",
    "our experts",
    "industry leader",
    "market leader",
    "best in class",
    "cutting-edge solution",
    "innovative solution",
    "revolutionizing",
    "game-changing",
    "world-class",
    "trusted by",
    "testimonial",
]


def filter_web_junk(
    text: str,
    keywords: Optional[List[str]] = None,
    max_matches: int = 2,
) -> bool:
    """C3: Drop if doc contains many web junk keywords."""
    if keywords is None:
        keywords = DEFAULT_WEB_JUNK_KEYWORDS
    text_lower = text.lower()
    match_count = sum(1 for kw in keywords if kw.lower() in text_lower)
    return match_count < max_matches


# Navigation keywords
NAV_KEYWORDS = ["home", "about", "contact", "privacy", "terms", "faq", "sitemap"]


def filter_nav_patterns(text: str, max_nav_keywords: int = 4, max_pipe_ratio: float = 0.005) -> bool:
    """C4: Drop if doc looks like navigation/menu content."""
    text_lower = text.lower()

    # Check for multiple nav keywords
    nav_count = sum(1 for kw in NAV_KEYWORDS if kw in text_lower)
    if nav_count >= max_nav_keywords:
        return False

    # Check for excessive pipe separators (common in nav)
    if text:
        pipe_ratio = text.count("|") / len(text)
        if pipe_ratio > max_pipe_ratio:
            return False

    return True


def filter_paragraph_quality(
    text: str,
    min_paragraphs: int = 1,
    min_avg_length: int = 100,
) -> bool:
    """C5: Drop if doc has insufficient paragraph structure.
    
    Relaxed for C4: web text often lacks blank-line paragraph breaks.
    Now splits on single newlines as well and has lower thresholds.
    """
    # Split on any newline sequence (single or double)
    paragraphs = re.split(r"\n+", text)
    # Filter out very short lines (likely headers or noise)
    real_paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 30]

    # Accept single-paragraph documents if they're substantial
    if len(real_paragraphs) == 0:
        # No real paragraphs - check if whole text is substantial
        if len(text.strip()) >= 300:
            return True
        return False

    if len(real_paragraphs) < min_paragraphs:
        return False

    avg_length = sum(len(p) for p in real_paragraphs) / len(real_paragraphs)
    return avg_length >= min_avg_length


def compute_char_entropy(text: str) -> float:
    """Compute character-level entropy of text."""
    if not text:
        return 0.0
    freq = Counter(text)
    total = len(text)
    entropy = 0.0
    for count in freq.values():
        if count > 0:
            prob = count / total
            entropy -= prob * math.log2(prob)
    return entropy


def filter_entropy(text: str, min_entropy: float = 3.5, max_entropy: float = 5.6) -> bool:
    """C6: Drop if entropy is outside acceptable range."""
    entropy = compute_char_entropy(text)
    return min_entropy <= entropy <= max_entropy


def apply_c4_filters(
    text: str,
    min_alpha_ratio: float = 0.70,
    max_punct_ratio: float = 0.20,
    web_junk_keywords: Optional[List[str]] = None,
    max_web_junk_matches: int = 2,
    min_paragraphs: int = 2,
    min_avg_para_length: int = 200,
    min_entropy: float = 3.5,
    max_entropy: float = 5.6,
) -> Tuple[bool, str]:
    """
    Apply all C4-specific filters.
    Returns (passed, reason).
    """
    if not filter_c4_alpha(text, min_alpha_ratio):
        return False, "C1_alpha_ratio"
    if not filter_punctuation_spam(text, max_punct_ratio):
        return False, "C2_punct_spam"
    if not filter_web_junk(text, web_junk_keywords, max_web_junk_matches):
        return False, "C3_web_junk"
    if not filter_nav_patterns(text):
        return False, "C4_nav_patterns"
    if not filter_paragraph_quality(text, min_paragraphs, min_avg_para_length):
        return False, "C5_paragraph_quality"
    if not filter_entropy(text, min_entropy, max_entropy):
        return False, "C6_entropy"
    return True, ""


# =============================================================================
# Wikipedia-Specific Filters (W0-W4)
# =============================================================================

# Section headers to remove from Wikipedia
WIKI_REMOVE_SECTIONS = [
    "references",
    "external links",
    "see also",
    "further reading",
    "notes",
    "bibliography",
    "sources",
    "citations",
]


def strip_wiki_sections(text: str, sections_to_remove: Optional[List[str]] = None) -> str:
    """
    W0: Remove non-prose sections from Wikipedia text.
    Removes sections starting with certain headers until next section or end.
    """
    if sections_to_remove is None:
        sections_to_remove = WIKI_REMOVE_SECTIONS

    lines = text.split("\n")
    result_lines = []
    skip_until_next_section = False

    for line in lines:
        stripped = line.strip().lower()
        # Check if this is a section header to remove
        # Wikipedia section headers are often "== Section ==" format
        is_section_header = stripped.startswith("==") or stripped.endswith("==")

        if is_section_header:
            # Extract section name
            section_name = stripped.strip("= ").lower()
            if any(remove_sec in section_name for remove_sec in sections_to_remove):
                skip_until_next_section = True
                continue
            else:
                skip_until_next_section = False

        if not skip_until_next_section:
            result_lines.append(line)

    return "\n".join(result_lines)


def filter_list_heavy(text: str, max_list_ratio: float = 0.30) -> bool:
    """W1: Drop paragraphs where too many lines are list items."""
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return True

    # Count lines that look like list items
    list_patterns = [
        r"^\s*[\*\-\•]\s",  # Bullet points
        r"^\s*\d+[\.\)]\s",  # Numbered lists
        r"^\s*[a-zA-Z][\.\)]\s",  # Lettered lists
    ]
    list_count = 0
    for line in lines:
        for pattern in list_patterns:
            if re.match(pattern, line):
                list_count += 1
                break

    return list_count / len(lines) <= max_list_ratio


def count_sentences(text: str) -> int:
    """Count approximate number of sentences in text."""
    # Simple heuristic: count sentence-ending punctuation followed by space or end
    sentence_endings = re.findall(r"[.!?]+(?:\s|$)", text)
    return len(sentence_endings)


def filter_short_paragraphs(text: str, min_sentences: int = 2) -> bool:
    """W2: Keep only paragraphs with sufficient sentences."""
    paragraphs = re.split(r"\n\s*\n", text)
    valid_paragraphs = 0
    for para in paragraphs:
        para = para.strip()
        if para and count_sentences(para) >= min_sentences:
            valid_paragraphs += 1

    # Need at least one valid paragraph
    return valid_paragraphs > 0


def strip_wiki_markup(text: str) -> str:
    """W3: Strip common wiki markup artifacts."""
    # Remove template markers {{ }}
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    # Remove [[ ]] links, keeping the display text
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    # Remove remaining brackets
    text = re.sub(r"\[|\]", "", text)
    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def apply_wiki_filters(text: str, max_list_ratio: float = 0.30) -> Tuple[str, bool, str]:
    """
    Apply Wikipedia-specific filters and transformations.
    Returns (transformed_text, passed, reason).
    """
    # First strip sections and markup
    text = strip_wiki_sections(text)
    text = strip_wiki_markup(text)

    # Then apply filters
    if not filter_list_heavy(text, max_list_ratio):
        return text, False, "W1_list_heavy"
    if not filter_short_paragraphs(text):
        return text, False, "W2_short_paragraphs"

    return text, True, ""


# =============================================================================
# Gutenberg-Specific Filters (GUT0-GUT5)
# =============================================================================

# Patterns for Gutenberg start/end markers
GUTENBERG_START_PATTERNS = [
    r"\*\*\*\s*START OF (?:THE |THIS )?PROJECT GUTENBERG",
    r"\*\*\*\s*START OF (?:THE )?EBOOK",
    r"START OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK",
]

GUTENBERG_END_PATTERNS = [
    r"\*\*\*\s*END OF (?:THE |THIS )?PROJECT GUTENBERG",
    r"\*\*\*\s*END OF (?:THE )?EBOOK",
    r"END OF (?:THE |THIS )?PROJECT GUTENBERG EBOOK",
]


def strip_gutenberg_boilerplate(text: str) -> str:
    """
    GUT0: Strip Gutenberg header/footer boilerplate.
    Keep text only between START and END markers.
    """
    # Find start marker
    start_idx = 0
    for pattern in GUTENBERG_START_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Start after the line containing the marker
            start_idx = text.find("\n", match.end())
            if start_idx == -1:
                start_idx = match.end()
            else:
                start_idx += 1
            break

    # Find end marker
    end_idx = len(text)
    for pattern in GUTENBERG_END_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # End before the line containing the marker
            end_idx = text.rfind("\n", 0, match.start())
            if end_idx == -1:
                end_idx = match.start()
            break

    # If no markers found, try fallback: look for chapter markers
    if start_idx == 0:
        chapter_match = re.search(r"\n\s*(CHAPTER|Chapter|PART|Part)\s+[IVX1-9]", text)
        if chapter_match:
            # Start a bit before the chapter marker
            start_idx = max(0, chapter_match.start())

    return text[start_idx:end_idx].strip()


# TOC patterns to remove
TOC_PATTERNS = [
    r"^CONTENTS?\s*$",
    r"^TABLE OF CONTENTS?\s*$",
    r"^INDEX\s*$",
]


def strip_toc(text: str) -> str:
    """GUT1: Remove table of contents sections."""
    lines = text.split("\n")
    result_lines = []
    in_toc = False
    toc_line_count = 0

    for i, line in enumerate(lines):
        stripped = line.strip().upper()

        # Check if this starts a TOC
        if any(re.match(pattern, stripped) for pattern in TOC_PATTERNS):
            in_toc = True
            toc_line_count = 0
            continue

        if in_toc:
            # TOC lines are usually short with page numbers or dots
            is_toc_line = (
                len(stripped) < 80
                and (
                    re.search(r"\.\s*\d+\s*$", stripped)  # "Chapter 1 ... 15"
                    or re.search(r"^\s*[IVX]+\s*$", stripped)  # Roman numerals alone
                    or re.search(r"^CHAPTER\s+[IVX\d]+", stripped)  # Chapter listings
                    or stripped == ""
                )
            )
            if is_toc_line:
                toc_line_count += 1
                if toc_line_count > 50:  # Safety: don't skip too much
                    in_toc = False
                continue
            else:
                # End of TOC
                in_toc = False

        result_lines.append(line)

    return "\n".join(result_lines)


def normalize_gutenberg(text: str) -> str:
    """
    GUT3: Normalize Gutenberg text.
    - Join hyphenated line breaks (scanning artifact)
    - Merge hard-wrapped lines into paragraphs
    """
    # Join hyphenated words split across lines
    text = re.sub(r"-\n\s*", "", text)

    # Normalize multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Merge lines that don't end with sentence-ending punctuation
    # and aren't followed by blank lines (hard-wrapped paragraphs)
    lines = text.split("\n")
    result_lines = []
    buffer = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            # Blank line - flush buffer and add blank line
            if buffer:
                result_lines.append(" ".join(buffer))
                buffer = []
            result_lines.append("")
            continue

        # Check if previous line ended mid-sentence
        if buffer:
            last_char = buffer[-1][-1] if buffer[-1] else ""
            # If last line didn't end with sentence punctuation, merge
            if last_char not in ".!?:;\"'" and not stripped[0].isupper():
                buffer.append(stripped)
                continue
            else:
                result_lines.append(" ".join(buffer))
                buffer = []

        buffer.append(stripped)

    if buffer:
        result_lines.append(" ".join(buffer))

    return "\n".join(result_lines)


def filter_poetry_heavy(text: str, max_short_line_ratio: float = 0.50, short_threshold: int = 40) -> bool:
    """GUT2: Filter out poetry-heavy content (optional)."""
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return True

    short_count = sum(1 for line in lines if len(line.strip()) < short_threshold)
    return short_count / len(lines) <= max_short_line_ratio


def apply_gutenberg_filters(
    text: str,
    filter_poetry: bool = False,
    max_short_line_ratio: float = 0.50,
) -> Tuple[str, bool, str]:
    """
    Apply Gutenberg-specific filters and transformations.
    Returns (transformed_text, passed, reason).
    """
    # Strip boilerplate first
    text = strip_gutenberg_boilerplate(text)
    text = strip_toc(text)
    text = normalize_gutenberg(text)

    # Optional poetry filter
    if filter_poetry and not filter_poetry_heavy(text, max_short_line_ratio):
        return text, False, "GUT2_poetry_heavy"

    return text, True, ""


# =============================================================================
# Text Chunking
# =============================================================================


def chunk_text(
    text: str,
    max_chars: int = 8000,
    min_chars: int = 500,
    split_on: str = "\n\n",
) -> List[str]:
    """
    Split long text into chunks of reasonable size.
    Tries to split on paragraph boundaries.
    """
    if len(text) <= max_chars:
        return [text] if len(text) >= min_chars else []

    chunks = []
    current_chunk = []
    current_len = 0

    parts = text.split(split_on)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        part_len = len(part)

        # If single part is too long, split it further
        if part_len > max_chars:
            # Flush current chunk
            if current_chunk:
                chunk_text = f"{split_on}".join(current_chunk)
                if len(chunk_text) >= min_chars:
                    chunks.append(chunk_text)
                current_chunk = []
                current_len = 0

            # Split large part on sentences
            sentences = re.split(r"(?<=[.!?])\s+", part)
            for sentence in sentences:
                if current_len + len(sentence) > max_chars and current_chunk:
                    chunk_text = " ".join(current_chunk)
                    if len(chunk_text) >= min_chars:
                        chunks.append(chunk_text)
                    current_chunk = []
                    current_len = 0
                current_chunk.append(sentence)
                current_len += len(sentence)
            continue

        # Check if adding this part would exceed max
        if current_len + part_len + len(split_on) > max_chars and current_chunk:
            chunk_text = f"{split_on}".join(current_chunk)
            if len(chunk_text) >= min_chars:
                chunks.append(chunk_text)
            current_chunk = []
            current_len = 0

        current_chunk.append(part)
        current_len += part_len + len(split_on)

    # Flush remaining
    if current_chunk:
        chunk_text = f"{split_on}".join(current_chunk)
        if len(chunk_text) >= min_chars:
            chunks.append(chunk_text)

    return chunks


# =============================================================================
# MinHash Deduplication
# =============================================================================


@dataclass
class MinHashDeduplicator:
    """
    Fast hash-based deduplication.
    Uses normalized text hashing for exact and near-duplicate detection.
    Much faster than true MinHash while catching most duplicates.
    """

    threshold: float = 0.85  # Kept for API compatibility
    num_perm: int = 128  # Kept for API compatibility
    _seen_hashes: set = field(default_factory=set)
    _seen_prefixes: set = field(default_factory=set)

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        import re
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:500]

    def is_duplicate(self, text: str, doc_id: Optional[str] = None) -> bool:
        """
        Check if text is a duplicate of any seen text.
        Uses fast hash-based comparison.
        Returns True if duplicate (should be dropped).
        """
        import re
        
        # Normalize: lowercase, collapse whitespace
        normalized = text.lower()
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Check exact duplicate (first 500 chars)
        text_hash = hash(normalized[:500])
        if text_hash in self._seen_hashes:
            return True
        
        # Check prefix duplicate (first 200 chars - catches near-duplicates)
        prefix_hash = hash(normalized[:200])
        if prefix_hash in self._seen_prefixes:
            return True
        
        # Not a duplicate, add to collection
        self._seen_hashes.add(text_hash)
        self._seen_prefixes.add(prefix_hash)
        return False

    def clear(self):
        """Clear all stored hashes."""
        self._seen_hashes.clear()
        self._seen_prefixes.clear()

    def __len__(self) -> int:
        """Return number of stored documents."""
        return len(self._seen_hashes)


# =============================================================================
# Filter Statistics Tracking
# =============================================================================


@dataclass
class FilterStats:
    """Track filtering statistics for debugging and reporting."""

    total_docs: int = 0
    passed_docs: int = 0
    rejected_by: dict = field(default_factory=lambda: Counter())

    def record_pass(self):
        self.total_docs += 1
        self.passed_docs += 1

    def record_reject(self, reason: str):
        self.total_docs += 1
        self.rejected_by[reason] += 1

    def pass_rate(self) -> float:
        if self.total_docs == 0:
            return 0.0
        return self.passed_docs / self.total_docs

    def report(self) -> str:
        lines = [
            f"Total documents: {self.total_docs}",
            f"Passed: {self.passed_docs} ({self.pass_rate():.1%})",
            "Rejected by filter:",
        ]
        for reason, count in sorted(self.rejected_by.items(), key=lambda x: -x[1]):
            pct = count / self.total_docs if self.total_docs > 0 else 0
            lines.append(f"  {reason}: {count} ({pct:.1%})")
        return "\n".join(lines)

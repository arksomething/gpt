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
    max_short_ratio: float = 0.40,
    max_caps_ratio: float = 0.25,
    short_threshold: int = 15,
) -> bool:
    """
    G4: Line structure filter.
    Drop if:
    - >max_short_ratio of lines are very short (<short_threshold chars)
    - OR >max_caps_ratio of lines are ALL CAPS
    Relaxed thresholds to allow more valid content.
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


def filter_repetitive_phrases(
    text: str,
    min_phrase_len: int = 4,
    max_phrase_len: int = 12,
    max_repeat_ratio: float = 0.03,
) -> bool:
    """
    C7: Drop if any phrase repeats too many times (SEO spam indicator).
    Checks for repeated word n-grams that appear suspiciously often.
    """
    words = text.lower().split()
    if len(words) < 20:
        return True
    
    total_words = len(words)
    
    # Check various n-gram sizes
    for n in range(min_phrase_len, min(max_phrase_len + 1, len(words))):
        ngram_counts: Counter = Counter()
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i+n])
            ngram_counts[ngram] += 1
        
        # Check if any n-gram appears too frequently
        for ngram, count in ngram_counts.most_common(5):
            if count > 2:  # Must appear at least 3 times
                # Ratio of words covered by this repeated phrase
                phrase_coverage = (count * n) / total_words
                if phrase_coverage > max_repeat_ratio:
                    return False
    
    return True


def filter_url_attribution(
    text: str,
    max_url_ratio: float = 0.02,
    max_attribution_matches: int = 3,
) -> bool:
    """
    C8: Drop if text contains excessive URL patterns or image attribution.
    Catches scraped content with "sourced from:", "screenshot from:", etc.
    """
    text_lower = text.lower()
    text_len = len(text)
    
    # Count URL-like patterns
    url_patterns = [
        r"https?://[^\s]+",
        r"www\.[^\s]+",
        r"\.com[/\s]",
        r"\.org[/\s]",
        r"\.net[/\s]",
    ]
    url_count = 0
    for pattern in url_patterns:
        url_count += len(re.findall(pattern, text_lower))
    
    if text_len > 0 and url_count / (text_len / 100) > max_url_ratio:
        return False
    
    # Check for attribution patterns (scraped content)
    attribution_patterns = [
        "sourced from:",
        "screenshot from:",
        "image from:",
        "photo from:",
        "received from:",
        "courtesy of:",
        "credit:",
        "via:",
        "source:",
        "from:",
    ]
    # More specific patterns that indicate scraping
    scrape_patterns = [
        r"screenshot (?:sourced|received|taken) from",
        r"image (?:sourced|received|taken) from",
        r"(?:sourced|received) from: \w+\.com",
        r"bring it (?:along|with you|together)",  # Common in scraped content
        r"take it (?:along|with you|together)",
    ]
    
    scrape_count = 0
    for pattern in scrape_patterns:
        scrape_count += len(re.findall(pattern, text_lower))
    
    if scrape_count >= max_attribution_matches:
        return False
    
    return True


def filter_bibliography_heavy(
    text: str,
    max_biblio_ratio: float = 0.25,
) -> bool:
    """
    C9: Drop if text is dominated by bibliography/reference patterns.
    Catches encyclopedia articles that are mostly citation lists.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 5:
        return True
    
    biblio_patterns = [
        r"^\d{4}[.:)]",  # Year at start: "1998." or "1998:"
        r"^\([^)]+\)\s*$",  # Parenthetical only: "(Paris, 1789)"
        r"vol\.\s*\d+",  # Volume references
        r"pp?\.\s*\d+",  # Page references  
        r"ISBN\s*[\d-]+",
        r"ISSN\s*[\d-]+",
        r"et\s+al\.",
        r"^\s*[\-–—]\s*",  # Lines starting with dashes (bibliographic entries)
        r"reprint|reprinted",
        r"edition|ed\.",
        r"translated by|trans\.",
        r"edited by",
        r"published by",
    ]
    
    biblio_lines = 0
    for line in lines:
        line_lower = line.lower()
        for pattern in biblio_patterns:
            if re.search(pattern, line_lower, re.IGNORECASE):
                biblio_lines += 1
                break
    
    if biblio_lines / len(lines) > max_biblio_ratio:
        return False
    
    return True


def filter_listicle_pattern(
    text: str,
    max_repetitive_starts: float = 0.20,
) -> bool:
    """
    C10: Drop if text has listicle structure (many lines starting same way).
    Catches "10 Ways to..." style low-quality content.
    """
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 10]
    if len(lines) < 5:
        return True
    
    # Get first few words of each line
    line_starts: Counter = Counter()
    for line in lines:
        words = line.split()[:3]
        if len(words) >= 2:
            start = " ".join(words[:2]).lower()
            line_starts[start] += 1
    
    # Check if any start pattern is too common
    total_lines = len(lines)
    for start, count in line_starts.most_common(3):
        if count >= 3 and count / total_lines > max_repetitive_starts:
            return False
    
    return True


def filter_low_info_density(
    text: str,
    min_unique_word_ratio: float = 0.35,
    max_filler_ratio: float = 0.15,
) -> bool:
    """
    C11: Drop if text has low information density.
    Catches repetitive, low-quality content.
    """
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if len(words) < 50:
        return True
    
    # Check unique word ratio
    unique_words = set(words)
    unique_ratio = len(unique_words) / len(words)
    if unique_ratio < min_unique_word_ratio:
        return False
    
    # Check for filler phrases
    filler_phrases = [
        "you will", "you can", "you may", "you should",
        "it is", "it was", "it will", "it can",
        "this is", "this was", "this will",
        "that is", "that was", "that will",
        "there is", "there are", "there was", "there were",
        "as well as", "in order to", "due to the fact",
        "at the end of the day", "when it comes to",
        "in terms of", "on the other hand",
    ]
    
    text_lower = text.lower()
    filler_count = 0
    for phrase in filler_phrases:
        filler_count += text_lower.count(phrase)
    
    word_count = len(words)
    if word_count > 0 and filler_count / (word_count / 10) > max_filler_ratio:
        return False
    
    return True


def filter_non_latin_heavy(
    text: str,
    max_non_latin_ratio: float = 0.01,
) -> bool:
    """
    C14: Drop if text has too many non-Latin characters.
    These become unknown tokens (⁇) after tokenization.
    Very strict - even 1% non-Latin is rejected.
    """
    if not text:
        return True
    
    # Count non-Latin alphabetic characters
    non_latin_count = 0
    latin_count = 0
    for char in text:
        if char.isalpha():
            # Check if it's in basic Latin range (ASCII + Latin-1 Supplement)
            if ord(char) < 256:
                latin_count += 1
            else:
                non_latin_count += 1
    
    total_alpha = latin_count + non_latin_count
    if total_alpha == 0:
        return True
    
    # Reject if ANY non-Latin characters when text is short
    if total_alpha < 500 and non_latin_count > 0:
        return False
    
    if non_latin_count / total_alpha > max_non_latin_ratio:
        return False
    
    return True


def filter_reference_sections(
    text: str,
    max_ref_line_ratio: float = 0.10,
) -> bool:
    """
    C15: Drop if text is dominated by reference/link patterns.
    Catches Wikipedia footers, citation sections, etc.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 3:
        return True
    
    ref_patterns = [
        r"^\^",  # Citation markers
        r"^\[\d+\]",  # [1], [2] references
        r"^https?://",  # URLs at line start
        r"^www\.",  # www URLs
        r"^\d+\.\s+\^",  # Numbered citations
        r"^Category:",  # Wikipedia categories
        r"^Categories:",
        r"^\*\s*\[",  # Bullet with link
        r"^\*\s*http",  # Bullet with URL
        r"^Retrieved\s",  # "Retrieved from..."
        r"^Archived\s",  # "Archived from..."
        r"^ISBN\s",
        r"^ISSN\s",
        r"^doi:",
        r"^pmid:",
        r"^arXiv:",
    ]
    
    ref_lines = 0
    for line in lines:
        for pattern in ref_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                ref_lines += 1
                break
    
    if ref_lines / len(lines) > max_ref_line_ratio:
        return False
    
    return True


def filter_category_lists(
    text: str,
    max_category_ratio: float = 0.25,
) -> bool:
    """
    C16: Drop if text ends with category/tag lists (Wikipedia footers).
    More lenient - only reject if dominated by categories.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 10:
        return True
    
    # Check last N lines for category patterns
    check_lines = lines[-min(15, len(lines)):]
    
    category_patterns = [
        r"^(People|Living people|Deaths|Births)$",
        r"^\d{4} (births|deaths)$",
    ]
    
    category_count = 0
    for line in check_lines:
        # Very short lines at the end that look like categories
        if len(line) < 40:
            for pattern in category_patterns:
                if re.match(pattern, line):
                    category_count += 1
                    break
    
    if len(check_lines) > 0 and category_count / len(check_lines) > max_category_ratio:
        return False
    
    return True


def filter_bad_start(text: str) -> bool:
    """
    C17: Drop if text starts mid-sentence or with garbage.
    Catches chunks that were cut at bad boundaries.
    """
    text = text.lstrip()
    if not text:
        return False
    
    # Bad starting characters (mid-sentence indicators)
    bad_starts = [
        ',', ';', ':', '.', '!', '?',  # Punctuation
        ')', ']', '}', '>',  # Closing brackets
        "'s ", "'t ", "'ve ", "'re ", "'ll ", "'d ",  # Contractions
        "and ", "or ", "but ", "so ", "yet ",  # Conjunctions (lowercase)
        "which ", "that ", "who ", "whom ", "whose ",  # Relative pronouns (lowercase)
        "is ", "are ", "was ", "were ", "be ", "been ",  # Verbs (lowercase)
        "in ", "on ", "at ", "to ", "for ", "with ",  # Prepositions (lowercase)
    ]
    
    for bad in bad_starts:
        if text.startswith(bad):
            return False
    
    # Check if first character is lowercase letter (mid-sentence)
    first_char = text[0]
    if first_char.islower():
        return False
    
    return True


def filter_bad_end(text: str) -> bool:
    """
    C18: Drop if text ends mid-sentence.
    """
    text = text.rstrip()
    if not text:
        return False
    
    # Should end with sentence-ending punctuation or quote
    good_endings = '.!?"\')'
    
    # Check last non-whitespace character
    last_char = text[-1]
    if last_char not in good_endings:
        words = text.split()
        if words:
            last_word = words[-1].rstrip('.,;:!?"\'-)')
            last_word_lower = last_word.lower()
            
            # Common words that indicate mid-sentence cutoff
            bad_ending_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'so', 'yet', 'for', 'nor',
                'in', 'on', 'at', 'to', 'of', 'by', 'with', 'from', 'into', 'onto',
                'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'has', 'have', 'had', 'having',
                'do', 'does', 'did', 'doing',
                'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                'that', 'which', 'who', 'whom', 'whose', 'where', 'when', 'while',
                'if', 'then', 'than', 'as', 'because', 'although', 'though',
                'this', 'that', 'these', 'those', 'its', 'his', 'her', 'their', 'our',
                'not', "didn't", "doesn't", "wasn't", "weren't", "isn't", "aren't",
                "won't", "wouldn't", "couldn't", "shouldn't", "can't", "haven't",
            }
            
            if last_word_lower in bad_ending_words:
                return False
            
            # If last word is very short (<=3), probably cut off
            if len(last_word) <= 3:
                return False
            
            # If ends with common incomplete patterns
            if last_word_lower.endswith("'s") or last_word_lower.endswith("n't"):
                return False
    
    return True


def apply_c4_filters(
    text: str,
    min_alpha_ratio: float = 0.70,
    max_punct_ratio: float = 0.20,
    web_junk_keywords: Optional[List[str]] = None,
    max_web_junk_matches: int = 2,
    min_paragraphs: int = 1,
    min_avg_para_length: int = 80,
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
    if not filter_repetitive_phrases(text):
        return False, "C7_repetitive_phrases"
    if not filter_url_attribution(text):
        return False, "C8_url_attribution"
    if not filter_bibliography_heavy(text):
        return False, "C9_bibliography"
    if not filter_listicle_pattern(text):
        return False, "C10_listicle"
    if not filter_low_info_density(text):
        return False, "C11_low_info"
    if not filter_list_heavy(text, max_list_ratio=0.15):
        return False, "C12_list_heavy"
    if not filter_structured_data(text, max_structured_ratio=0.20):
        return False, "C13_structured_data"
    if not filter_non_latin_heavy(text):
        return False, "C14_non_latin"
    if not filter_reference_sections(text):
        return False, "C15_references"
    if not filter_category_lists(text):
        return False, "C16_categories"
    if not filter_bad_start(text):
        return False, "C17_bad_start"
    if not filter_bad_end(text):
        return False, "C18_bad_end"
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
        r"^\s*[\*\-\•\→\►\▪\◦\‣]\s",  # Bullet points (expanded)
        r"^\s*\d+[\.\):\-]\s",  # Numbered lists: "1." "1)" "1:" "1-"
        r"^\s*[a-zA-Z][\.\)]\s",  # Lettered lists
        r"^\s*[ivxIVX]+[\.\)]\s",  # Roman numeral lists
        r"^\s*\[[^\]]+\]\s*$",  # Category tags: [Category Name]
        r"^\s*\w+[^.!?]{0,40}:\s*[\d,\.]+\s*(rifles?|troops?|men|soldiers?|people|km|m|ft|miles?|hours?|days?|years?)?\s*$",  # "Label: number" patterns
    ]
    list_count = 0
    for line in lines:
        for pattern in list_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                list_count += 1
                break

    return list_count / len(lines) <= max_list_ratio


def filter_structured_data(text: str, max_structured_ratio: float = 0.25) -> bool:
    """
    W1b: Drop text that looks like structured data (tables, key-value pairs).
    Catches Wikipedia infoboxes, data tables, etc.
    """
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) < 5:
        return True
    
    structured_patterns = [
        r"^[^:]{1,40}:\s+.+$",  # Key: Value patterns (short keys)
        r"^\|",  # Wiki table rows
        r"^!",  # Wiki table headers
        r"^\{",  # Template starts
        r"^[A-Z][a-z]+:\s*\d",  # "Population: 12345"
        r"^[A-Z][a-z]+:\s*[A-Z]",  # "Capital: London"
        r"^\s*\d{4}\s*[-–:]\s*\d",  # Year ranges or year: number patterns
    ]
    
    structured_count = 0
    for line in lines:
        for pattern in structured_patterns:
            if re.match(pattern, line.strip()):
                structured_count += 1
                break
    
    return structured_count / len(lines) <= max_structured_ratio


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


def apply_wiki_filters(text: str, max_list_ratio: float = 0.20) -> Tuple[str, bool, str]:
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
    if not filter_structured_data(text):
        return text, False, "W1b_structured_data"
    if not filter_short_paragraphs(text):
        return text, False, "W2_short_paragraphs"
    if not filter_non_latin_heavy(text):
        return text, False, "W3_non_latin"
    if not filter_reference_sections(text):
        return text, False, "W4_references"
    if not filter_category_lists(text):
        return text, False, "W5_categories"

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


def strip_chapter_headers(text: str) -> str:
    """
    GUT2b: Strip chapter/section headers from Gutenberg text.
    Removes lines like "CHAPTER XVIII", "Chapter 1", "PART II", etc.
    """
    lines = text.split("\n")
    result_lines = []
    
    # Patterns for chapter/section headers to remove
    header_patterns = [
        r"^\s*CHAPTER\s+[IVXLCDM\d]+\.?\s*$",  # CHAPTER XVIII or CHAPTER 18
        r"^\s*Chapter\s+[IVXLCDM\d]+\.?\s*$",  # Chapter XVIII or Chapter 18
        r"^\s*PART\s+[IVXLCDM\d]+\.?\s*$",  # PART II
        r"^\s*Part\s+[IVXLCDM\d]+\.?\s*$",  # Part II
        r"^\s*BOOK\s+[IVXLCDM\d]+\.?\s*$",  # BOOK III
        r"^\s*Book\s+[IVXLCDM\d]+\.?\s*$",  # Book III
        r"^\s*SECTION\s+[IVXLCDM\d]+\.?\s*$",  # SECTION IV
        r"^\s*Section\s+[IVXLCDM\d]+\.?\s*$",  # Section IV
        r"^\s*ACT\s+[IVXLCDM\d]+\.?\s*$",  # ACT III (plays)
        r"^\s*Act\s+[IVXLCDM\d]+\.?\s*$",  # Act III
        r"^\s*SCENE\s+[IVXLCDM\d]+\.?\s*$",  # SCENE II
        r"^\s*Scene\s+[IVXLCDM\d]+\.?\s*$",  # Scene II
        r"^\s*[IVXLCDM]+\.\s*$",  # Just "XVIII." alone
        r"^\s*\d+\.\s*$",  # Just "18." alone
    ]
    
    for line in lines:
        is_header = False
        for pattern in header_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                is_header = True
                break
        if not is_header:
            result_lines.append(line)
    
    return "\n".join(result_lines)


def normalize_gutenberg(text: str) -> str:
    """
    GUT3: Normalize Gutenberg text.
    - Strip chapter headers
    - Join hyphenated line breaks (scanning artifact)
    - Merge hard-wrapped lines into paragraphs
    """
    # Strip chapter/section headers first
    text = strip_chapter_headers(text)
    
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
    Tries to split on paragraph boundaries, then sentence boundaries.
    Ensures chunks start and end cleanly.
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

        # If single part is too long, split it further on sentences
        if part_len > max_chars:
            # Flush current chunk
            if current_chunk:
                chunk_text_str = f"{split_on}".join(current_chunk)
                if len(chunk_text_str) >= min_chars:
                    chunks.append(chunk_text_str)
                current_chunk = []
                current_len = 0

            # Split large part on sentence boundaries
            # More robust sentence splitting
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', part)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if current_len + len(sentence) > max_chars and current_chunk:
                    chunk_text_str = " ".join(current_chunk)
                    if len(chunk_text_str) >= min_chars:
                        chunks.append(chunk_text_str)
                    current_chunk = []
                    current_len = 0
                current_chunk.append(sentence)
                current_len += len(sentence) + 1  # +1 for space
            continue

        # Check if adding this part would exceed max
        if current_len + part_len + len(split_on) > max_chars and current_chunk:
            chunk_text_str = f"{split_on}".join(current_chunk)
            if len(chunk_text_str) >= min_chars:
                chunks.append(chunk_text_str)
            current_chunk = []
            current_len = 0

        current_chunk.append(part)
        current_len += part_len + len(split_on)

    # Flush remaining
    if current_chunk:
        chunk_text_str = f"{split_on}".join(current_chunk)
        if len(chunk_text_str) >= min_chars:
            chunks.append(chunk_text_str)

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

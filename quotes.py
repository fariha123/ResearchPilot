"""Checking that a quote really comes from a paper's text.

Local models often copy a sentence with small slips (a dropped "the", different punctuation or dashes),
so a strict character-for-character check throws away genuine findings. This accepts:
  "exact": the quote appears word for word (ignoring case, spacing and dash/quote styles), or
  "close": a stretch of the text matches at least 85% of the quote's words, in order, AND every number
           in the quote appears there exactly, so a changed value (94.2% vs 99%) never passes.
"""
import re
import unicodedata
from difflib import SequenceMatcher

WORD = re.compile(r"\d+(?:\.\d+)?|[a-z]+")  # "80nm" -> "80", "nm"; "94.2" stays one number
NUMBER = re.compile(r"\d")
CLOSE_ENOUGH = 0.85


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = re.sub(r"[‐-―−]", "-", text).replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def _fragments(quote):
    parts = (normalize(x).strip(" .\"'") for x in re.split(r"\.\.\.|…", str(quote or "")))
    return [p for p in parts if len(p) >= 12]


def _close(fragment, source_words):
    q = WORD.findall(fragment)
    if len(q) < 4:
        return False
    numbers = [w for w in q if NUMBER.search(w)]
    anchors = set(q[:3])
    best = 0.0
    for i, word in enumerate(source_words):
        if word not in anchors:
            continue
        for start in (max(i - 2, 0), max(i - 1, 0), i):
            window = source_words[start:start + len(q) + 3]
            if numbers and not all(n in window for n in numbers):
                continue
            matched = sum(b.size for b in SequenceMatcher(None, q, window, autojunk=False).get_matching_blocks())
            best = max(best, matched / len(q))
            if best >= CLOSE_ENOUGH:
                return True
    return False


def quote_match(quote, source_text):
    """Return "exact", "close" or None."""
    fragments = _fragments(quote)
    if not fragments:
        return None
    source = normalize(source_text)
    if all(f in source for f in fragments):
        return "exact"
    words = WORD.findall(source)
    if all(_close(f, words) for f in fragments):
        return "close"
    return None


def quote_found(quote, source_text):
    return quote_match(quote, source_text) is not None

from typing import List
from spacy.tokens import Span
from .constants import HEADER_STARTERS
from .normalization import strip_diacritics, canonical_org_key
import re
import unicodedata, string

_PUNCT = set(string.punctuation) | {"–", "—", "«", "»", "·", "•", "…", "º", "ª"}

DEBUG = True

def _is_all_caps_line(ln: str) -> bool:
    t = ln.strip()
    if not t: return False
    letters = [ch for ch in t if ch.isalpha()]
    if not letters: return False
    return all(ch == ch.upper() for ch in letters)

# Cache a sanitized, normalized starter list once (avoids type surprises and repeat work)
STARTERS_NORM = tuple(
    strip_diacritics(str(s)).upper().replace(" ", "").replace("-", "")
    for s in HEADER_STARTERS
)

def _norm_org_key(s: str) -> str:
    """
    Canonical identity for an ORG header:
        - Uicode normalize + remove diacrics
        - Uppercase
        - Drop space, dash variants, and punctuation
    Collapses ALL CAPS vs Mixed Case and minor formatting differences.    
    """
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.upper()
    out = []
    for ch in s:
        if ch.isspace() or ch in {"-", "–", "—"} or ch in _PUNCT:
            continue
        out.append(ch)
    return "".join(out)

def _starts_with_starter(ln: str) -> bool:
    t = ln.strip()
    if not t:
        return False

    # Accumulate only letters from the beginning, skipping spaces, dash variants, soft hyphen, ZW* chars
    head_chars = []
    for ch in t:
        if ch.isalpha():
            head_chars.append(ch)
            continue
        # skip spacing/dash/format chars commonly produced by OCR or letter-spacing
        if ch in {" ", "\t", "-", "–", "—", "\u00AD", "\u200B", "\u200C", "\u200D"}:
            continue
        # stop at other punctuation/digits once we’ve started reading letters
        if head_chars:
            break
        # if we haven't started letters yet, keep skipping leading punctuation
        continue

    head = "".join(head_chars)
    if not head:
        return False

    norm_head = strip_diacritics(head).upper().replace(" ", "").replace("-", "")
    return any(norm_head.startswith(s) for s in STARTERS_NORM)


# Optional: expose normalized key on spans for downstream comparisons
if not Span.has_extension("org_key"):
    Span.set_extension("org_key", default=None)


def find_org_spans(doc, text: str) -> List[Span]:
    """
    Pass 1: detect ALL-CAPS ORG blocks and record their normalized keys (built from the WHOLE block).
    Pass 2: tag later single-line headers as ORG if their canonical key matches a known key.
    """
    org_spans: List[Span] = []
    lines = text.splitlines(keepends=True)
    line_starts, pos = [], 0
    for ln in lines:
        line_starts.append(pos); pos += len(ln)

    canon_keys = set()
    blocks_char_ranges = []

    #--------------- Pass 1: ALL-CAPS blocks -----------------
    i = 0
    while i < len(lines):
        if _starts_with_starter(lines[i]) and _is_all_caps_line(lines[i]):
            start_i = i
            j = i + 1
            while j < len(lines) and _is_all_caps_line(lines[j]):
                j += 1

            start_char = line_starts[start_i]
            end_char = line_starts[j - 1] + len(lines[j - 1])

            # Build the ORG identity from the ENTIRE block (collapse newlines)
            block_text = " ".join(ln.strip() for ln in lines[start_i:j] if ln.strip())
            key = canonical_org_key(block_text)
            canon_keys.add(key)

            chspan = doc.char_span(start_char, end_char, alignment_mode="expand")
            if chspan is not None:
                sp = Span(doc, chspan.start, chspan.end, label="ORG")
                sp._.org_key = key
                org_spans.append(sp)
                blocks_char_ranges.append((start_char, end_char))
            i = j
        else:
            i += 1

    #---------- Pass 2: mixed-case single-line repeats matching known keys --------------
    for idx, ln in enumerate(lines):
        if not _starts_with_starter(ln):
            continue
        ln_start = line_starts[idx]
        ln_end = ln_start + len(ln)
        # skip if already within a Pass 1 block
        if any(a <= ln_start and ln_end <= b for (a, b) in blocks_char_ranges):
            continue
        line_clean = ln.rstrip("\n")
        key = canonical_org_key(line_clean)          # <-- use same canonicalization
        if key in canon_keys:
            chspan = doc.char_span(ln_start, ln_end, alignment_mode="expand")
            if chspan is not None:
                sp = Span(doc, chspan.start, chspan.end, label="ORG")
                sp._.org_key = key
                org_spans.append(sp)

    return org_spans

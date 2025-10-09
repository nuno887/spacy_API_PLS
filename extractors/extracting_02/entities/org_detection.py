from typing import List
from spacy.tokens import Span
from .constants import HEADER_STARTERS
from .normalization import strip_diacritics
import re

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

def _starts_with_starter(ln: str) -> bool:
    #Accent-insensitive, tolerant to spaces/hyphens/fused tokens; prefix match
    t = ln.strip()
    if not t:
        return False
    first_token = re.split(r'[\s\---:,;./]+', t, 1)[0]
    norm_first = strip_diacritics(first_token).upper().replace(" ", "").replace("-", "")
    return any(norm_first.startswith(s) for s in STARTERS_NORM)

def find_org_spans(doc, text: str) -> List[Span]:
    org_spans = []
    lines = text.splitlines(keepends=True)
    line_starts, pos = [], 0
    for ln in lines:
        line_starts.append(pos); pos += len(ln)

    i = 0
    while i < len(lines):
        if _starts_with_starter(lines[i]) and _is_all_caps_line(lines[i]):
            start_i = i
            j = i + 1
            while j < len(lines) and _is_all_caps_line(lines[j]):
                j += 1
            start_char = line_starts[start_i]
            end_char = line_starts[j - 1] + len(lines[j - 1])
            chspan = doc.char_span(start_char, end_char, alignment_mode="expand")
            if chspan is not None:
                org_spans.append(Span(doc, chspan.start, chspan.end, label="ORG"))
            i = j
        else:
            i += 1
    return org_spans

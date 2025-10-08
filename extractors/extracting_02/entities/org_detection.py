from typing import List
from spacy.tokens import Span
from .constants import HEADER_STARTERS
from .normalization import canonical_org_key

def _is_all_caps_line(ln: str) -> bool:
    t = ln.strip()
    if not t: return False
    letters = [ch for ch in t if ch.isalpha()]
    if not letters: return False
    return all(ch == ch.upper() for ch in letters)

def _starts_with_starter(ln: str) -> bool:
    t = ln.strip()
    if not t: return False
    first = __import__("re").split(r'[\s\-–—:,;./]+', t, 1)[0]
    return first.upper() in HEADER_STARTERS

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

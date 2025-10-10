import re
import unicodedata
from typing import List, Iterable, Tuple
import spacy
from spacy.tokens import Doc, Span

def strip_diacritics(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")

def normalize_for_match(s: str) -> str:
    s = s.replace("\u00A0", " ")

    # Fix common line-break hyphenation patterns (hard + soft hyphen)
    s = s.replace("-\r\n", "").replace("-\n", "").replace("\u00AD\n", "").replace("\n00AD\r\n", "")
    # Also remove standalone soft hyphen if present
    s = s.replace("\u00AD", "")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = strip_diacritics(s).casefold()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def item_anchor_phrases(text: str) -> List[str]:
    """Pick distinctive anchors from item text (order matters)."""
    t = re.sub(r"\s+", " ", text.strip())
    anchors: List[str] = []
    if t:
        anchors.append(t[:120])  # leading chunk
    # cue patterns
    m = re.search(r"\b(Portaria n[ºo]\.? [^\s,;:.]+)", t, flags=re.IGNORECASE)
    if m: anchors.append(m.group(1))
    m = re.search(r"\b(CCT ?entre\b.+?)\b - ", t, flags=re.IGNORECASE)
    if m: anchors.append(m.group(1)[:140])
    m = re.search(r"\b(Aviso de Projecto de Portaria)\b", t, flags=re.IGNORECASE)
    if m: anchors.append(m.group(1))

    seen = set(); out = []
    for a in anchors:
        k = normalize_for_match(a)
        if k in seen: continue
        seen.add(k); out.append(a)
    return out

def sent_span_covering(doc: Doc, abs_start: int, abs_end: int) -> Tuple[int, int]:
    """Given absolute char span on the same text as doc, expand to sentence-ish bounds."""
    # find tokens hitting the char span
    tok_start = doc.char_span(abs_start, abs_start, alignment_mode="contract")
    tok_end   = doc.char_span(abs_end, abs_end, alignment_mode="contract")
    # fallback if None
    if tok_start is None:
        tok_start = doc.char_span(max(0, abs_start-1), abs_start, alignment_mode="expand") or doc[0:1]
    if tok_end is None:
        tok_end = doc.char_span(abs_end, min(len(doc.text), abs_end+1), alignment_mode="expand") or doc[-1:]
    s = tok_start.start
    e = max(tok_end.end, tok_start.end)

    # expand to sentence boundaries if possible
    left_i = doc[s].sent.start if doc[s].sent is not None else s
    right_j = doc[e-1].sent.end if doc[e-1].sent is not None else e
    span = doc[left_i:right_j]
    return (span.start_char, span.end_char)

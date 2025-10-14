# body_extraction/finders.py
from typing import Optional, Tuple, Dict, Any, List
import re
import unicodedata
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc
from .body_taxonomy import BODY_SECTIONS

FIRST_LINE_MAX_TOKENS = 12
SIG_CONTEXT_CHARS = 400
HEAD_LEN = 120


def _first_nonblank_line(s: str) -> str:
    for ln in s.splitlines():
        t = ln.strip()
        if t:
            return t
    return ""


def _build_firstline_pattern(nlp, item_text: str):
    first = _first_nonblank_line(item_text)
    if not first:
        return None
    doc = nlp.make_doc(first)
    toks = [t for t in doc if (t.is_alpha or t.is_digit or (t.text.isupper() and len(t.text) >= 2))]
    if not toks:
        toks = [t for t in doc if not t.is_space]
    toks = toks[:FIRST_LINE_MAX_TOKENS]
    if not toks:
        return None
    pat = []
    for i, t in enumerate(toks):
        if i > 0:
            pat.append({"IS_PUNCT": True, "OP": "*"})
        pat.append({"LOWER": t.text.lower()})
    return pat


def find_by_firstline(nlp, window_text: str, item_text: str) -> Optional[Tuple[int, str, str]]:
    """
    Return (start_rel, strategy, preview).
    """
    pat = _build_firstline_pattern(nlp, item_text)
    if not pat:
        return None
    doc: Doc = nlp.make_doc(window_text)
    m = Matcher(nlp.vocab)
    m.add("FIRSTLINE", [pat])
    hits = m(doc)
    if not hits:
        return None
    _, s, _e = min(hits, key=lambda h: doc[h[1]].idx)
    start_char = doc[s].idx
    return start_char, "first_line_matcher", window_text[start_char:start_char+60]


def _extract_keywords(item_text: str) -> List[str]:
    toks = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ]+|[A-Z]{2,}", item_text)
    toks = sorted(set(toks), key=lambda w: (-w.isupper(), -len(w), w.lower()))
    return toks[:8]


def find_by_signatures(nlp, window_text: str, item_text: str, section_key: str) -> Optional[Tuple[int, str, str]]:
    sec = BODY_SECTIONS.get(section_key)
    phrases = (sec.item_anchor_phrases if sec else None) or []
    if not phrases:
        return None
    doc: Doc = nlp.make_doc(window_text)
    pm = PhraseMatcher(nlp.vocab, attr="LOWER")
    for i, p in enumerate(phrases):
        pm.add(f"SIG_{i}", [nlp.make_doc(p)])
    hits = pm(doc)
    if not hits:
        return None

    kws = _extract_keywords(item_text)
    best = None
    best_cnt = -1
    for _, s, e in hits:
        start = doc[s].idx
        end = doc[e-1].idx + len(doc[e-1])
        L = max(0, start - SIG_CONTEXT_CHARS)
        R = min(len(window_text), end + SIG_CONTEXT_CHARS)
        ctx = window_text[L:R]
        cnt = sum(1 for kw in kws if re.search(rf"\b{re.escape(kw)}\b", ctx, flags=re.IGNORECASE))
        if cnt > best_cnt and cnt >= 1:
            best_cnt = cnt
            best = start
    if best is None:
        return None
    return best, "signature_phrasematcher", window_text[best:best+60]


def _strip_diacritics(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")


def _normalize_for_search(s: str) -> str:
    s = s.replace("\u00AD", "")
    s = re.sub(r"-\r?\n", "", s)
    s = s.replace("\u00A0", " ")
    s = _strip_diacritics(s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def find_by_head_substr(window_text: str, item_text: str) -> Optional[Tuple[int, str, str]]:
    head = " ".join(item_text.strip().split())[:HEAD_LEN]
    if not head:
        return None

    # raw regex with hyphen/whitespace tolerance
    esc = re.escape(head).replace(r"\ ", r"\s+").replace(r"\-", r"(?:-\s*|\s+)")
    rx = re.compile(esc, re.IGNORECASE)
    m = rx.search(window_text)
    if m:
        s = m.start()
        return s, "head_substr", window_text[s:s+60]

    # normalized fallback
    norm_win = _normalize_for_search(window_text)
    norm_head = _normalize_for_search(head)
    j = norm_win.find(norm_head)
    if j != -1:
        # approximate back to raw position
        approx_raw = max(0, int(j * max(1, len(window_text)) / max(1, len(norm_win))) - 50)
        return approx_raw, "head_substr_norm", window_text[approx_raw:approx_raw+60]

    return None

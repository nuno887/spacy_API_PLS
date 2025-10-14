# body_extraction/extract.py
from typing import List, Dict, Any, Optional, Tuple, FrozenSet
import re
import unicodedata
from .sections import build_windows_for_sections, pick_first_window_for_key

from spacy.matcher import Matcher, PhraseMatcher

from .sections import (
    index_global_headers_strict,
    build_section_windows_strict,
    pick_first_window_for_key,
)
from .body_taxonomy import BODY_SECTIONS


# =========================
# Small utils (kept minimal)
# =========================

FIRST_LINE_MAX_TOKENS = 12


def _snap_left_to_line_start(text: str, start: int, floor: int) -> int:
    rel = text.rfind("\n", floor, start)
    return floor if rel == -1 else rel + 1


def _first_nonblank_line(s: str) -> str:
    for ln in s.splitlines():
        t = ln.strip()
        if t:
            return t
    return ""


def _strip_diacritics(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")


def _normalize_for_search(s: str) -> str:
    s = s.replace("\u00AD", "")
    s = re.sub(r"-\r?\n", "", s)
    s = s.replace("\u00A0", " ")
    s = _strip_diacritics(s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


# =========================
# Title indexing via taxonomy
# =========================

def _build_title_matcher(nlp, section_key: str) -> Optional[Matcher]:
    sec = BODY_SECTIONS.get(section_key)
    if not sec or not sec.item_title_patterns:
        return None
    m = Matcher(nlp.vocab)
    for i, pat in enumerate(sec.item_title_patterns):
        if pat:
            m.add(f"TIT_{section_key}_{i}", [pat])
    return m


def _build_line_anchor_regex(title_line: str) -> re.Pattern:
    """
    Build a regex anchored at start-of-line (multiline) for the given title line,
    tolerant to hyphen+linebreak quirks and extra spaces.
    """
    # escape; then relax spaces & hyphens
    esc = re.escape(title_line.strip())
    esc = esc.replace(r"\ ", r"[ \t]+")                     # any run of spaces/tabs
    esc = esc.replace(r"\-", r"(?:-\s*|\s+)")               # hyphenated or wrapped
    return re.compile(rf"(?m)^\s*{esc}")


def _find_line_start_anchored(window_text: str, title_line: str) -> Optional[int]:
    """
    Return relative char index in window where a line starts with title_line;
    None if not found.
    """
    if not title_line:
        return None
    rx = _build_line_anchor_regex(title_line)
    m = rx.search(window_text)
    return None if not m else m.start()

def _build_signature_phrasematcher(nlp, section_key: str) -> Optional[PhraseMatcher]:
    sec = BODY_SECTIONS.get(section_key)
    phrases = (sec.item_anchor_phrases if sec else None) or []
    if not phrases:
        return None
    pm = PhraseMatcher(nlp.vocab, attr="LOWER")
    for i, p in enumerate(phrases):
        if p:
            pm.add(f"ANC_{section_key}_{i}", [nlp.make_doc(p)])
    return pm


def index_block_titles(nlp, window_text: str, section_key: str) -> List[int]:
    """
    Return sorted, de-duplicated relative positions (to window_text) where a section item title likely starts.
    We use taxonomy: spaCy Matcher over item_title_patterns + PhraseMatcher over item_anchor_phrases.
    Keep only hits at the beginning of a line (±2 chars).
    """
    doc = nlp.make_doc(window_text)
    starts: List[int] = []

    m = _build_title_matcher(nlp, section_key)
    if m:
        for _, s, _e in m(doc):
            starts.append(doc[s].idx)

    pm = _build_signature_phrasematcher(nlp, section_key)
    if pm:
        for _, s, _e in pm(doc):
            starts.append(doc[s].idx)

    if not starts:
        return []

    def _is_line_start(p: int) -> bool:
        return (p - (window_text.rfind("\n", 0, p) + 1)) <= 2

    starts = [p for p in starts if _is_line_start(p)]
    starts = sorted(starts)

    # Dedup by collapsing hits on same line
    dedup: List[int] = []
    last_line_start = None
    for pos in starts:
        line_start = window_text.rfind("\n", 0, pos) + 1
        if last_line_start is None or abs(line_start - last_line_start) > 2:
            dedup.append(pos)
            last_line_start = line_start

    return dedup


# =========================
# Strict per-item (first-line) matcher
# =========================

def _build_firstline_pattern(nlp, item_text: str) -> Optional[List[Dict[str, Any]]]:
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

    pat: List[Dict[str, Any]] = []
    for i, t in enumerate(toks):
        if i > 0:
            pat.append({"IS_PUNCT": True, "OP": "*"})
        pat.append({"LOWER": t.text.lower()})
    return pat


def _find_by_firstline(nlp, window_text: str, item_text: str) -> Optional[Tuple[int, str]]:
    pat = _build_firstline_pattern(nlp, item_text)
    if not pat:
        return None
    doc = nlp.make_doc(window_text)
    m = Matcher(nlp.vocab)
    m.add("FIRSTLINE", [pat])
    hits = m(doc)
    if not hits:
        return None
    _, s, _e = min(hits, key=lambda h: doc[h[1]].idx)
    start_char = doc[s].idx
    return start_char, window_text[start_char:start_char + 60]


# =========================
# MAIN (strict-only)
# =========================

def run_extraction(
    body_text: str,
    sections: List[Dict[str, Any]],
    relations_org_to_org: Optional[List[Dict[str, Any]]],  # kept for API compat; unused here
    nlp,
) -> Dict[str, Any]:
    """
    Starts-only extractor:
      1) Find section windows via global headers (sections.py)
      2) For each section window, anchor each item by its *first non-blank line*
         at start-of-line in the body (strict line-start match).
      3) End of an item = next item's start; last ends at window end.
      4) No fuzzy fallbacks. Items that don't anchor -> not_found (0-length span).
    """

    # ---- tiny local helpers ----
    def _first_nonblank_line(s: str) -> str:
        for ln in (s or "").splitlines():
            t = ln.strip()
            if t:
                # collapse internal whitespace; keep letters as-is
                return " ".join(t.split())
        return ""

    def _find_line_start_anchored(window_text: str, needle_line: str) -> Optional[int]:
        """
        Return relative char index where a line equal to `needle_line` (after whitespace collapse)
        starts in window_text. Anchored to start-of-line.
        """
        if not needle_line:
            return None
        # pre-normalize the needle (collapse spaces)
        needle_norm = " ".join(needle_line.split())
        # scan lines with their absolute positions in the window
        pos = 0
        for raw in window_text.splitlines(True):  # keep line breaks
            # raw includes the newline (except maybe last)
            ln = raw.rstrip("\r\n")
            ln_norm = " ".join(ln.strip().split())
            if ln_norm == needle_norm:
                return pos  # start of this line in window
            pos += len(raw)
        return None

    def _snap_left_to_line_start(text: str, start: int, floor: int) -> int:
        j = text.rfind("\n", floor, start)
        return floor if j == -1 else (j + 1)

    # ---- 1) Build section windows up-front ----
    windows_by_key: Dict[str, List[Tuple[int, int]]] = build_windows_for_sections(
        nlp=nlp,
        body_text=body_text,
        sections=sections,
        
    )

    results: List[Dict[str, Any]] = []

    # ---- 2) Main slicing loop (strict starts-only) ----
    for s in sections:
        sec_path = s.get("path", [])
        sec_key = sec_path[-1] if sec_path else "(unknown)"
        items = list(s.get("items", []))

        picked = pick_first_window_for_key(windows_by_key, sec_key)
        if not picked:
            # No header → nothing to slice. Emit not_found for all items.
            for it in items:
                results.append({
                    "section_path": sec_path,
                    "section_name": sec_key,
                    "section_span": s.get("span", {"start": 0, "end": 0}),
                    "item_text": it.get("text", ""),
                    "item_span_sumario": it.get("span"),
                    "org_context": s.get("org_context", {}),
                    "body_span": {"start": 0, "end": 0},
                    "confidence": 0.0,
                    "method": "not_found",
                    "diagnostics": {"reason": "no_header_window"},
                })
            continue

        win_start, win_end = picked
        win_start = max(0, min(win_start, len(body_text)))
        win_end   = max(win_start, min(win_end, len(body_text)))
        win_text  = body_text[win_start:win_end]

        # Collect (absolute_start, item_idx) via strict line-start anchoring
        starts: List[Tuple[int, int]] = []
        for idx, it in enumerate(items):
            title_line = _first_nonblank_line(it.get("text", "") or "")
            rel = _find_line_start_anchored(win_text, title_line)
            if rel is not None:
                starts.append((win_start + rel, idx))

        # Sort starts and compute end as next start (last ends at window end)
        starts.sort(key=lambda t: t[0])
        start_pos_by_idx: Dict[int, int] = {idx: pos for (pos, idx) in starts}

        for i, (s_abs_raw, idx) in enumerate(starts):
            right = starts[i + 1][0] if i + 1 < len(starts) else win_end
            s_abs_raw = max(win_start, min(win_end, int(s_abs_raw)))
            e_abs_raw = max(s_abs_raw, min(win_end, int(right)))

            # snap left edge to the true line start within the window
            s_abs = _snap_left_to_line_start(body_text, s_abs_raw, win_start)
            e_abs = max(s_abs, e_abs_raw)

            results.append({
                "section_path": sec_path,
                "section_name": sec_key,
                "section_span": s.get("span", {"start": 0, "end": 0}),
                "item_text": items[idx].get("text", ""),
                "item_span_sumario": items[idx].get("span"),
                "org_context": s.get("org_context", {}),
                "body_span": {"start": s_abs, "end": e_abs},
                "confidence": 0.85,
                "method": "title_match",
                "diagnostics": {"strategy": "line_start_anchor"},
            })

        # Emit "not found" for items that didn't anchor
        missing = [i for i in range(len(items)) if i not in start_pos_by_idx]
        for idx in missing:
            results.append({
                "section_path": sec_path,
                "section_name": sec_key,
                "section_span": s.get("span", {"start": 0, "end": 0}),
                "item_text": items[idx].get("text", ""),
                "item_span_sumario": items[idx].get("span"),
                "org_context": s.get("org_context", {}),
                "body_span": {"start": win_start, "end": win_start},  # zero-length
                "confidence": 0.0,
                "method": "not_found",
                "diagnostics": {"reason": "no_line_anchor"},
            })

    # ---- 3) Summary ----
    summary = {
        "found": sum(1 for r in results if r["method"] not in ("not_found",)),
        "not_found": sum(1 for r in results if r["method"] in ("not_found",)),
        "avg_confidence": round(sum(r["confidence"] for r in results) / max(1, len(results)), 4),
    }
    return {"summary": summary, "results": results}

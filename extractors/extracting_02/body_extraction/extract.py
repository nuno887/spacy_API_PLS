# body_extraction/extract.py
from typing import List, Dict, Any, Optional, Tuple, FrozenSet
import re
import unicodedata
from .sections import build_windows_for_sections, pick_first_window_for_key
from .debug_print import dbg_dump_window_lines, dbg_print_spacy_title_hits
from spacy.matcher import Matcher, PhraseMatcher
from .titles_bol import (
    starts_from_item_prefixes,
    augment_with_spacy_bol_hits,
    augment_with_firstline_bol,
    slice_by_sorted_starts,
)
from .title_index import index_block_titles, section_title_prefixes_from_items



from .sections import (
    index_global_headers_strict,
    build_section_windows_strict,
    pick_first_window_for_key,
)
from .body_taxonomy import BODY_SECTIONS

from .debug_tools import (
    dbg_header_windows,
    dbg_window_first_lines,
    dbg_scan_lines_with_prefixes,
    dbg_spacy_title_hits,
    dbg_item_alignment,
)




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
    relations_org_to_org: Optional[List[Dict[str, Any]]],  # kept for signature compatibility; unused
    nlp,
) -> Dict[str, Any]:
    """
    Simpler pipeline:
      1) Build global windows per section via header detection (spaCy PhraseMatcher).
      2) For each section window:
         a) Generate title prefixes from Sumário items.
         b) Detect titles using BOTH:
            - taxonomy-driven spaCy matcher (index_block_titles)
            - fast prefix scan (starts-with) using the generated prefixes
         c) Merge hits → sort & de-dup at line-start.
         d) If count == items, slice [start .. next_start) per item (high confidence).
         e) Otherwise, consume detected starts in order; if we still need starts,
            try a light "first-line" matcher per missing item. Emit medium confidence.
    """
    results: List[Dict[str, Any]] = []

    # 1) windows per section (based solely on body headers)
    windows_by_key: Dict[str, List[Tuple[int, int]]] = build_windows_for_sections(nlp, body_text, sections)
    dbg_header_windows(body_text, windows_by_key, wanted_keys=[k for k in windows_by_key.keys()])

    def pick_first_window_for_key(wmap: Dict[str, List[Tuple[int, int]]], key: str) -> Optional[Tuple[int, int]]:
        lst = wmap.get(key) or []
        return lst[0] if lst else None

    for s in sections:
        sec_path = s.get("path", [])
        sec_key = (sec_path[-1] if sec_path else "(unknown)")
        items = list(s.get("items", []))

        picked = pick_first_window_for_key(windows_by_key, sec_key)
        if not picked:
            # no header window → emit not_found for all items
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
        win_end = max(win_start, min(win_end, len(body_text)))
        win_text = body_text[win_start:win_end]

        # --- debug: quick look at the window
        dbg_window_first_lines(sec_key, body_text, win_start, win_end)

        # 2a) generate item-derived prefixes (robust to variety in titles)
        prefixes = section_title_prefixes_from_items(items, tokens=6)
        dbg_scan_lines_with_prefixes(body_text, win_start, win_end, prefixes)

        # 2b) titles from taxonomy (spaCy) within the window
        spacy_rel_starts = index_block_titles(nlp, win_text, sec_key)
        spacy_abs_starts = [win_start + r for r in spacy_rel_starts]

        # 2c) titles from prefix scan (fast line-by-line)
        prefix_abs_starts: List[int] = []
        cur = win_start
        while cur < win_end:
            nl = body_text.find("\n", cur, win_end)
            line_end = nl if nl != -1 else win_end
            raw = body_text[cur:line_end]
            norm = " ".join(raw.strip().split())
            if norm:
                for p in prefixes:
                    if norm.lower().startswith(p.lower()):
                        prefix_abs_starts.append(cur)
                        break
            if nl == -1:
                break
            cur = nl + 1

        # 2d) merge hits → line starts only → sort + de-dup (one per line)
        merged_starts = sorted(set(spacy_abs_starts + prefix_abs_starts))
        # collapse to the beginning of each line
        normed_starts: List[int] = []
        last_line_start = None
        for pos in merged_starts:
            line_start = body_text.rfind("\n", win_start, pos) + 1
            if (last_line_start is None) or (line_start - last_line_start > 2):
                normed_starts.append(line_start)
                last_line_start = line_start

        dbg_spacy_title_hits(sec_key, normed_starts, body_text)

        # 3) perfect case: 1:1
        if normed_starts and len(normed_starts) == len(items):
            for i, ts in enumerate(normed_starts):
                s_abs_raw = max(win_start, min(win_end, ts))
                e_abs = normed_starts[i + 1] if i + 1 < len(normed_starts) else win_end
                e_abs = max(s_abs_raw, min(win_end, e_abs))
                s_abs = _snap_left_to_line_start(body_text, s_abs_raw, win_start)
                if e_abs < s_abs:
                    e_abs = s_abs

                results.append({
                    "section_path": sec_path,
                    "section_name": sec_key,
                    "section_span": s.get("span", {"start": 0, "end": 0}),
                    "item_text": items[i].get("text", ""),
                    "item_span_sumario": items[i].get("span"),
                    "org_context": s.get("org_context", {}),
                    "body_span": {"start": s_abs, "end": e_abs},
                    "confidence": 0.85,
                    "method": "title_match",
                    "diagnostics": {
                        "titles_indexed": len(normed_starts),
                        "sources": {
                            "spacy": len(spacy_abs_starts),
                            "prefix": len(prefix_abs_starts),
                        },
                    },
                })
            dbg_item_alignment(sec_key, items, normed_starts, body_text)
            continue  # next section

        # 4) mixed/low-signal: consume detected starts, then try first-line
        occupied: List[int] = list(normed_starts)
        chosen_starts_abs: List[int] = []

        for it in items:
            chosen = None

            if occupied:
                chosen = occupied.pop(0)
            else:
                # light fallback: first-line pattern on this window
                hit = _find_by_firstline(nlp, win_text, it.get("text", "") or "")
                if hit:
                    chosen = win_start + hit[0]

            if chosen is not None:
                chosen_starts_abs.append(chosen)
                # slice
                e_abs = win_end
                future = [p for p in occupied if p > chosen]
                if future:
                    e_abs = min(e_abs, min(future))
                s_abs_raw = max(win_start, min(win_end, chosen))
                e_abs = max(s_abs_raw, min(win_end, e_abs))
                s_abs = _snap_left_to_line_start(body_text, s_abs_raw, win_start)
                if e_abs < s_abs:
                    e_abs = s_abs

                results.append({
                    "section_path": sec_path,
                    "section_name": sec_key,
                    "section_span": s.get("span", {"start": 0, "end": 0}),
                    "item_text": it.get("text", ""),
                    "item_span_sumario": it.get("span"),
                    "org_context": s.get("org_context", {}),
                    "body_span": {"start": s_abs, "end": e_abs},
                    "confidence": 0.70 if normed_starts else 0.55,
                    "method": "title_or_firstline_mix" if normed_starts else "firstline_match",
                    "diagnostics": {
                        "titles_indexed": len(normed_starts),
                        "sources": {
                            "spacy": len(spacy_abs_starts),
                            "prefix": len(prefix_abs_starts),
                        },
                    },
                })
            else:
                # still nothing
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
                    "diagnostics": {
                        "reason": "no_title_or_firstline_match",
                        "titles_indexed": len(normed_starts),
                        "sources": {
                            "spacy": len(spacy_abs_starts),
                            "prefix": len(prefix_abs_starts),
                        },
                    },
                })

        dbg_item_alignment(sec_key, items, chosen_starts_abs, body_text)

    summary = {
        "found": sum(1 for r in results if r["method"] not in ("not_found",)),
        "not_found": sum(1 for r in results if r["method"] in ("not_found",)),
        "avg_confidence": round(sum(r["confidence"] for r in results) / max(1, len(results)), 4),
    }
    return {"summary": summary, "results": results}

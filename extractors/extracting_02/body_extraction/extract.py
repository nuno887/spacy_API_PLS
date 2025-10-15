# body_extraction/extract.py
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple, Set
import re

from spacy.matcher import Matcher

from .sections import build_windows_for_sections, pick_first_window_for_key
from .debug_tools import (
    dbg_window_first_lines,
    dbg_scan_lines_with_prefixes,
    dbg_spacy_title_hits,
    dbg_item_alignment,
)

from .title_index import index_block_titles, section_title_prefixes_from_items
from .body_taxonomy import BODY_SECTIONS
from .text_norm import normalize_for_search

# =========================
# Small utils
# =========================

FIRST_LINE_MAX_TOKENS = 12

def _first_nonblank_line(s: str) -> str:
    for ln in s.splitlines():
        t = ln.strip()
        if t:
            return t
    return ""

def _snap_left_to_line_start(text: str, start: int, floor: int) -> int:
    """Snap `start` to the physical line start at/above it, but not before `floor`."""
    rel = text.rfind("\n", floor, start)
    return floor if rel == -1 else rel + 1

def _iter_lines_with_offsets(text: str, start: int, end: int):
    """Yield (line_start_abs, line_end_abs, raw_line) for non-empty lines in [start, end)."""
    cur = start
    while cur < end:
        nl = text.find("\n", cur, end)
        line_end = nl if nl != -1 else end
        raw = text[cur:line_end]
        yield cur, line_end, raw
        if nl == -1:
            break
        cur = nl + 1

def _build_firstline_pattern(nlp, item_text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Build a Matcher pattern from the item's first nonblank line, capped to FIRST_LINE_MAX_TOKENS.
    Used ONLY for comparison, never to extract offsets directly.
    """
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
            # allow punctuation between tokens
            pat.append({"IS_PUNCT": True, "OP": "*"})
        pat.append({"LOWER": t.text.lower()})
    return pat

def _anchored_firstline_match(nlp, window_text: str, item_text: str, win_abs_start: int) -> Optional[int]:
    """
    Strict fallback: accept ONLY if the first-line pattern matches at the start of a physical line.
    Returns absolute start index for the matched line (snapped to the true line start) or None.
    """
    pat = _build_firstline_pattern(nlp, item_text)
    if not pat:
        return None

    # Prepare a matcher over each physical line (start-anchored check)
    # Normalize both sides for robust comparison, but maintain offsets from raw lines.
    for line_abs_start, line_abs_end, raw_line in _iter_lines_with_offsets(window_text, 0, len(window_text)):
        # Allow up to two leading spaces/tabs before content
        lstrip_len = len(raw_line) - len(raw_line.lstrip(" \t"))
        if lstrip_len > 2:
            # too much indentation → treat as not a header-like line
            continue

        # Create a doc for this raw line
        doc_line = nlp.make_doc(raw_line)
        m = Matcher(nlp.vocab)
        m.add("FIRSTLINE", [pat])
        hits = m(doc_line)
        if not hits:
            continue

        # Ensure the match starts at (or near) the beginning of this line
        _, start_tok, _ = min(hits, key=lambda h: doc_line[h[1]].idx)
        match_char = doc_line[start_tok].idx
        if match_char > 2:  # must be anchored near start
            continue

        # Accept: snap to true physical line start (abs in body text known as: win_abs_start + line_abs_start)
        return win_abs_start + line_abs_start

    return None

# =========================
# MAIN
# =========================

def run_extraction(
    body_text: str,
    sections: List[Dict[str, Any]],
    relations_org_to_org: Optional[List[Dict[str, Any]]],  # kept for signature compatibility; unused
    nlp,
) -> Dict[str, Any]:
    """
    Pipeline:
      1) Build global windows per section via header detection (strict, start-of-line).
      2) For each section window (best window chosen among candidates):
         a) Generate title prefixes from Sumário items (normalized).
         b) Detect titles using taxonomy-driven spaCy matcher.
         c) Detect titles using normalized prefix scan at line-start.
         d) Merge hits with provenance → line-start de-dup → ordered sequence.
         e) If 1:1 with items, slice spans at [start .. next_start).
         f) Otherwise, fill gaps with STRICT anchored first-line matches, merge, re-slice.
    """
    all_results: List[Dict[str, Any]] = []

    # 1) windows per section (based solely on body headers)
    windows_by_key: Dict[str, List[Tuple[int, int]]] = build_windows_for_sections(nlp, body_text, sections)
    # (Existing debug helpers are called from the caller; we keep function quiet here.)

    # Helper: run alignment inside a single (start, end) window; return results + stats
    def align_in_window(sec_rec: Dict[str, Any], win_start: int, win_end: int):
        sec_path = sec_rec.get("path", [])
        sec_key = (sec_path[-1] if sec_path else "(unknown)")
        items = list(sec_rec.get("items", []))

        win_start = max(0, min(win_start, len(body_text)))
        win_end = max(win_start, min(win_end, len(body_text)))
        win_text = body_text[win_start:win_end]

        # --- debug: quick look at the window
        dbg_window_first_lines(sec_key, body_text, win_start, win_end)

        # 2a) generate item-derived prefixes (robust)
        prefixes = section_title_prefixes_from_items(items, tokens=6)
        dbg_scan_lines_with_prefixes(body_text, win_start, win_end, prefixes)

        # 2b) titles from taxonomy (spaCy) within the window (RELATIVE → ABSOLUTE)
        spacy_rel_starts = index_block_titles(nlp, win_text, sec_key)
        spacy_abs_starts = [win_start + r for r in spacy_rel_starts]

        # 2c) titles from prefix scan (fast line-by-line, normalized comparison at line start)
        prefix_abs_starts: List[int] = []
        if prefixes:
            for line_start, line_end, raw in _iter_lines_with_offsets(body_text, win_start, win_end):
                raw_trim = raw.strip()
                if not raw_trim:
                    continue
                norm = normalize_for_search(raw_trim)
                if not norm:
                    continue
                for p in prefixes:
                    if norm.startswith(p):
                        prefix_abs_starts.append(line_start)
                        break

        # ---- Merge hits with provenance per physical line
        # Normalize to true physical line starts within the window
        merged_map: Dict[int, Set[str]] = {}
        for pos in spacy_abs_starts:
            line_start = body_text.rfind("\n", win_start, pos) + 1
            merged_map.setdefault(line_start, set()).add("spacy")
        for pos in prefix_abs_starts:
            merged_map.setdefault(pos, set()).add("prefix")

        # If counts fall short, try STRICT anchored first-line per missing item
        chosen_line_starts = sorted(merged_map.keys())
        # Build a fast lookup of already used starts to avoid duplicates
        used_lines: Set[int] = set(chosen_line_starts)

        if len(chosen_line_starts) < len(items):
            for it in items:
                hit = _anchored_firstline_match(nlp, win_text, it.get("text", "") or "", win_start)
                if hit is not None and hit not in used_lines and (win_start <= hit < win_end):
                    merged_map.setdefault(hit, set()).add("firstline")
                    used_lines.add(hit)
                    if len(used_lines) >= len(items):
                        break

        # Final ordered starts (one per physical line)
        ordered_starts = sorted(merged_map.keys())
        extra_starts = max(0, len(ordered_starts) - len(items))

        dbg_spacy_title_hits(sec_key, ordered_starts, body_text)

        # Slicing: map ordered starts to items in order
        results: List[Dict[str, Any]] = []
        for i, it in enumerate(items):
            if i < len(ordered_starts):
                s_abs_raw = max(win_start, min(win_end, ordered_starts[i]))
                e_abs = ordered_starts[i + 1] if i + 1 < len(ordered_starts) else win_end
                e_abs = max(s_abs_raw, min(win_end, e_abs))
                s_abs = _snap_left_to_line_start(body_text, s_abs_raw, win_start)
                if e_abs < s_abs:
                    e_abs = s_abs

                sources = sorted(merged_map.get(ordered_starts[i], set()))
                method = (
                    "title_match" if ("spacy" in sources or "prefix" in sources)
                    else "firstline_match"
                )
                conf = 0.85 if method == "title_match" else 0.55
                if method != "title_match" and ("spacy" in sources or "prefix" in sources):
                    conf = 0.70  # mixed path

                results.append({
                    "section_path": sec_path,
                    "section_name": sec_key,
                    "section_span": sec_rec.get("span", {"start": 0, "end": 0}),
                    "item_text": it.get("text", ""),
                    "item_span_sumario": it.get("span"),
                    "org_context": sec_rec.get("org_context", {}),
                    "body_span": {"start": s_abs, "end": e_abs},
                    "confidence": conf,
                    "method": "title_match" if method == "title_match" else ("firstline_match"),
                    "diagnostics": {
                        "titles_indexed": len(ordered_starts),
                        "method_sources": sources,  # provenance on this line
                        "sources": {
                            "spacy": sum(1 for v in merged_map.values() if "spacy" in v),
                            "prefix": sum(1 for v in merged_map.values() if "prefix" in v),
                            "firstline": sum(1 for v in merged_map.values() if "firstline" in v),
                        },
                    },
                })
            else:
                # no candidate line left for this item
                results.append({
                    "section_path": sec_path,
                    "section_name": sec_key,
                    "section_span": sec_rec.get("span", {"start": 0, "end": 0}),
                    "item_text": it.get("text", ""),
                    "item_span_sumario": it.get("span"),
                    "org_context": sec_rec.get("org_context", {}),
                    "body_span": {"start": 0, "end": 0},
                    "confidence": 0.0,
                    "method": "not_found",
                    "diagnostics": {
                        "reason": "no_candidate_line",
                        "titles_indexed": len(ordered_starts),
                        "extra_starts_in_window": extra_starts,
                    },
                })
        
        starts_for_dbg = ordered_starts[:len(items)]
        dbg_item_alignment(sec_key, items, starts_for_dbg, body_text)

        # stats for window quality (fraction of items aligned)
        found = sum(1 for r in results if r["method"] != "not_found")
        return results, found, len(items)

    # 2) Iterate sections and choose best window when multiple exist
    for s in sections:
        sec_path = s.get("path", [])
        sec_key = (sec_path[-1] if sec_path else "(unknown)")
        items = list(s.get("items", []))
        wins = windows_by_key.get(sec_key) or []

        # If no window: emit not_found items
        if not wins:
            for it in items:
                all_results.append({
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

        # Try windows in order; pick the first with acceptable quality (>=50% items found)
        best_results: Optional[List[Dict[str, Any]]] = None
        threshold = max(1, int(0.5 * len(items)))  # 50%
        for idx, (ws, we) in enumerate(wins):
            res, found, total = align_in_window(s, ws, we)
            if best_results is None:
                best_results = res
            # annotate chosen window index in diagnostics
            for r in res:
                r.setdefault("diagnostics", {})["chosen_window_index"] = idx
            if found >= threshold:
                best_results = res
                break  # pick this window

        all_results.extend(best_results or [])

    # Summary + simple section-level quality marker (ok/low)
    total = len(all_results)
    found = sum(1 for r in all_results if r["method"] != "not_found")
    not_found = total - found
    avg_conf = round(sum(r["confidence"] for r in all_results) / max(1, total), 4)

    # Build per-section quality (optional; lightweight)
    section_found: Dict[Tuple[str, Tuple[str, ...]], int] = {}
    section_total: Dict[Tuple[str, Tuple[str, ...]], int] = {}
    for r in all_results:
        key = (r.get("section_name", ""), tuple(r.get("section_path", [])))
        section_total[key] = section_total.get(key, 0) + 1
        if r["method"] != "not_found":
            section_found[key] = section_found.get(key, 0) + 1
    section_quality: Dict[str, str] = {}
    for (name, path), tot in section_total.items():
        fnd = section_found.get((name, path), 0)
        section_quality[" / ".join(path) or name] = "ok" if (tot and (fnd / tot) >= 0.3) else "low"

    return {
        "summary": {
            "found": found,
            "not_found": not_found,
            "avg_confidence": avg_conf,
            "section_quality": section_quality,
            "version": "1.0.0-cleanup1",
        },
        "results": all_results,
    }

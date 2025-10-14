# body_extraction/extrat.py (refactored)
from typing import List, Dict, Any, Optional, Tuple
import re
import unicodedata

from spacy.matcher import Matcher, PhraseMatcher

from .anchors import build_body_org_bands_from_relations, pick_band_for_section
from .sections import build_section_blocks_in_band, find_section_block_in_band_spacy
from .body_taxonomy import BODY_SECTIONS
from .matchers import get_header_matcher

# =========================
# DEBUG SWITCHES
# =========================
DEBUG = True
DEBUG_SECTIONS = {"PortariasExtensao"}   # e.g., {"PortariasExtensao", "Convencoes"} or set() for all
DEBUG_MAX_CONTEXT = 240                  # chars of context around hits

def _dbg_enabled(sec_key: str) -> bool:
    return DEBUG and (not DEBUG_SECTIONS or sec_key in DEBUG_SECTIONS)

def _dbg_print_block_header(sec_key: str, win_start: int, win_end: int, body_text: str):
    if not _dbg_enabled(sec_key): return
    print(f"\n=== [DBG] SECTION WINDOW [{sec_key}] @{win_start}..{win_end} (len={win_end-win_start}) ===")
    head = body_text[win_start: min(win_end, win_start + 200)]
    first_line = (head.splitlines() or [""])[0]
    print(f"[DBG] window first line: {first_line!r}")

def _dbg_print_indexed_titles(sec_key: str, win_start: int, window_text: str, title_starts_rel: List[int]):
    if not _dbg_enabled(sec_key): return
    print(f"[DBG] indexed titles: {len(title_starts_rel)} hit(s)")
    for i, rel in enumerate(sorted(title_starts_rel)):
        abs_pos = win_start + rel
        line_start = window_text.rfind("\n", 0, rel) + 1
        line_end = window_text.find("\n", rel)
        if line_end == -1: line_end = len(window_text)
        line = window_text[line_start: line_end].strip()
        preview = line if len(line) <= 160 else (line[:157] + "...")
        print(f"  - [{i:02d}] @{abs_pos}  line: {preview!r}")

def _dbg_print_items(sec_key: str, items: List[dict]):
    if not _dbg_enabled(sec_key): return
    print(f"[DBG] sumário items: {len(items)}")
    for i, it in enumerate(items):
        first = next((ln.strip() for ln in (it.get('text') or '').splitlines() if ln.strip()), "")
        if len(first) > 160: first = first[:157] + "..."
        print(f"  - item[{i:02d}]: {first!r}")

def _dbg_print_found_starts(sec_key: str, starts_abs: List[Tuple[int,int,str, Optional[str]]], body_text: str):
    if not _dbg_enabled(sec_key): return
    print(f"[DBG] per-item starts found: {len(starts_abs)}")
    for abs_pos, idx, strat, _prev in sorted(starts_abs, key=lambda t: t[0]):
        L = max(0, abs_pos - DEBUG_MAX_CONTEXT//2)
        R = min(len(body_text), abs_pos + DEBUG_MAX_CONTEXT//2)
        ctx = body_text[L:R].replace("\n", " ").strip()
        if len(ctx) > DEBUG_MAX_CONTEXT: ctx = ctx[:DEBUG_MAX_CONTEXT-3] + "..."
        print(f"  - item[{idx:02d}] @{abs_pos} via {strat}: ctx='{ctx}'")

def _dbg_print_fallback_clamp(sec_key: str, win_start: int, win_end: int, clamp_end: Optional[int]):
    if not _dbg_enabled(sec_key): return
    if clamp_end is None:
        print(f"[DBG] fallback clamp: no next section header found → using window end @{win_end}")
    else:
        print(f"[DBG] fallback clamp: next section header at @{clamp_end} → fallback end clamped")

def _dbg_print_window_probe(sec_key: str, win_start: int, win_end: int, body_text: str):
    if not _dbg_enabled(sec_key):
        return

    window_text = body_text[win_start:win_end]
    print("[DBG] --- window probe ---")

    # A) context around win_start
    L = max(0, win_start - 120)
    R = min(len(body_text), win_start + 200)
    ctx = body_text[L:R].replace("\n", " ")
    if len(ctx) > 320: ctx = ctx[:317] + "..."
    print(f"[DBG] win_start context @{win_start}: “{ctx}”")

    # B) first 3 non-blank lines in the window
    lines = [ln.strip() for ln in window_text.splitlines() if ln.strip()]
    print(f"[DBG] first non-blank lines in window ({min(3, len(lines))} shown):")
    for i, ln in enumerate(lines[:3]):
        if len(ln) > 160: ln = ln[:157] + "..."
        print(f"      {i+1:>2}. {ln!r}")

    # C) literal header probes inside the window
    probes = ["Portarias de Extensão:", "Portarias de Extensao:"]
    for p in probes:
        j = window_text.find(p)
        if j != -1:
            print(f"[DBG] literal header probe found: {p!r} at window offset {j} (abs @{win_start + j})")
        else:
            print(f"[DBG] literal header probe NOT found: {p!r}")

    # D) show taxonomy header aliases for this section
    sec = BODY_SECTIONS.get(sec_key)
    aliases = (sec.header_aliases if sec else []) or []
    print(f"[DBG] taxonomy header_aliases ({len(aliases)}): {aliases}")

def _dbg_print_realign(sec_key: str, reason: str, old_start: int, new_start: int, body_text: str, win_end: int):
    if not _dbg_enabled(sec_key): return
    print(f"[DBG] realign: {reason}  start {old_start} → {new_start}")
    L = max(0, new_start - 120); R = min(len(body_text), new_start + 200)
    ctx = body_text[L:R].replace("\n", " ")
    if len(ctx) > 320: ctx = ctx[:317] + "..."
    print(f"[DBG] realign new-start context @{new_start}: “{ctx}”")
    win_text = body_text[new_start:win_end]
    lines = [ln.strip() for ln in win_text.splitlines() if ln.strip()]
    print(f"[DBG] realign first lines ({min(3, len(lines))} shown):")
    for i, ln in enumerate(lines[:3]):
        print(f"      {i+1:>2}. {(ln[:157] + '...') if len(ln) > 160 else ln!r}")

# =========================
# Small utils
# =========================

FIRST_LINE_MAX_TOKENS = 12   # first-line tokens used for per-item matcher
SIG_CONTEXT_CHARS     = 400  # +/- chars for keyword neighborhood
HEAD_LEN              = 120  # length for head-substring fallback

def _first_nonblank_line(s: str) -> str:
    for ln in s.splitlines():
        t = ln.strip()
        if t:
            return t
    return ""

def _snap_left_to_line_start(text: str, start: int, floor: int) -> int:
    rel = text.rfind("\n", floor, start)
    return floor if rel == -1 else rel + 1

def _strip_diacritics(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")

def _normalize_for_search(s: str) -> str:
    s = s.replace("\u00AD", "")
    s = re.sub(r"-\r?\n", "", s)
    s = s.replace("\u00A0", " ")
    s = _strip_diacritics(s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _extract_keywords(item_text: str) -> List[str]:
    toks = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ]+|[A-Z]{2,}", item_text)
    toks = sorted(set(toks), key=lambda w: (-w.isupper(), -len(w), w.lower()))
    return toks[:8]

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
    Return sorted, de-duplicated relative positions (to window_text)
    where a section item title likely starts, using taxonomy:
      - spaCy Matcher over item_title_patterns
      - plus PhraseMatcher over item_anchor_phrases
    We keep only hits at the beginning of a line (±2 chars).
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

    # 1) keep only hits at (or very near) line start
    def _is_line_start(p: int) -> bool:
        return (p - (window_text.rfind("\n", 0, p) + 1)) <= 2

    starts = [p for p in starts if _is_line_start(p)]

    # 2) sort + dedup by collapsing hits on the same line
    starts = sorted(starts)
    dedup: List[int] = []
    last_line_start = None
    for pos in starts:
        line_start = window_text.rfind("\n", 0, pos) + 1
        if last_line_start is None or abs(line_start - last_line_start) > 2:
            dedup.append(pos)
            last_line_start = line_start

    return dedup

# =========================
# Lightweight per-item fallback (still simple)
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
    m = Matcher(nlp.vocab); m.add("FIRSTLINE", [pat])
    hits = m(doc)
    if not hits:
        return None
    _, s, _e = min(hits, key=lambda h: doc[h[1]].idx)
    start_char = doc[s].idx
    return start_char, window_text[start_char:start_char+60]

def _find_by_signatures(nlp, window_text: str, item_text: str, section_key: str) -> Optional[Tuple[int, str]]:
    sec = BODY_SECTIONS.get(section_key)
    phrases = (sec.item_anchor_phrases if sec else None) or []
    if not phrases:
        return None
    doc = nlp.make_doc(window_text)
    pm = PhraseMatcher(nlp.vocab, attr="LOWER")
    for i, p in enumerate(phrases):
        pm.add(f"SIG_{i}", [nlp.make_doc(p)])
    hits = pm(doc)
    if not hits:
        return None
    kws = _extract_keywords(item_text)
    best = None; best_cnt = -1
    for _, s, e in hits:
        start = doc[s].idx; end = doc[e-1].idx + len(doc[e-1])
        L = max(0, start - SIG_CONTEXT_CHARS); R = min(len(window_text), end + SIG_CONTEXT_CHARS)
        ctx = window_text[L:R]
        cnt = sum(1 for kw in kws if re.search(rf"\b{re.escape(kw)}\b", ctx, flags=re.IGNORECASE))
        if cnt > best_cnt and cnt >= 1:
            best_cnt = cnt; best = start
    if best is None:
        return None
    return best, window_text[best:best+60]

def _find_by_head_fallback(window_text: str, item_text: str) -> Optional[Tuple[int, str, str]]:
    head = " ".join(item_text.strip().split())[:HEAD_LEN]
    if not head:
        return None
    esc = re.escape(head).replace(r"\ ", r"\s+").replace(r"\-", r"(?:-\s*|\s+)")
    rx = re.compile(esc, re.IGNORECASE)
    m = rx.search(window_text)
    if m:
        s = m.start()
        return s, "head_substr", window_text[s:s+60]
    norm_win = _normalize_for_search(window_text)
    norm_head = _normalize_for_search(head)
    j = norm_win.find(norm_head)
    if j != -1:
        approx_raw = max(0, int(j * max(1, len(window_text)) / max(1, len(norm_win))) - 50)
        return approx_raw, "head_substr_norm", window_text[approx_raw:approx_raw+60]
    return None

# =========================
# Clamp fallback to next header
# =========================

def _find_next_section_cut(nlp, body_text: str, win_start: int, win_end: int, section_keys_in_org: List[str]) -> Optional[int]:
    """
    Find the earliest header occurrence > win_start (but < win_end) among the given section keys.
    Returns the absolute char index or None.
    """
    text = body_text[win_start:win_end]
    doc = nlp.make_doc(text)
    pm = get_header_matcher(nlp, section_keys_in_org)
    best = None
    for _, s, _e in pm(doc):
        pos = win_start + doc[s].idx
        if pos > win_start and (best is None or pos < best):
            best = pos
    return best

# =========================
# MAIN API: Start→NextStart with title indexing + realignment
# =========================

def run_extraction(
    body_text: str,
    sections: List[Dict[str, Any]],
    relations_org_to_org: Optional[List[Dict[str, Any]]],
    nlp,
    body_offset: int = 0,
) -> Dict[str, Any]:
    """
    1) ORG band + section window
    2) realign window to section header or first title if needed
    3) index ALL titles in window via taxonomy (Matcher + PhraseMatcher)
    4) if count == items, map 1:1 and done
    5) else, fallback to tiny per-item finder
    - In all cases, snap left edge to line start; clamp fallbacks to next header.
    """
    body_bands = build_body_org_bands_from_relations(len(body_text), relations_org_to_org or [], body_offset)
    results: List[Dict[str, Any]] = []

    for s in sections:
        sec_path = s.get("path", []); sec_key = sec_path[-1] if sec_path else "(unknown)"
        sec_span = s.get("span", {"start": 0, "end": 0})

        default_band = (0, len(body_text), {"surface_raw": "", "span": {"start": 0, "end": len(body_text)}})
        a, z, band_meta = pick_band_for_section(body_bands, int(sec_span.get("start", 0)) - body_offset, default_band)

        section_keys_in_this_org = list({sec2["path"][-1] for sec2 in sections if sec2.get("path")})
        blocks = build_section_blocks_in_band(nlp, body_text, a, z, section_keys_in_this_org)
        win_start, win_end = (blocks[sec_key] if sec_key in blocks else (a, z))
        win_text = body_text[win_start:win_end]
        items = list(s.get("items", []))
        next_cut = _find_next_section_cut(nlp, body_text, win_start, z, section_keys_in_this_org)
        if next_cut is not None:
            old_end = win_end
            win_end = min(win_end, next_cut)
            win_text = body_text[win_start:win_end]
            if _dbg_enabled(sec_key):
                print(f"[DBG] tighten window end: {old_end} → {win_end} (next header @{next_cut})")

        # --- REALIGN WINDOW IF WE STARTED MID-LINE OR MISSED THE HEADER/TITLE ---
        def _first_nonblank(txt: str) -> str:
            for ln in txt.splitlines():
                t = ln.strip()
                if t:
                    return t
            return ""

        def _looks_like_section_header(sec_key_local: str, line: str) -> bool:
            sec = BODY_SECTIONS.get(sec_key_local)
            if not sec:
                return False
            cand = (line or "").rstrip(":").strip().lower()
            aliases = [h.rstrip(":").strip().lower() for h in (sec.header_aliases or [])]
            return cand in aliases or cand.startswith("portarias de extensão") or cand.startswith("portarias de extensao")

        first_line = _first_nonblank(win_text)
        need_realign = (not first_line) or (not _looks_like_section_header(sec_key, first_line)
                                            and not first_line.lower().startswith("aviso"))

        if need_realign:
            old_start = win_start
            # search band: a little before the org band start (to catch header above) up to band end
            pad_left = 1500
            search_start = max(0, a - pad_left)
            search_end = z

            # 1) try to snap to the section header inside the wider band
            sec_label_surface = (s.get("surface_path") or [sec_key])[-1]
            sec_start_abs, _sec_end_abs = find_section_block_in_band_spacy(
                nlp, body_text, search_start, search_end, sec_label_surface, section_key=sec_key
            )
            if search_start <= sec_start_abs < search_end:
                win_start = sec_start_abs
                win_text = body_text[win_start:win_end]
                _dbg_print_realign(sec_key, "snap to section header", old_start, win_start, body_text, win_end)
                first_line = _first_nonblank(win_text)

            # 2) if still not aligned (not header nor 'Aviso' line), snap to earliest indexed title
            if not _looks_like_section_header(sec_key, first_line) and not first_line.lower().startswith("aviso"):
                probe_text = body_text[search_start:search_end]
                title_starts_rel_probe = index_block_titles(nlp, probe_text, sec_key)
                if title_starts_rel_probe:
                    first_title_abs = search_start + min(title_starts_rel_probe)
                    if win_start <= first_title_abs < win_end:
                        win_start = first_title_abs
                        win_text = body_text[win_start:win_end]
                        _dbg_print_realign(sec_key, "snap to first indexed title", old_start, win_start, body_text, win_end)

        # DEBUG probes/header + items
        _dbg_print_block_header(sec_key, win_start, win_end, body_text)
        _dbg_print_items(sec_key, items)
        _dbg_print_window_probe(sec_key, win_start, win_end, body_text)

        # --- NEW: pre-index all titles in this block
        title_starts_rel = index_block_titles(nlp, win_text, sec_key)
        # DEBUG list titles
        _dbg_print_indexed_titles(sec_key, win_start, win_text, title_starts_rel)

        def _emit(idx: int, body_span: Dict[str, int], method: str, confidence: float, strategy: str, preview: Optional[str]):
            it = items[idx]
            diag = {
                "band": {"start": a, "end": z},
                "block": {"start": win_start, "end": win_end, "name": sec_key},
                "strategy": strategy,
                "match_preview": preview,
                "titles_indexed": len(title_starts_rel),
            }
            results.append({
                "section_path": sec_path,
                "section_name": sec_key,
                "section_span": sec_span,
                "item_text": it.get("text", ""),
                "item_span_sumario": it.get("span"),
                "org_context": band_meta,
                "body_span": body_span,
                "confidence": confidence,
                "method": method,
                "diagnostics": diag,
            })

        # Fast path: same number of titles and items → 1:1 mapping
        title_starts = [win_start + p for p in title_starts_rel]
        if title_starts and len(title_starts) == len(items):
            title_starts.sort()
            for i, ts in enumerate(title_starts):
                s_abs_raw = max(win_start, min(win_end, ts))
                e_abs = title_starts[i+1] if i+1 < len(title_starts) else win_end
                e_abs = max(s_abs_raw, min(win_end, e_abs))
                s_abs = _snap_left_to_line_start(body_text, s_abs_raw, win_start)
                if e_abs < s_abs:
                    e_abs = s_abs
                _emit(i, {"start": s_abs, "end": e_abs}, "title_index", 0.70, "title_index", body_text[s_abs:s_abs+60])
            continue  # next section

        # Fallback: per-item tiny search
        starts: List[Tuple[int, int, str, Optional[str]]] = []  # (abs_start, item_index, strategy, preview)
        for idx, it in enumerate(items):
            pos_rel = None; strategy = "none"; preview = None

            hit = _find_by_firstline(nlp, win_text, it.get("text", "") or "")
            if hit:
                pos_rel, preview = hit[0], hit[1]; strategy = "first_line_matcher"
            else:
                hit2 = _find_by_signatures(nlp, win_text, it.get("text", "") or "", sec_key)
                if hit2:
                    pos_rel, preview = hit2[0], hit2[1]; strategy = "signature_phrasematcher"
                else:
                    hit3 = _find_by_head_fallback(win_text, it.get("text", "") or "")
                    if hit3:
                        pos_rel, strategy, preview = hit3

            if pos_rel is not None:
                starts.append((win_start + pos_rel, idx, strategy, preview))

        # sort + de-dup (nudge ties)
        starts.sort(key=lambda t: t[0])
        dedup: List[Tuple[int, int, str, Optional[str]]] = []
        last = None
        for pos, idx, strat, prev in starts:
            if last is not None and pos <= last:
                pos = last + 1
            pos = min(pos, win_end)
            dedup.append((pos, idx, strat, prev))
            last = pos

        idx_to_start = {idx: pos for (pos, idx, _st, _pv) in dedup}
        ordered = sorted(dedup, key=lambda t: t[0])

        # DEBUG per-item starts
        _dbg_print_found_starts(sec_key, ordered, body_text)

        # next-start mapping
        start_to_end: Dict[int, int] = {}
        for i in range(len(ordered)):
            cur_pos = ordered[i][0]
            nxt_pos = ordered[i+1][0] if i+1 < len(ordered) else win_end
            start_to_end[cur_pos] = max(cur_pos, min(win_end, nxt_pos))

        # emit found
        for pos, idx, strat, prev in ordered:
            right = start_to_end.get(pos, win_end)
            s_abs_raw = max(win_start, min(win_end, int(pos)))
            e_abs = max(s_abs_raw, min(win_end, int(right)))
            s_abs = _snap_left_to_line_start(body_text, s_abs_raw, win_start)
            if e_abs < s_abs:
                e_abs = s_abs
            _emit(idx, {"start": s_abs, "end": e_abs}, strat, 0.60, strat, prev)

        # emit missing as section fallback (CLAMP to next header)
        cut = _find_next_section_cut(nlp, body_text, win_start, win_end, section_keys_in_this_org)
        fallback_end = cut if cut is not None else win_end
        _dbg_print_fallback_clamp(sec_key, win_start, win_end, cut)

        missing = [i for i in range(len(items)) if i not in idx_to_start]
        for idx in missing:
            _emit(idx, {"start": win_start, "end": fallback_end}, "section_fallback", 0.10, "fallback", None)

    summary = {
        "found": sum(1 for r in results if r["method"] != "section_fallback"),
        "not_found": sum(1 for r in results if r["method"] == "section_fallback"),
        "avg_confidence": round(sum(r["confidence"] for r in results) / max(1, len(results)), 4),
    }
    return {"summary": summary, "results": results}

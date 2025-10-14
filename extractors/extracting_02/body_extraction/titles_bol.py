# body_extraction/titles_bol.py
from __future__ import annotations
from typing import Dict, List, Tuple
import re



# titles_bol.py (top of file)
DEBUG = True

def _dbg_enabled() -> bool:
    return DEBUG

def _dbg_print_lines_preview(win_start: int, win_text: str, limit: int = 50):
    if not _dbg_enabled(): return
    print(f"[WIN-DBG] scan normalized lines for prefixes:")
    pos = win_start
    for i, raw in enumerate(win_text.splitlines()):
        text = " ".join(raw.split())
        bol_ok = True  # we are iterating per physical line, so this is true
        mark = ""
        # we only show first `limit` lines to avoid spam
        if i >= limit:
            print("  ... (truncated after", limit, "lines)")
            break
        print(f"  {i+1:03d} @ {pos:5d} bol_ok={str(bol_ok):5} {mark:>12} | {text!r}")
        pos += len(raw) + 1  # +1 for '\n'

def _dbg_print_title_hits(section_key: str, starts_rel_sorted: list, win_start: int, win_text: str):
    if not _dbg_enabled(): return
    print(f"[TITLE-DBG] spaCy/BOL title hits for {section_key}: {len(starts_rel_sorted)}")
    for rel in starts_rel_sorted:
        abs_pos = win_start + rel
        line_start = win_text.rfind("\n", 0, rel) + 1
        line_end   = win_text.find("\n", rel)
        if line_end == -1: line_end = len(win_text)
        line = win_text[line_start:line_end].strip()
        print(f"  - @ {abs_pos:5d} | line={line!r}")



# ---------- tiny utils ----------

def _norm_for_bol(s: str) -> str:
    # Normalize common OCR/whitespace quirks
    s = s.replace("\u00AD", "")                          # soft hyphen
    s = re.sub(r"-\s*\r?\n\s*", "", s)                   # join hyphen line-breaks
    s = s.replace("\u00A0", " ")                         # NBSP -> space
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _first_line_tokens(item_text: str, max_tokens: int = 12) -> List[str]:
    first = ""
    for ln in (item_text or "").splitlines():
        t = ln.strip()
        if t:
            first = t
            break
    if not first:
        return []
    toks = re.findall(r"[A-Za-zÀ-ÿ0-9]+(?:-[A-Za-zÀ-ÿ0-9]+)?", first)
    toks = [t for t in toks if len(t) > 1][:max_tokens]
    return toks

def _compile_bol_regex_from_tokens(tokens: List[str]) -> re.Pattern | None:
    if not tokens:
        return None
    parts = [re.escape(tokens[0])]
    for t in tokens[1:]:
        parts.append(r"(?:\W+\s*)?")
        parts.append(re.escape(t))
    pat = r"^\s*" + "".join(parts) + r"\b"
    return re.compile(pat, flags=re.IGNORECASE)

def _scan_bol_lines(win_text: str) -> Tuple[List[int], List[str]]:
    offs, lines = [], []
    pos = 0
    for ln in win_text.splitlines(True):  # keepends
        lines.append(ln.rstrip("\r\n"))
        offs.append(pos)
        pos += len(ln)
    if not lines:
        lines, offs = [win_text], [0]
    return offs, lines

def _snap_left_to_line_start(text: str, start: int, floor: int) -> int:
    rel = text.rfind("\n", floor, start)
    return floor if rel == -1 else rel + 1

# ---------- public API ----------

def starts_from_item_prefixes(win_text: str, items: List[dict]) -> Dict[int, int]:
    """
    Return {item_index: rel_start_char} for items whose first-line token
    prefix matches a window line strictly at BOL. Conflicts resolved by
    'longest token prefix wins'.
    """
    line_starts, lines = _scan_bol_lines(win_text)

    # Build patterns per item
    patterns: List[tuple[int, re.Pattern, int]] = []  # (item_idx, regex, token_count)
    for i, it in enumerate(items):
        toks = _first_line_tokens(it.get("text", "") or "", max_tokens=12)
        rx = _compile_bol_regex_from_tokens(toks)
        if rx:
            patterns.append((i, rx, len(toks)))

    # Claim lines
    claims: Dict[int, List[tuple[int, int]]] = {}  # line_idx -> [(item_idx, tok_cnt)]
    for li, raw in enumerate(lines):
        norm_line = _norm_for_bol(raw)
        for idx, rx, tok_cnt in patterns:
            if rx.match(norm_line):
                claims.setdefault(li, []).append((idx, tok_cnt))

    # Choose winner per line (longest prefix), and ensure each item used once
    chosen: Dict[int, int] = {}
    used_items: set[int] = set()
    for li, lst in claims.items():
        lst.sort(key=lambda t: (-t[1], t[0]))  # more tokens wins
        for idx, _tk in lst:
            if idx not in used_items:
                chosen[idx] = line_starts[li]  # rel start
                used_items.add(idx)
                break
    return chosen

def augment_with_spacy_bol_hits(nlp, win_text: str, items: List[dict], got: Dict[int, int]) -> Dict[int, int]:
    """
    For items still missing, try a PhraseMatcher over the full item text but
    accept hits only at (or very near) beginning of a line.
    """
    from spacy.matcher import PhraseMatcher

    missing = [i for i in range(len(items)) if i not in got]
    if not missing:
        return got

    doc = nlp.make_doc(win_text)
    pm = PhraseMatcher(nlp.vocab, attr="LOWER")
    id2idx = {}
    for i in missing:
        txt = _norm_for_bol(items[i].get("text", "") or "")
        if not txt:
            continue
        label = f"ITM__{i}"
        pm.add(label, [nlp.make_doc(txt)])
        id2idx[nlp.vocab.strings[label]] = i

    hits = pm(doc)
    for rule_id, s, _e in sorted(hits, key=lambda h: doc[h[1]].idx):
        idx = id2idx.get(rule_id)
        if idx is None or idx in got:
            continue
        hit_pos = doc[s].idx
        line_start = win_text.rfind("\n", 0, hit_pos) + 1
        if hit_pos - line_start <= 2:
            got[idx] = line_start
    return got

def augment_with_firstline_bol(nlp, win_text: str, items: List[dict], got: Dict[int, int]) -> Dict[int, int]:
    """
    Final tiny fallback: approx first-line match at BOL.
    """
    from spacy.matcher import Matcher

    missing = [i for i in range(len(items)) if i not in got]
    if not missing:
        return got

    doc = nlp.make_doc(win_text)
    for i in missing:
        first = ""
        for ln in (items[i].get("text", "") or "").splitlines():
            t = ln.strip()
            if t:
                first = t
                break
        if not first:
            continue

        m = Matcher(nlp.vocab)
        toks = [t for t in re.findall(r"[A-Za-zÀ-ÿ0-9]+", first)][:10]
        if not toks:
            continue
        pat = [{"LOWER": toks[0].lower()}] + [{"LOWER": t.lower()} for t in toks[1:]]
        m.add("BOL_FIRST", [pat])
        hits = m(doc)
        if not hits:
            continue
        start = min(doc[s].idx for _, s, _ in hits)
        line_start = win_text.rfind("\n", 0, start) + 1
        if start - line_start <= 2:
            got[i] = line_start
    return got

def slice_by_sorted_starts(
    body_text: str,
    win_start: int,
    win_end: int,
    starts_map: Dict[int, int],
    item_count: int,
) -> List[Tuple[int, int]]:
    """
    Convert {item_idx: rel_start} → absolute (start,end) spans per item, using the
    next start (or window end) as the right edge. Missing items get the current
    'best guess' sequence position.
    """
    # Build absolute starts in item order (fallback to window start)
    abs_starts = []
    for i in range(item_count):
        rel = starts_map.get(i, 0)
        abs_starts.append(max(win_start, min(win_end, win_start + rel)))

    # Make spans by next start
    spans: List[Tuple[int, int]] = []
    for i, s_abs_raw in enumerate(abs_starts):
        e_abs = abs_starts[i + 1] if i + 1 < len(abs_starts) else win_end
        e_abs = max(s_abs_raw, min(win_end, e_abs))
        s_abs = _snap_left_to_line_start(body_text, s_abs_raw, win_start)
        if e_abs < s_abs:
            e_abs = s_abs
        spans.append((s_abs, e_abs))
    return spans

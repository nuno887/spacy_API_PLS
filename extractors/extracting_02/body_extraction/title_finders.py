# body_extraction/title_finders.py
from typing import List, Dict, Optional, Tuple
import re

_SOFT_HYPHEN = "\u00AD"
_NBSP = "\u00A0"

def normalize_line(s: str) -> str:
    """
    Normalize for robust line-anchored comparisons:
    - remove soft hyphen
    - dehyphenate across line breaks '-\\n'
    - collapse whitespace, keep accents
    - lowercase compare will be done by caller
    """
    if not s:
        return ""
    s = s.replace(_SOFT_HYPHEN, "")
    s = re.sub(r"-\s*\n", "", s)
    s = s.replace(_NBSP, " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _first_nonblank_line(s: str) -> str:
    for ln in (s or "").splitlines():
        t = ln.strip()
        if t:
            return t
    return ""

def build_title_prefix(item_text: str, tokens: int = 10) -> str:
    """
    Take the first N tokens of the item's first line, to keep matching specific
    but resilient to small OCR or punctuation drifts.
    """
    head = _first_nonblank_line(item_text or "")
    # words and common in-word separators (e.g., siglas or slashes)
    toks = re.findall(r"[A-Za-zÀ-ÿ0-9]+(?:[-/][A-Za-zÀ-ÿ0-9]+)?", head)
    toks = toks[:tokens]
    return normalize_line(" ".join(toks))

def section_title_prefixes_from_items(items: List[Dict], tokens: int = 10) -> List[Dict[str, str]]:
    """
    For each Sumário item, build two candidates:
      - full: normalized full first line
      - prefix: normalized first-N-tokens (tokens param)
    """
    res: List[Dict[str, str]] = []
    for it in items:
        raw_first = _first_nonblank_line(it.get("text", "") or "")
        res.append({
            "full":   normalize_line(raw_first),
            "prefix": build_title_prefix(it.get("text", "") or "", tokens=tokens),
        })
    return res

def find_title_starts_by_prefixes(
    body_text: str,
    win_start: int,
    win_end:   int,
    items_candidates: List[Dict[str, str]],
) -> List[Optional[Tuple[int, str]]]:
    """
    Scan the window line-by-line and assign the first matching line to each item
    in order. Returns per-item either (abs_pos, mode) where mode ∈ {"full","prefix"}
    or None if no match.
    """
    window = body_text[win_start:win_end]
    # Split lines with offsets (keepends=True so BOL positions are exact)
    lines = []
    off = 0
    for ln in window.splitlines(True):
        abs_pos = win_start + off
        norm = normalize_line(ln)
        lines.append((abs_pos, norm))
        off += len(ln)

    found: List[Optional[Tuple[int, str]]] = [None] * len(items_candidates)

    for abs_pos, norm in lines:
        if not norm:
            continue
        low = norm.lower()
        # assign in order; once an item is matched, skip it for future lines
        for i, cand in enumerate(items_candidates):
            if found[i] is not None:
                continue
            full = (cand.get("full") or "").lower()
            pref = (cand.get("prefix") or "").lower()

            if full and low.startswith(full):
                found[i] = (abs_pos, "full")
                break
            if pref and low.startswith(pref):
                found[i] = (abs_pos, "prefix")
                break

    return found

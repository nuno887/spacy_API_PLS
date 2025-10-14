# debug_tools.py
from __future__ import annotations
from typing import Dict, List, Tuple, Iterable

# Global switch (you can override from main/extract)
DEBUG = True

def set_debug(enabled: bool) -> None:
    """Enable/disable all prints at runtime."""
    global DEBUG
    DEBUG = enabled

def _p(s: str) -> None:
    if DEBUG:
        print(s)

# ──────────────────────────────────────────────────────────────────────────────
# High-level: header windows
# ──────────────────────────────────────────────────────────────────────────────

def dbg_header_windows(
    body_text: str,
    windows_by_key: Dict[str, List[Tuple[int, int]]],
    wanted_keys: Iterable[str] | None = None,
) -> None:
    """
    Pretty print the section windows discovered by the header matcher.
    """
    if not DEBUG:
        return

    _p("[HDR-DBG] windows_by_key:")
    keys = list(windows_by_key.keys())
    if wanted_keys:
        want = set(wanted_keys)
        keys = [k for k in keys if k in want]

    for k in keys:
        for (a, z) in windows_by_key.get(k, []):
            a = max(0, min(a, len(body_text)))
            z = max(a, min(z, len(body_text)))
            first_line = _first_nonblank_line(body_text[a:z])
            _p(f"  • {k:>20}: @{a}..{z} | first_line='{first_line}'")

# ──────────────────────────────────────────────────────────────────────────────
# Window probes
# ──────────────────────────────────────────────────────────────────────────────

def dbg_window_first_lines(
    label: str,
    body_text: str,
    start: int,
    end: int,
    max_lines: int = 5,
) -> None:
    """
    Show first raw lines in the window for quick sanity-checks.
    """
    if not DEBUG:
        return
    start = max(0, min(start, len(body_text)))
    end   = max(start, min(end, len(body_text)))

    _p(f"[WIN-DBG] {label} window @{start}..{end} (len={end-start})")
    lines = body_text[start:end].splitlines()
    _p("[WIN-DBG] first raw lines:")
    for ln in lines[:max_lines]:
        _p(f"  · {ln!r}")

def dbg_scan_lines_with_prefixes(
    body_text: str,
    start: int,
    end:   int,
    prefixes: List[str],
    tag: str = "scan normalized lines for prefixes",
    max_show: int = 50,
) -> None:
    """
    Line-by-line normalized scan: shows BOL-ok and whether a line starts with any of the provided prefixes.
    """
    if not DEBUG:
        return
    T = body_text
    start = max(0, min(start, len(T)))
    end   = max(start, min(end, len(T)))

    _p(f"[WIN-DBG] {tag}:")
    window = T[start:end]
    offs = start
    lines = window.splitlines(True)  # keepends to compute offsets
    shown = 0
    for raw in lines:
        if shown >= max_show:
            _p("  ... (truncated after 50 lines)")
            break
        ln = raw.rstrip("\r\n")
        norm = _normalize_line(ln)
        bol_ok = True  # we are at real BOL because we split on \n boundaries
        is_title = any(norm.startswith(_normalize_line(pfx)) for pfx in prefixes)
        marker = " <<< TITLE?" if is_title else ""
        _p(f"  {shown+1:03d} @ {offs:5d} bol_ok={str(bol_ok):<5} {('' if not is_title else ''):>12}{'|'} {norm!r}{marker}")
        offs += len(raw)
        shown += 1

# ──────────────────────────────────────────────────────────────────────────────
# Title hits & alignment
# ──────────────────────────────────────────────────────────────────────────────

def dbg_spacy_title_hits(
    section_key: str,
    title_starts_abs: List[int],
    body_text: str,
) -> None:
    """
    Print the absolute positions spaCy/patterns returned for titles in a window.
    """
    if not DEBUG:
        return
    _p(f"[TITLE-DBG] spaCy title hits for {section_key}: {len(title_starts_abs)}")
    for pos in title_starts_abs:
        line = _line_at(body_text, pos)
        _p(f"  - @ {pos:5d} | line={line!r}")

def dbg_item_alignment(
    section_key: str,
    items: List[dict],
    chosen_starts_abs: List[int],
    body_text: str,
) -> None:
    """
    Show which Sumário items aligned to which absolute starts (first-line preview).
    """
    if not DEBUG:
        return
    _p(f"[ALIGN-DBG] {section_key}: items={len(items)} matched_starts={len(chosen_starts_abs)}")
    for i, pos in enumerate(chosen_starts_abs):
        title = _first_line_of_item(items[i].get("text", ""))
        line = _line_at(body_text, pos)
        _p(f"  - item[{i:02d}] pos={pos:5d} | item_first='{title}' | win_line='{line}'")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _first_nonblank_line(s: str) -> str:
    for ln in (s or "").splitlines():
        t = ln.strip()
        if t:
            return t
    return ""

def _normalize_line(s: str) -> str:
    # light normalization for debug viewing; leave real matching to your matchers
    return " ".join((s or "").replace("\u00AD", "").replace("\u00A0", " ").split())

def _line_at(text: str, abs_pos: int) -> str:
    abs_pos = max(0, min(abs_pos, len(text)))
    bol = text.rfind("\n", 0, abs_pos) + 1
    eol = text.find("\n", abs_pos)
    if eol == -1:
        eol = len(text)
    return text[bol:eol].strip()

def _first_line_of_item(item_text: str) -> str:
    for ln in (item_text or "").splitlines():
        t = ln.strip()
        if t:
            return t
    return ""

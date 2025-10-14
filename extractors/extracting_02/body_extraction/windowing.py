# body_extraction/windowing.py
from typing import Optional, List, Tuple
from spacy.tokens import Doc
from spacy.matcher import PhraseMatcher

from .matchers import get_header_matcher
from .sections import find_section_block_in_band_spacy
from .title_index import index_block_titles
from .body_taxonomy import BODY_SECTIONS


# --- Debug (local to windowing.py) ---
DEBUG = globals().get("DEBUG", True)
DEBUG_SECTIONS = globals().get("DEBUG_SECTIONS", set())

def _dbg_enabled(sec_key: str) -> bool:
    return DEBUG and (not DEBUG_SECTIONS or sec_key in DEBUG_SECTIONS)

def _dbg_print_realign(sec_key: str, reason: str, old_start: int, new_start: int, body_text: str, win_end: int):
    if not _dbg_enabled(sec_key):
        return
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



def find_next_global_header(nlp, text: str, start: int, end: int) -> Optional[int]:
    """
    Scan [start, end) for the earliest header of ANY known section.
    Returns absolute char position or None.
    """
    start = max(0, min(len(text), start))
    end   = max(start, min(len(text), end))
    if start >= end:
        return None

    doc: Doc = nlp.make_doc(text[start:end])
    pm = get_header_matcher(nlp, BODY_SECTIONS.keys())
    hits = pm(doc)
    if not hits:
        return None

    # earliest by token char index
    s_abs = min((doc[s].idx for _, s, _ in hits), default=None)
    return None if s_abs is None else (start + s_abs)


def snap_left_to_line_start(text: str, start: int, floor: int) -> int:
    """
    Snap 'start' left to the beginning of its line, but never before 'floor'.
    """
    start = max(floor, min(len(text), start))
    rel = text.rfind("\n", floor, start)
    return floor if rel == -1 else rel + 1


def guard_span(start: int, end: int, floor: int, ceil: int) -> Tuple[int, int]:
    """
    Clamp [start,end) to [floor,ceil) and repair inversions.
    """
    s = max(floor, min(ceil, int(start)))
    e = max(s,     min(ceil, int(end)))
    return s, e


def _first_nonblank_line(s: str) -> str:
    for ln in s.splitlines():
        t = ln.strip()
        if t:
            return t
    return ""


def _looks_like_section_header(sec_key: str, line: str) -> bool:
    sec = BODY_SECTIONS.get(sec_key)
    if not sec:
        return False
    cand = (line or "").rstrip(":").strip().lower()
    aliases = [h.rstrip(":").strip().lower() for h in (sec.header_aliases or [])]
    return cand in aliases


def realign_window(
    nlp,
    body_text: str,
    sec_key: str,
    win_start: int,
    win_end: int,
    search_start: int,
    search_end: int,
    surface_label: str,
) -> int:
    """
    If the window begins mid-paragraph or not on a header/title,
    try to snap to the section header; if that fails, snap to the
    earliest indexed title. Returns the (possibly) new win_start.
    """
    # clamp the search area safely
    N = len(body_text)
    search_start = max(0, min(N, search_start))
    search_end   = max(search_start, min(N, search_end))
    win_start    = max(search_start, min(win_end, win_start))

    win_text = body_text[win_start:win_end]
    first_line = _first_nonblank_line(win_text)

    # already aligned? (header or typical title like “Aviso …”)
    if first_line and (_looks_like_section_header(sec_key, first_line) or first_line.lower().startswith("aviso")):
        return win_start

    # 1) try explicit section header in the wider band
    sec_start_abs, _ = find_section_block_in_band_spacy(
        nlp, body_text, search_start, search_end, surface_label, section_key=sec_key
    )
    if search_start <= sec_start_abs < search_end:
        if '_dbg_enabled' in globals() and _dbg_enabled(sec_key):
            _dbg_print_realign(sec_key, "snap to section header", win_start, sec_start_abs, body_text, win_end)
        return sec_start_abs

    # 2) else snap to earliest title we can index in the wider band
    probe_text = body_text[search_start:search_end]
    rel_hits = index_block_titles(nlp, probe_text, sec_key)
    if rel_hits:
        first_title_abs = search_start + min(rel_hits)
        if win_start <= first_title_abs < win_end:
            if '_dbg_enabled' in globals() and _dbg_enabled(sec_key):
                _dbg_print_realign(sec_key, "snap to first indexed title", win_start, first_title_abs, body_text, win_end)
            return first_title_abs

    # keep original
    return win_start


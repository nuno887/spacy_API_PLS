from __future__ import annotations

from typing import Iterable, List, Tuple, Dict

from .matchers import header_matcher_for, find_header_hits_strict
from .body_taxonomy import BODY_SECTIONS

def _line_at(text: str, pos: int) -> str:
    """Return the full line (trimmed) containing absolute char index `pos`."""
    L = text.rfind("\n", 0, pos) + 1
    R = text.find("\n", pos)
    if R == -1:
        R = len(text)
    return " ".join(text[L:R].strip().split())

def index_global_headers_strict(
    nlp,
    text: str,
    section_keys: Iterable[str],
) -> List[Tuple[int, str]]:
    """
    Return a list of (start_char, section_key) for headers found in `text`,
    but ONLY when the match begins at the start of a line (ignores leading spaces).

    We rely on BODY_SECTIONS.header_aliases via `header_matcher_for(...)`.
    """
    doc = nlp.make_doc(text)
    pm = header_matcher_for(nlp, section_keys)

    headers: List[Tuple[int, str]] = []
    for match_id, start, _end in pm(doc):
        label = nlp.vocab.strings[match_id]  # e.g. "HDR__Convencoes"
        # Labels follow "HDR__{section_key}" — keep the part after the prefix.
        key = label.split("__", 1)[-1] if "__" in label else label

        start_char = doc[start].idx

        # Enforce "header is at line start" (ignoring left spaces)
        ln_start = text.rfind("\n", 0, start_char) + 1  # 0 if no newline before
        if text[ln_start:start_char].strip() != "":
            continue

        # Standardize tuple order to (start_char, section_key)
        headers.append((start_char, key))

    # Sort by position (absolute start char)
    headers.sort(key=lambda t: t[0])
    return headers  # <-- (bug fix) previously missing

def build_windows_for_sections(
    nlp,
    body_text: str,
    sections: List[dict],
    *,
    debug: bool = False,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Scan the entire body for TRUE section headers (start-of-line only), restricted
    to the section keys present in `sections`. Convert each header hit into a window
    [start, next_start) where the last window ends at EOF.

    Returns:
        { section_key: [(start, end), ...] }
    """
    # Only consider section keys that actually appear in this document & are known
    wanted_keys = {(s.get("path") or ["(unknown)"])[-1] for s in sections}
    wanted_keys = {k for k in wanted_keys if k in BODY_SECTIONS}

    # Strict, start-of-line header hits (already de-duped)
    # NOTE: find_header_hits_strict must yield [(start_char, sec_key)]
    hits = find_header_hits_strict(nlp, body_text, wanted_keys)  # [(start_char, sec_key)]
    hits.sort(key=lambda t: t[0])

    if debug:
        print("[HDR-DBG] wanted_keys:", sorted(wanted_keys))
        print(f"[HDR-DBG] total header hits: {len(hits)}")
        for start, key in hits:
            # show the whole header line
            line_start = body_text.rfind("\n", 0, start) + 1
            line_end = body_text.find("\n", start)
            if line_end == -1:
                line_end = len(body_text)
            line = body_text[line_start:line_end].strip()
            print(f"  - hit @ {start:5d} | {key:>20} | line={line!r}")

    # Build windows per key
    windows_by_key: Dict[str, List[Tuple[int, int]]] = {k: [] for k in wanted_keys}
    if not hits:
        return windows_by_key

    for i, (start_pos, key) in enumerate(hits):
        end_pos = len(body_text) if i + 1 == len(hits) else hits[i + 1][0]
        # Guards
        start_pos = max(0, min(start_pos, end_pos))
        end_pos = max(start_pos, min(end_pos, len(body_text)))
        windows_by_key[key].append((start_pos, end_pos))

    if debug:
        print("[HDR-DBG] windows_by_key:")
        for key in sorted(windows_by_key.keys()):
            for (a, b) in windows_by_key[key]:
                # first line inside the window
                win = body_text[a:b]
                first = next((ln.strip() for ln in win.splitlines() if ln.strip()), "")
                print(f"  • {key:>20}: @{a}..{b} | first_line={first!r}")

    return windows_by_key

def pick_first_window_for_key(
    windows_by_key: Dict[str, List[Tuple[int, int]]],
    section_key: str,
) -> Tuple[int, int] | None:
    """
    Return the first (start, end) window for a given section key, if any.
    """
    wins = windows_by_key.get(section_key) or []
    return wins[0] if wins else None

def build_section_windows_strict(
    nlp,
    body_text: str,
    section_keys: Iterable[str],
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Build windows [start, end) in `body_text` for each known section key by:
      1) Finding all global headers (strict: must start at line start).
      2) Sorting them by position.
      3) Window end = next header start (or text end for the last one).

    Returns:
      { section_key: [(start, end), ...], ... }
    """
    # 1) Find every header occurrence with strict "line start" rule
    headers: List[Tuple[int, str]] = index_global_headers_strict(nlp, body_text, section_keys)

    # 2) Already sorted by index_global_headers_strict; sort again defensively
    headers.sort(key=lambda kv: kv[0])

    # 3) Produce windows
    windows_by_key: Dict[str, List[Tuple[int, int]]] = {}
    text_len = len(body_text)

    for i, (start_pos, key) in enumerate(headers):
        end_pos = headers[i + 1][0] if i + 1 < len(headers) else text_len
        # guard against inversions
        start_pos = max(0, min(start_pos, text_len))
        end_pos = max(start_pos, min(end_pos, text_len))
        windows_by_key.setdefault(key, []).append((start_pos, end_pos))

    return windows_by_key

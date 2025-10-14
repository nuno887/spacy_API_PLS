# body_extraction/align.py

from typing import List, Dict
from spacy.tokens import Doc
from .features import normalize_for_match, item_anchor_phrases
from .matchers import get_title_matcher, get_anchor_phrase_matcher

# simple cue words that often introduce references rather than item starts
REFERENCE_LEFT = {"ver", "vide", "cf.", "conforme", "nos", "termos", "de", "acordo"}


def _is_line_start_title(nlp, text: str, section_key: str) -> bool:
    """Return True if 'text' matches a title pattern at the beginning of the line for this section."""
    d = nlp.make_doc(text)
    tm = get_title_matcher(nlp, section_key)
    for _, s, _ in tm(d):
        if s == 0:  # ensure it's anchored at line start
            return True
    return False


def _find_titles_in_window(nlp, window_text: str, section_key: str) -> List[int]:
    """
    Scan by lines and return offsets (relative to window_text) where a line-start title is detected.
    """
    offs = []
    pos = 0
    for ln in window_text.splitlines(True):  # keepends=True to track positions
        stripped = ln.strip()
        if stripped and _is_line_start_title(nlp, stripped, section_key):
            offs.append(pos + ln.find(stripped))
        pos += len(ln)
    return offs


def _looks_like_reference_left(ctx: str) -> bool:
    """
    Heuristic: if a candidate anchor is preceded on its line by common reference cues,
    downrank or skip (handled by caller).
    """
    ctx = (ctx or "").strip().lower()
    # simple left-trim to the last ~30 chars
    ctx = ctx[-30:]
    return any(ctx.startswith(tok) or f" {tok} " in ctx for tok in REFERENCE_LEFT)


def locate_candidates_in_window(
    nlp,
    full_text: str,
    window_start: int,
    window_end: int,
    item_text: str,
    section_key: str,
) -> List[Dict]:
    """
    Section-aware candidate finder:
      1) title-like anchors at line start (spaCy Matcher per-section)
      2) anchor phrases (PhraseMatcher per-section)
      3) normalized substring fallback (robust to diacritics/spacing)
    Returns candidates as dicts with keys: start, end, method, confidence, anchor_used.
    """
    window_text = full_text[window_start:window_end]

    candidates: List[Dict] = []

    # ---- 1) Titles at line start
    starts = _find_titles_in_window(nlp, window_text, section_key)
    for s0 in starts:
        s_abs = window_start + s0
        # grab a reasonable snippet; body_extractor will expand later
        e_abs = min(window_end, s_abs + max(80, len(item_text) + 120))
        candidates.append({
            "start": s_abs,
            "end": e_abs,
            "method": "title",
            "confidence": 0.85,
            "anchor_used": "title",
        })
        if len(candidates) >= 8:
            return candidates
    if candidates:
        return candidates

    # ---- 2) Anchor phrases anywhere in window
    pm = get_anchor_phrase_matcher(nlp, section_key)
    doc_win: Doc = nlp.make_doc(window_text)
    for match_id, s, e in pm(doc_win):
        span = doc_win[s:e]
        # penalize if likely a reference: inspect left-of-span on the same line
        line_start = window_text.rfind("\n", 0, span.start_char) + 1
        left_ctx = window_text[line_start:span.start_char]
        ref_penalty = 0.12 if _looks_like_reference_left(left_ctx) else 0.0

        s_abs = window_start + span.start_char
        e_abs = window_start + span.end_char
        candidates.append({
            "start": s_abs,
            "end": min(window_end, e_abs + 120),
            "method": "anchor",
            "confidence": 0.70 - ref_penalty,
            "anchor_used": doc_win.vocab.strings[match_id],
        })
        if len(candidates) >= 8:
            break
    if candidates:
        return candidates

    # ---- 3) Normalized substring fallback
    win_norm = normalize_for_match(window_text)
    anchors = item_anchor_phrases(item_text)
    for a in anchors:
        a_norm = normalize_for_match(a)
        pos = 0
        while pos < len(win_norm):
            j = win_norm.find(a_norm, pos)
            if j == -1:
                break
            raw_start = max(0, min(len(window_text), j))
            s_abs = window_start + raw_start
            e_abs = min(window_start + raw_start + max(60, len(a) + 120), window_end)
            candidates.append({
                "start": s_abs,
                "end": e_abs,
                "method": "norm_substr",
                "confidence": 0.60,
                "anchor_used": "norm_substr",
            })
            if len(candidates) >= 8:
                break
            pos = j + max(1, len(a_norm))
        if len(candidates) >= 8:
            break

    return candidates

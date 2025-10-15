# body_extraction/matchers.py
from __future__ import annotations

from typing import FrozenSet, Iterable, List, Tuple
from spacy.matcher import PhraseMatcher

from .body_taxonomy import BODY_SECTIONS


def _is_line_start(text: str, char_pos: int, slop: int = 2) -> bool:
    """
    True if `char_pos` is at the start of a physical line, allowing up to `slop`
    leading spaces/tabs before the token (helps with OCR/formatting quirks).
    """
    line_start = text.rfind("\n", 0, char_pos) + 1
    # accept if the first non-newline characters up to char_pos are only spaces/tabs
    return (char_pos - line_start) <= slop


def build_header_phrasematcher(
    nlp,
    section_keys: FrozenSet[str] | Iterable[str],
) -> PhraseMatcher:
    """
    Build a PhraseMatcher that detects *section headers*.
    - Labels are set to the *section key* itself (e.g., 'Convencoes').
    - Aliases come from BODY_SECTIONS[key].header_aliases
    - We normalize minor punctuation: strip trailing ':' and outer whitespace.
    """
    pm = PhraseMatcher(nlp.vocab, attr="LOWER")
    for key in section_keys:
        sec = BODY_SECTIONS.get(key)
        if not sec:
            continue
        aliases = sec.header_aliases or []
        patterns = []
        for a in aliases:
            if not a:
                continue
            norm = a.rstrip(":").strip()
            if not norm:
                continue
            patterns.append(nlp.make_doc(norm))
        if patterns:
            pm.add(str(key), patterns)
    return pm


def find_header_hits_strict(
    nlp,
    body_text: str,
    section_keys: FrozenSet[str] | Iterable[str],
    *,
    line_start_slop: int = 2,
) -> List[Tuple[int, str]]:
    """
    Find strict header hits for the given section keys in `body_text`.

    Returns:
        List of (start_char, section_key) tuples, de-duplicated and sorted by start.
    Notes:
        - Uses labels == section_key (no 'HDR__' prefixes).
        - Keeps only matches that begin at (or within `line_start_slop` of) the start of a line.
    """
    pm = build_header_phrasematcher(nlp, section_keys)
    doc = nlp.make_doc(body_text)
    raw = pm(doc)

    hits: List[Tuple[int, str]] = []
    for rule_id, s_tok, _e_tok in raw:
        key = nlp.vocab.strings[rule_id]  # label is the section key
        start_char = doc[s_tok].idx
        if _is_line_start(body_text, start_char, slop=line_start_slop):
            hits.append((start_char, key))

    # de-dup and sort (pos, key)
    dedup = sorted(set(hits), key=lambda t: (t[0], t[1]))
    return dedup


# --- Back-compat alias (preferred callers should use build_header_phrasematcher / find_header_hits_strict) ---

def header_matcher_for(nlp, section_keys: set[str]) -> PhraseMatcher:
    """
    Back-compat: historically returned a PhraseMatcher with labels like 'HDR__{key}__{i}'.
    We now return a matcher with labels == section_key to match the rest of the pipeline.
    """
    return build_header_phrasematcher(nlp, section_keys)


def get_header_matcher(nlp, section_keys):
    """Back-compat alias."""
    return build_header_phrasematcher(nlp, section_keys)

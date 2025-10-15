# body_extraction/title_index.py
from __future__ import annotations

from typing import List, Optional, Dict
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc

from .body_taxonomy import BODY_SECTIONS
from .text_norm import normalize_for_search  # <-- unified normalization

import re

def section_title_prefixes_from_items(items: List[Dict], tokens: int = 6) -> List[str]:
    """
    From Sumário items (each with "text"), build short normalized prefixes
    like "Aviso de Projeto de Portaria de" or "Contrato Coletivo entre a".
    These are used for fast line-start matching inside the section window.
    Normalization matches body scanning rules (diacritics, wrapped hyphens, NBSP).
    """
    # Normalization collapses newlines & hyphen wraps, so multi-line item titles
    # produce single-line prefixes that match body titles.


    prefixes: List[str] = []

    for it in items or []:
        raw = (it.get("text") or "").strip()
        if not raw:
            continue
        # normalize consistently, then take the first N whitespace-delimited tokens
        norm = normalize_for_search(raw)
        if not norm:
            continue
        parts = norm.split(" ")
        head = " ".join(parts[:tokens]).strip()
        if head:
            prefixes.append(head)

    # dedup while preserving order (case-insensitive keying over normalized form)
    seen = set()
    uniq: List[str] = []
    for p in prefixes:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq

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
    Return sorted, de-duplicated RELATIVE positions (to window_text)
    where a section item title likely starts, using taxonomy:
      - spaCy Matcher over item_title_patterns
      - plus PhraseMatcher over item_anchor_phrases
    We keep only hits at the beginning of a line (±2 chars).
    """
    if not window_text:
        return []
    doc: Doc = nlp.make_doc(window_text)
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

    # keep only near line start
    def _is_line_start(p: int) -> bool:
        return (p - (window_text.rfind("\n", 0, p) + 1)) <= 2

    starts = [p for p in starts if _is_line_start(p)]
    if not starts:
        return []

    # sort + dedup by physical line
    starts.sort()
    dedup: List[int] = []
    last_line_start = None
    for pos in starts:
        line_start = window_text.rfind("\n", 0, pos) + 1
        if last_line_start is None or abs(line_start - last_line_start) > 2:
            dedup.append(pos)
            last_line_start = line_start

    return dedup

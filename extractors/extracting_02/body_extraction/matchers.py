# body_extraction/matchers.py
from typing import Dict, Iterable, List, Tuple
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Doc
from .body_taxonomy import BODY_SECTIONS

# simple in-process caches keyed by (id(vocab), key)
_HDR_CACHE: Dict[Tuple[int, Tuple[str, ...]], PhraseMatcher] = {}
_TIT_CACHE: Dict[Tuple[int, str], Matcher] = {}
_ANC_CACHE: Dict[Tuple[int, str], PhraseMatcher] = {}

def get_header_matcher(nlp, section_keys: Iterable[str]) -> PhraseMatcher:
    keys = tuple(sorted(set(section_keys)))
    cache_key = (id(nlp.vocab), keys)
    if cache_key in _HDR_CACHE:
        return _HDR_CACHE[cache_key]

    m = PhraseMatcher(nlp.vocab, attr="LOWER")
    for k in keys:
        sec = BODY_SECTIONS.get(k)
        if not sec:
            continue
        for alias in sec.header_aliases:
            if not alias:
                continue
            m.add(f"HDR_{k}", [nlp.make_doc(alias)])
    _HDR_CACHE[cache_key] = m
    return m

def get_title_matcher(nlp, section_key: str) -> Matcher:
    cache_key = (id(nlp.vocab), section_key)
    if cache_key in _TIT_CACHE:
        return _TIT_CACHE[cache_key]

    sec = BODY_SECTIONS.get(section_key)
    m = Matcher(nlp.vocab)
    if sec:
        for i, pat in enumerate(sec.item_title_patterns):
            m.add(f"TIT_{section_key}_{i}", [pat])
    _TIT_CACHE[cache_key] = m
    return m

def get_anchor_phrase_matcher(nlp, section_key: str) -> PhraseMatcher:
    cache_key = (id(nlp.vocab), section_key)
    if cache_key in _ANC_CACHE:
        return _ANC_CACHE[cache_key]

    sec = BODY_SECTIONS.get(section_key)
    pm = PhraseMatcher(nlp.vocab, attr="LOWER")
    if sec:
        for i, phr in enumerate(sec.item_anchor_phrases):
            if not phr:
                continue
            pm.add(f"ANC_{section_key}_{i}", [nlp.make_doc(phr)])
    _ANC_CACHE[cache_key] = pm
    return pm

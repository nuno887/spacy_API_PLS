# body_extraction/matchers.py
from typing import FrozenSet
import spacy
from spacy.matcher import PhraseMatcher
from .body_taxonomy import BODY_SECTIONS
# --- add near the top of the file ---
from typing import Iterable
from spacy.matcher import PhraseMatcher
from .body_taxonomy import BODY_SECTIONS




def _is_line_start(text: str, char_pos: int, slop: int = 0) -> bool:
    line_start = text.rfind("\n", 0, char_pos) + 1
    # allow up to `slop` spaces before the match (accounts for OCR or stray spaces)
    return char_pos - line_start <= slop

def header_matcher_for(nlp, section_keys: set[str]):
    from .body_taxonomy import BODY_SECTIONS
    pm = PhraseMatcher(nlp.vocab, attr="LOWER")
    for key in section_keys:
        sec = BODY_SECTIONS.get(key)
        if not sec:
            continue
        # ONLY section headers — no title patterns here
        aliases = (sec.header_aliases or [])
        for i, alias in enumerate(aliases):
            if not alias:
                continue
            pm.add(f"HDR__{key}__{i}", [nlp.make_doc(alias.rstrip(":"))])
    return pm

# Optional: keep backward compatibility if other modules still call the old name
def get_header_matcher(nlp, section_keys):
    """Back-compat alias."""
    return header_matcher_for(nlp, section_keys)


def build_header_phrasematcher(
    nlp: "spacy.language.Language",
    section_keys: FrozenSet[str],
) -> PhraseMatcher:
    """
    Build a PhraseMatcher that detects section headers.
    The matcher labels are the *section keys* (e.g., 'PortariasExtensao').
    """
    pm = PhraseMatcher(nlp.vocab, attr="LOWER")
    for key in section_keys:
        sec = BODY_SECTIONS.get(key)
        if not sec:
            continue
        aliases = (sec.header_aliases or [])
        if not aliases:
            continue
        # Normalize minor punctuation differences (strip trailing ':')
        patterns = [nlp.make_doc(a.rstrip(":").strip()) for a in aliases if a and a.strip()]
        if patterns:
            pm.add(key, patterns)
    return pm

def find_header_hits_strict(nlp, body_text: str, section_keys: set[str]) -> list[tuple[int, str]]:
    pm = header_matcher_for(nlp, section_keys)
    doc = nlp.make_doc(body_text)
    raw = pm(doc)

    hits: list[tuple[int, str]] = []
    for rule_id, s_tok, _e_tok in raw:
        label = nlp.vocab.strings[rule_id]      # e.g. "HDR__Convencoes__0"
        key = label.split("__", 2)[1]           # -> "Convencoes"
        start_char = doc[s_tok].idx
        # keep ONLY hits at line start (slop 0..2 depending on your OCR)
        if _is_line_start(body_text, start_char, slop=0):
            hits.append((start_char, key))

    # de-dup by (pos,key)
    dedup = sorted(set(hits), key=lambda t: (t[0], t[1]))
    return dedup
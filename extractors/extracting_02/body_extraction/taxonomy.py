# body_extraction/taxonomy.py
from typing import List
from .body_taxonomy import BODY_SECTIONS

def aliases_for(section_key: str) -> List[str]:
    sec = BODY_SECTIONS.get(section_key)
    return list(sec.header_aliases) if sec and sec.header_aliases else []

def anchor_phrases_for(section_key: str) -> List[str]:
    sec = BODY_SECTIONS.get(section_key)
    return list(sec.item_anchor_phrases) if sec and sec.item_anchor_phrases else []

def has_title_patterns(section_key: str) -> bool:
    sec = BODY_SECTIONS.get(section_key)
    return bool(sec and sec.item_title_patterns)

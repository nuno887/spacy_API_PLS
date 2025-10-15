from __future__ import annotations
import re
import unicodedata

_SOFT_HYPHEN = "\u00AD"
_NBSP = "\u00A0"

def strip_diacritics(s: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", s)
        if unicodedata.category(ch) != "Mn"
    )

def normalize_for_search(s: str) -> str:
    """
    Normalize text for robust heading comparisons:
      - remove soft hyphens
      - unwrap hyphen+linebreak (e.g. 'pa-\nlavra' -> 'palavra')
      - convert NBSP to normal space
      - collapse whitespace
      - strip diacritics
      - lowercase
    NOTE: Use ONLY for comparison; do not use to compute offsets.
    """
    if not s:
        return ""
    s = s.replace(_SOFT_HYPHEN, "")
    s = re.sub(r"-\r?\n", "", s)
    s = s.replace(_NBSP, " ")
    s = strip_diacritics(s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

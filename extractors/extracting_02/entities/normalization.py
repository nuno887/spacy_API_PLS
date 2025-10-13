import re
import unicodedata
import string


_EXTRA_PUNCT = {"–", "—", "·", "•", "…", "«", "»", "º", "ª", "‐", "-", "‒", "―", "­"}  # includes soft hyphen U+00AD
_PUNCT_ALL = set(string.punctuation) | _EXTRA_PUNCT

def strip_diacritics(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")

def normalize_text_offsetsafe(s: str) -> str:
    # 1:1 char replacements only (keeps offsets aligned)
    return (
        s.replace("\u00A0", " ")
         .replace("“", "\"").replace("”", "\"")
         .replace("’", "'").replace("‘", "'")
         .replace("«", "\"").replace("»", "\"")
    )

def canonical_org_key(s: str) -> str:
    """
    Canonical identity for an ORG header:
      - Unicode NFKC normalize
      - Remove diacritics (combining marks)
      - Uppercase
      - Drop ALL whitespace (spaces, tabs, newlines)
      - Drop punctuation/dash variants (ASCII + common Unicode)
    """
    if not s:
        return ""
    # Normalize to compatibility form and strip combining marks
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.upper()
    out = []
    for ch in s:
        if ch.isspace():
            continue
        if ch in _PUNCT_ALL:
            continue
        out.append(ch)
    return "".join(out)

def normalize_heading_text(s: str) -> str:
    s = s.strip()
    s = s[:-1] if s.endswith(":") else s
    s = strip_diacritics(s).lower()
    s = re.sub(r'\s+', ' ', s)
    return s

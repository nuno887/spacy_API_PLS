import re
import unicodedata

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
    t = strip_diacritics(s).upper()
    return re.sub(r'[^A-Z0-9]+', '', t)

def normalize_heading_text(s: str) -> str:
    s = s.strip()
    s = s[:-1] if s.endswith(":") else s
    s = strip_diacritics(s).lower()
    s = re.sub(r'\s+', ' ', s)
    return s

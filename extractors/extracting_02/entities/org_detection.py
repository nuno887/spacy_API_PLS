from typing import List
from spacy.tokens import Span
from .constants import HEADER_STARTERS
from .normalization import strip_diacritics
import re
import unicodedata, string

_PUNCT = set(string.punctuation) | {"–", "—", "«", "»", "·", "•", "…", "º", "ª"}

DEBUG = True

def _is_all_caps_line(ln: str) -> bool:
    t = ln.strip()
    if not t: return False
    letters = [ch for ch in t if ch.isalpha()]
    if not letters: return False
    return all(ch == ch.upper() for ch in letters)

# Cache a sanitized, normalized starter list once (avoids type surprises and repeat work)
STARTERS_NORM = tuple(
    strip_diacritics(str(s)).upper().replace(" ", "").replace("-", "")
    for s in HEADER_STARTERS
)

def _norm_org_key(s: str) -> str:
    """
    Canonical identity for an ORG header:
        - Uicode normalize + remove diacrics
        - Uppercase
        - Drop space, dash variants, and punctuation
    Collapses ALL CAPS vs Mixed Case and minor formatting differences.    
    """
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.upper()
    out = []
    for ch in s:
        if ch.isspace() or ch in {"-", "–", "—"} or ch in _PUNCT:
            continue
        out.append(ch)
    return "".join(out)

def _starts_with_starter(ln: str) -> bool:
    #Accent-insensitive, tolerant to spaces/hyphens/fused tokens; prefix match
    t = ln.strip()
    if not t:
        return False
    # split on whitespace + hyphen/en-dash/em-dash + light punctuation
    first_token = re.split(r'[\s\---:,;./]+', t, 1)[0]
    norm_first = strip_diacritics(first_token).upper().replace(" ", "").replace("-", "")
    return any(norm_first.startswith(s) for s in STARTERS_NORM)


# Optional: expose normalized key on spans for downstream comparisons
if not Span.has_extension("org_key"):
    Span.set_extension("org_key", default=None)


def find_org_spans(doc, text: str) -> List[Span]:
    """
    Pass 1: detect ALL-CAPS ORG blocks and record their normalized keys.
    Pass 2: tag later single-line headers as ORG **only if** their key matches a known key (so mixed-case repeats are captured; different endings like '... Relações Internacionais' remains distinct)
    """
    org_spans: List[Span] = []
    lines = text.splitlines(keepends=True)
    line_starts, pos = [], 0
    for ln in lines:
        line_starts.append(pos); pos += len(ln)
    
    canon_keys = set()
    blocks_char_ranges = []

    #--------------- Pass 1: ALL-CAPS blocks -----------------
    i = 0
    while i < len(lines):
        if _starts_with_starter(lines[i])and _is_all_caps_line(lines[i]):
            start_i = i
            j = i + 1
            while j < len(lines) and _is_all_caps_line(lines[j]):
                j += 1
            
            start_char = line_starts[start_i]
            end_char = line_starts[j - 1] + len(lines[j - 1])

            # Use the FIRST LINE for the org key (best matches later single-line repeats)
            header_line = lines[start_i].rstrip("\n")
            key = _norm_org_key(header_line)
            canon_keys.add(key)

            chspan = doc.char_span(start_char, end_char, alignment_mode="expand")
            if chspan is not None:
                sp = Span(doc, chspan.start, chspan.end, label="ORG")
                sp._.org_key = key
                org_spans.append(sp)
                blocks_char_ranges.append((start_char, end_char))
                if DEBUG:
                    preview = header_line[:120].replace("\n", " ")
                    print(f"[ORG PASS1] chars={start_char}-{end_char} key={key} header='{preview}'")
            i = j
        else:
            i += 1
    #---------- Pass 2: mixed-case single-lie repeats matching known keys --------------
    for idx, ln in enumerate(lines):
        if not _starts_with_starter(ln):
            continue
        ln_start = line_starts[idx]
        ln_end = ln_start + len(ln)
        # skip if already within a Pass 1 block
        if any(a <= ln_start and ln_end <= b for (a, b) in blocks_char_ranges):
            continue
        line_clean = ln.rstrip("\n")
        key = _norm_org_key(line_clean)
        if key in canon_keys:
            chspan = doc.char_span(ln_start, ln_end, alignment_mode="expand")
            if chspan is not None:
                sp = Span(doc, chspan.start, chspan.end, label="ORG")
                sp._.org_key = key
                org_spans.append(sp)
                if DEBUG:
                    preview = line_clean[:120].replace("\n", " ")
                    print(f"[ORG PASS2] chars={ln_start}-{ln_end} key={key} line='{preview}'")

    return org_spans


import re
from typing import List, Tuple, Iterable
import spacy
from spacy.matcher import Matcher
from .constants import (
    DOT_LEADER_LINE_RE,
    DOT_LEADER_TAIL_RE,
    BLANK_RE,
    ITEM_STARTERS,
    HEADER_STARTERS,
)
from .normalization import normalize_text_offsetsafe, strip_diacritics
from .org_detection import _is_all_caps_line, _starts_with_starter

_TITLE_MATCHER = None

INLINE_DOTLEADER_SPLIT_RE = re.compile(r'(?:[.\u2026·]{5,})\s*(?:\d{1,3})?\s*')

TITLE_LINE_RE = re.compile(
    r"""(?xi)
    \b(?:acordo|contrato)\s+coletiv\w+\s+de\s+trabalho.*?
    (?:n[\.\º°]?\s*o?)?\s*\d+\s*/\s*\d{2,4}\s*:?\s*\Z
    """
)
_NUMERIC_NOISE_STRIP_RE = re.compile(r'[\s\.\-/,·•…]+')

def build_title_matcher(nlp) -> Matcher:
    """
    Build a token-based matcher for instrument title (no global cache). 
    """
    m = Matcher(nlp.vocab)
    
    def _type(pattern_words):
        return [{"LOWER": w} for w in pattern_words]
    
    # Allowed instrument "Types"

    TYPES = [
        _type(["acordo", "coletivo", "de", "trabalho"]),
        _type(["contrato", "coletivo", "de", "trabalho"]),
        _type(["acordo", "de", "empresa"]),
        _type(["acordo", "de", "adesão"]),
        _type(["acordo", "de", "adesao"]),
        _type(["aditamento"]),
    ]

    # Optional "nº / n.º / n. / n o" marker (very tolerant)
    NUM_MARKER = [
            [{"LOWER": {"IN": ["n", "nº", "n.º", "n.o", "n."]}}],
            [{"LOWER": "n"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": {"IN": ["o", "º"]}, "OP": "?"}],
            [],  # completely absent is OK
        ]

        # number "/" year (+ optional ":" at end)
    NUM_YEAR_TAIL = [
            {"LIKE_NUM": True},
            {"ORTH": "/"},
            {"LIKE_NUM": True},       # 2–4 digits ok
            {"ORTH": ":", "OP": "?"}, # optional colon
        ]
    NUM_YEAR_TAIL_SINGLE = [
    {"TEXT": {"REGEX": r"^\d+/\d{2,4}:?$"}},  # e.g. "1/2014" or "1/2014:"
        ]

    pid = 0
    for t in TYPES:
        for marker in NUM_MARKER:
            head = t + [{"IS_PUNCT": True, "OP": "*"}, *marker, {"IS_SPACE": True, "OP": "*"}]

            # variant A: multi-token  number "/" year [ ":" ]
            mA = head + NUM_YEAR_TAIL
            m.add(f"TITLE_{pid}", [mA]); pid += 1

            # variant B: single-token "1/2014" or "1/2014:"
            mB = head + NUM_YEAR_TAIL_SINGLE
            m.add(f"TITLE_{pid}", [mB]); pid += 1


    return m


def is_numeric_only_item(txt: str) -> bool:
    """
    True if, after stripping whitespace and common separators, the text is only digits.
    This filters out page-number artifacts like '2  2  1' or '12/13'.
    """
    if not txt:
        return True
    core = _NUMERIC_NOISE_STRIP_RE.sub('', txt)
    return core.isdigit() and len(core) > 0

def _looks_like_item_start(ln: str) -> bool:
    raw = normalize_text_offsetsafe(ln).strip()
    if not raw:
        return False
    norm = strip_diacritics(raw).lower()
    starts_itemish = raw[:1] in {'"', "'"} or (raw[:1].isalpha() and raw[:1].upper() == raw[:1])
    has_dash_early = " - " in raw[:80]
    starts_with_keyword = any(norm.startswith(k) for k in ITEM_STARTERS)
    return starts_itemish and (has_dash_early or starts_with_keyword)


def _looks_like_inline_org_start(snippet: str) -> bool:
    s = normalize_text_offsetsafe(snippet).lstrip(' "\'\u201c\u201d\u00ab\u00bb')[:80]
    letters = [ch for ch in s[:40] if ch.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)
    if upper_ratio < 0.75:
        return False
    letters_only = re.sub(r'[^A-Za-zÁÂÃÀÄÅáâãàäåÉÊÈËéêèëÍÎÌÏíîìïÓÔÕÒÖóôõòöÚÛÙÜúûùüÇç]', '', s)
    norm = strip_diacritics(letters_only).upper()
    starters_norm = [strip_diacritics(x).upper() for x in HEADER_STARTERS]
    return any(norm.startswith(st) for st in starters_norm)


def _truncate_item_before_inline_org(raw: str) -> str:
    s = normalize_text_offsetsafe(raw)
    for m in re.finditer(r'\.(?:[ \t]+)', s):
        tail = s[m.end():].lstrip(' "\'\u201c\u201d\u00ab\u00bb')
        if _looks_like_inline_org_start(tail):
            return s[:m.start() + 1].rstrip()
    return s


def clean_item_text(raw: str) -> str:
    raw = _truncate_item_before_inline_org(raw)
    raw = raw.replace("-\n", "").replace("­\n", "")
    raw = re.sub(r'\s*\n\s*', ' ', raw).strip()

    # NEW: kill inline dot-leader runs (..... / ……) + optional small page number
    raw = re.sub(r'(?:[.\u2026·]{5,})\s*(?:\d{1,3})?\s*', ' ', raw)


    raw = re.sub(r'\.*\s*$', '', raw).strip()
    return raw


def _find_inline_item_boundaries(item_text: str, abs_start_char: int, nlp=None) -> List[int]:
    """
    Return absolute cut positions inside an already-detected item block.
    We split when:
      - a period is followed by whitespace and the next token looks like a new item or inline ORG header;
      - an inline dot-leader run (..... / ……) appears and the tail looks like a new item or inline ORG header.
    """
    boundaries: List[int] = []
    s = item_text

    # Rule A: split on ". " when the tail looks like a fresh item or inline ORG
    for m in re.finditer(r'\.(?:\s+)', s):
        j = m.start()
        tail = s[m.end():].lstrip(' \t\r\n"\'\u201c\u201d\u00ab\u00bb')
        if _looks_like_item_start(tail) or _looks_like_inline_org_start(tail):
            boundaries.append(abs_start_char + j + 1)

    # Rule B: split on inline dot-leader separators (..... [digits] ) when the tail looks fresh
    for m in INLINE_DOTLEADER_SPLIT_RE.finditer(s):
        cut = abs_start_char + m.start()  # cut *at* the start of the dot-leader run
        tail = s[m.end():].lstrip(' \t\r\n"\'\u201c\u201d\u00ab\u00bb')
        if _looks_like_item_start(tail) or _looks_like_inline_org_start(tail):
            boundaries.append(cut)

    # Dedup + sort to keep order stable
    boundaries = sorted(set(boundaries))
    return boundaries



def find_item_char_spans(
    full_text: str, start_char: int, end_char: int, next_heading_starts: set
) -> Iterable[Tuple[int, int]]:
    """
    Yield (start_char, end_char) for items within [start_char, end_char).

    An item ends when:
      1) the line is only dot leaders,
      2) the line ends with dot leaders (.....),
      2b) the line ends with a single period AND the next non-blank line looks like a new item,
      3) the next line begins a recognized heading (fallback),
      3b) the next NON-BLANK line is an ORG header (ALL-CAPS + starter).
    Also emits a final flush if the segment ends with an open block.
    """
    segment = full_text[start_char:end_char]
    seg_lines = segment.splitlines(keepends=True)

    # absolute offsets for each line in the segment
    offs: List[int] = []
    p = start_char
    for ln in seg_lines:
        offs.append(p)
        p += len(ln)

    block_start = 0
    i = 0
    while i < len(seg_lines):
        ln = seg_lines[i]

        # Case 1: pure dots line → close previous block
        if DOT_LEADER_LINE_RE.match(ln):
            s = block_start
            e = i
            while s < e and BLANK_RE.match(seg_lines[s]):
                s += 1
            j = e - 1
            while j >= s and BLANK_RE.match(seg_lines[j]):
                j -= 1
            if j >= s:
                yield offs[s], offs[j] + len(seg_lines[j])
            block_start = i + 1
            i += 1
            continue

        # Case 2: trailing dot leaders on the same line
        m = DOT_LEADER_TAIL_RE.search(ln)
        if m:
            s = block_start
            while s <= i and BLANK_RE.match(seg_lines[s]):
                s += 1
            if s <= i:
                end_char_abs = offs[i] + m.start()
                yield offs[s], end_char_abs
            block_start = i + 1
            i += 1
            continue

        # Case 2b: single-period end IF next non-blank looks like a new item
        if re.search(r'\.\s*$', ln) and len(ln.strip()) >= 40:
            k = i + 1
            while k < len(seg_lines) and BLANK_RE.match(seg_lines[k]):
                k += 1
            if k < len(seg_lines) and _looks_like_item_start(seg_lines[k]):
                s = block_start
                while s <= i and BLANK_RE.match(seg_lines[s]):
                    s += 1
                if s <= i:
                    end_char_abs = offs[i] + len(seg_lines[i].rstrip("\n"))
                    yield offs[s], end_char_abs
                block_start = i + 1
                i += 1
                continue

        # Case 3: fallback — if next line starts a recognized heading, close before it
        next_line_start = offs[i + 1] if i + 1 < len(seg_lines) else None
        if next_line_start is not None and next_line_start in next_heading_starts:
            s = block_start
            e = i
            while s < e and BLANK_RE.match(seg_lines[s]):
                s += 1
            j = e
            while j >= s and BLANK_RE.match(seg_lines[j]):
                j -= 1
            if j >= s:
                yield offs[s], offs[j] + len(seg_lines[j])
            block_start = i + 1
            i += 1
            continue

        # Case 3b: if the *next non-blank line* is an ORG header (ALL-CAPS + starter), close before it
        if i + 1 < len(seg_lines):
            k = i + 1
            while k < len(seg_lines) and BLANK_RE.match(seg_lines[k]):
                k += 1
            if k < len(seg_lines):
                nxt = seg_lines[k]
                if _starts_with_starter(nxt) and _is_all_caps_line(nxt):
                    s = block_start
                    e = i
                    while s < e and BLANK_RE.match(seg_lines[s]):
                        s += 1
                    j = e
                    while j >= s and BLANK_RE.match(seg_lines[j]):
                        j -= 1
                    if j >= s:
                        yield offs[s], offs[j] + len(seg_lines[j])
                    # resume at the header line (not the immediate next line)
                    block_start = k
                    i = k
                    continue

        # advance if no rule fired
        i += 1

    # Final flush: emit any open block up to the end of the segment
    if block_start < len(seg_lines):
        s = block_start
        e = len(seg_lines) - 1
        while s <= e and BLANK_RE.match(seg_lines[s]):
            s += 1
        while e >= s and BLANK_RE.match(seg_lines[e]):
            e -= 1
        if s <= e:
            yield offs[s], offs[e] + len(seg_lines[e])

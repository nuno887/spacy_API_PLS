import re
from typing import List, Tuple, Iterable
from .constants import DOT_LEADER_LINE_RE, DOT_LEADER_TAIL_RE, BLANK_RE, ITEM_STARTERS, HEADER_STARTERS
from .normalization import normalize_text_offsetsafe, strip_diacritics

def _looks_like_item_start(ln: str) -> bool:
    raw = normalize_text_offsetsafe(ln).strip()
    if not raw: return False
    norm = strip_diacritics(raw).lower()
    starts_itemish = raw[:1] in {'"', "'"} or (raw[:1].isalpha() and raw[:1].upper() == raw[:1])
    has_dash_early = " - " in raw[:80]
    starts_with_keyword = any(norm.startswith(k) for k in ITEM_STARTERS)
    return starts_itemish and (has_dash_early or starts_with_keyword)

def _looks_like_inline_org_start(snippet: str) -> bool:
    s = normalize_text_offsetsafe(snippet).lstrip(' "\'\u201c\u201d\u00ab\u00bb')[:80]
    letters = [ch for ch in s[:40] if ch.isalpha()]
    if not letters: return False
    upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)
    if upper_ratio < 0.75: return False
    letters_only = re.sub(r'[^A-Za-zÁÂÃÀÄÅáâãàäåÉÊÈËéêèëÍÎÌÏíîìïÓÔÕÒÖóôõòöÚÛÙÜúûùüÇç]', '', s)
    norm = strip_diacritics(letters_only).upper()
    starters_norm = [strip_diacritics(x).upper() for x in HEADER_STARTERS]
    return any(norm.startswith(st) for st in starters_norm)

def _truncate_item_before_inline_org(raw: str) -> str:
    s = normalize_text_offsetsafe(raw)
    for m in re.finditer(r'\.(?:[ \t]+)', s):
        tail = s[m.end():].lstrip(' "\'\u201c\u201d\u00ab\u00bb')
        if _looks_like_inline_org_start(tail):
            return s[:m.start()+1].rstrip()
    return s

def clean_item_text(raw: str) -> str:
    raw = _truncate_item_before_inline_org(raw)
    raw = raw.replace("-\n", "").replace("­\n", "")
    raw = re.sub(r'\s*\n\s*', ' ', raw).strip()
    raw = re.sub(r'\.*\s*$', '', raw).strip()
    return raw

def _find_inline_item_boundaries(item_text: str, abs_start_char: int) -> List[int]:
    boundaries: List[int] = []
    s = item_text
    for m in re.finditer(r'\.(?:[ \t\u00A0]+)', s):
        j = m.start()
        tail = s[m.end():].lstrip(' "\'\u201c\u201d\u00ab\u00bb')
        if _looks_like_item_start(tail) or _looks_like_inline_org_start(tail):
            boundaries.append(abs_start_char + j + 1)
    return boundaries

def find_item_char_spans(full_text: str, start_char: int, end_char: int, next_heading_starts: set) -> Iterable[Tuple[int,int]]:
    segment = full_text[start_char:end_char]
    seg_lines = segment.splitlines(keepends=True)
    offs, p = [], start_char
    for ln in seg_lines:
        offs.append(p); p += len(ln)

    block_start = 0
    for i, ln in enumerate(seg_lines):
        if DOT_LEADER_LINE_RE.match(ln):
            s, e = block_start, i
            while s < e and BLANK_RE.match(seg_lines[s]): s += 1
            j = e - 1
            while j >= s and BLANK_RE.match(seg_lines[j]): j -= 1
            if j >= s: yield offs[s], offs[j] + len(seg_lines[j])
            block_start = i + 1
            continue

        m = DOT_LEADER_TAIL_RE.search(ln)
        if m:
            s = block_start
            while s <= i and BLANK_RE.match(seg_lines[s]): s += 1
            if s <= i:
                end_char_abs = offs[i] + m.start()
                yield offs[s], end_char_abs
            block_start = i + 1
            continue

        if re.search(r'\.\s*$', ln) and len(ln.strip()) >= 40:
            k = i + 1
            while k < len(seg_lines) and BLANK_RE.match(seg_lines[k]): k += 1
            if k < len(seg_lines) and _looks_like_item_start(seg_lines[k]):
                s = block_start
                while s <= i and BLANK_RE.match(seg_lines[s]): s += 1
                if s <= i:
                    end_char_abs = offs[i] + len(seg_lines[i].rstrip("\n"))
                    yield offs[s], end_char_abs
                block_start = i + 1
                continue

        next_line_start = offs[i + 1] if i + 1 < len(seg_lines) else None
        if next_line_start is not None and next_line_start in next_heading_starts:
            s, e = block_start, i
            while s < e and BLANK_RE.match(seg_lines[s]): s += 1
            j = e
            while j >= s and BLANK_RE.match(seg_lines[j]): j -= 1
            if j >= s: yield offs[s], offs[j] + len(seg_lines[j])
            block_start = i + 1
            continue

        if i + 1 < len(seg_lines):
            nxt = seg_lines[i + 1]
            from .org_detection import _is_all_caps_line, _starts_with_starter
            if _starts_with_starter(nxt) and _is_all_caps_line(nxt):
                s, e = block_start, i
                while s < e and BLANK_RE.match(seg_lines[s]): s += 1
                j = e
                while j >= s and BLANK_RE.match(seg_lines[j]): j -= 1
                if j >= s: yield offs[s], offs[j] + len(seg_lines[j])
                block_start = i + 1
                continue

    if block_start < len(seg_lines):
        s, e = block_start, len(seg_lines) - 1
        while s <= e and BLANK_RE.match(seg_lines[s]): s += 1
        while e >= s and BLANK_RE.match(seg_lines[e]): e -= 1
        if s <= e: yield offs[s], offs[e] + len(seg_lines[e])

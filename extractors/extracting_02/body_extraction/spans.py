from typing import Tuple, List, Dict
from spacy.tokens import Doc
import re



# Title-like lines that usually start a new item
TITLE_LINE_RE = re.compile(
    r'^\s*(?:Acordo|Contrato|Portaria|Aviso|CCT|Conven[cç][aã]o|Conven[cç]oes?|Regulamento)\b',
    re.IGNORECASE
)

# Section names that also act as hard stops (lowercased, no trailing colon)
_SECTION_STOPS = {
    "despachos",
    "despacho",
    "portarias de condições de trabalho",
    "portarias de condicoes de trabalho",
    "portarias de extensão",
    "portarias de extensao",
    "convenções coletivas de trabalho",
    "convencoes coletivas de trabalho",
    "regulamentos de extensão",
    "regulamentos de extensao",
    "regulamentos de condições mínimas",
    "regulamentos de condicoes minimas",
}

def _is_all_caps_header(line: str) -> bool:
    t = line.strip()
    if not t:
        return False
    letters = [ch for ch in t if ch.isalpha()]
    if len(letters) < 6:
        return False
    return all(ch == ch.upper() for ch in letters)

def grow_to_block_boundary(
    body_text: str,
    start: int,
    end: int,
    block_start: int,
    block_end: int,
) -> Tuple[int, int, bool]:
    """
    Grow [start, end) forward within [block_start, block_end) until a structural stop:
      - next title-like line (TITLE_LINE_RE),
      - next section cue (_SECTION_STOPS),
      - next ORG-like all-caps header,
      - or block_end.
    Returns (new_start, new_end, used_boundary).
    """
    start = max(block_start, min(start, block_end))
    end = max(start, min(end, block_end))

    seg = body_text[end:block_end]
    off = end
    for ln in seg.splitlines(keepends=True):
        stripped = ln.strip()
        if TITLE_LINE_RE.match(stripped):
            return start, off, True
        if stripped.rstrip(":").lower() in _SECTION_STOPS:
            return start, off, True
        if _is_all_caps_header(stripped):
            return start, off, True
        off += len(ln)
    return start, block_end, False

def resolve_overlaps_in_block(results_in_block: List[Dict]) -> None:
    """
    In-place: sort by start; if two items overlap, clamp the earlier one's end
    to the next start so items become sequential within the block.
    """
    results_in_block.sort(key=lambda r: (r["body_span"]["start"], r["body_span"]["end"]))
    for i in range(len(results_in_block) - 1):
        cur = results_in_block[i]["body_span"]
        nxt = results_in_block[i + 1]["body_span"]
        if cur["end"] > nxt["start"]:
            cur["end"] = max(cur["start"], nxt["start"])

def expand_to_sentence(doc: Doc, start: int, end: int) -> Tuple[int, int]:
    """
    Expand a [start, end) character span to full sentence boundaries using doc.sents.
    - If no sentencizer is present (no sents), returns (start, end).
    - If start==end, snaps to the sentence covering that point.
    - If span crosses multiple sentences, expands from the first to the last.
    """
    # Safety clamps
    start = max(0, min(len(doc.text), start))
    end   = max(start, min(len(doc.text), end))

    # If the pipeline has no sentence boundaries, just return as-is
    if not hasattr(doc, "sents"):
        return start, end

    # Find the first and last sentence covering the span
    first_sent_start = None
    last_sent_end = None

    # Fast path for point spans: pick the sentence containing `start`
    if start == end:
        for sent in doc.sents:
            s0 = sent.start_char
            s1 = sent.end_char
            if s0 <= start < s1:
                return s0, s1
        # If no sentence found (shouldn't happen), return as-is
        return start, end

    # General case: expand to cover all sentences intersecting the span
    for sent in doc.sents:
        s0 = sent.start_char
        s1 = sent.end_char
        # Intersects if there is any overlap between [start,end) and [s0,s1)
        if not (end <= s0 or s1 <= start):
            if first_sent_start is None:
                first_sent_start = s0
            last_sent_end = s1

    if first_sent_start is not None and last_sent_end is not None:
        return first_sent_start, last_sent_end

    # Fallback: if the span didn’t intersect any sentence (e.g., odd tokenization),
    # try snapping to the sentence containing `start`.
    for sent in doc.sents:
        s0 = sent.start_char
        s1 = sent.end_char
        if s0 <= start < s1:
            return s0, s1

    # Final fallback
    return start, end

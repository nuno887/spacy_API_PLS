# sections_instruments.py
from dataclasses import dataclass
from typing import List, Optional, Tuple
import re

from .items import (
    find_item_char_spans,
    clean_item_text,
    _looks_like_item_start,   # reuse your heuristic for “just text” items
)

# --- Recognize explicit instrument titles (minimal but robust) ---
# Accepts: Acordo/Contrato Coletivo de Trabalho, Acordo de Empresa/Adesão, Aditamento
# Number/year token variants: n.º | nº | n.o | n. | no  (colon optional)
TITLE_RE = re.compile(
    r"""^\s*
        (?P<type>
            (?:Acordo|Contrato)\s+Coletivo\s+de\s+Trabalho
            |Acordo\s+de\s+Empresa
            |Acordo\s+de\s+Ades[aã]o
            |Aditamento
        )
        [\s,]* (?:n[\.\º°]?\s*o?)? \s*
        (?P<number>\d+)\s*/\s*(?P<year>\d{2,4})
        \s*:?
        \s*$""",
    re.IGNORECASE | re.VERBOSE,
)

# Lightweight keyword probe for “just text” items (no formal title line)
INSTR_KEYWORDS = tuple([
    "acordo coletivo", "contrato coletivo",
    "acordo de empresa", "acordo de ades", "aditamento",
])

@dataclass(frozen=True)
class InstrItem:
    start_char: int
    end_char: int
    span_title: Optional[Tuple[int, int]]
    span_body: Tuple[int, int]
    title_surface: Optional[str]
    instrument_type: Optional[str]
    number: Optional[int]
    year: Optional[int]
    confidence: float

@dataclass(frozen=True)
class InstrSeries:
    start_char: int
    end_char: int
    org_key: Optional[str]          
    items: List[InstrItem]


def _split_title_body(full_text: str, s: int, e: int) -> Tuple[Optional[Tuple[int,int]], Tuple[int,int], Optional[str], Optional[re.Match]]:
    """Return (span_title, span_body, title_surface, title_match) for an item block."""
    block = full_text[s:e]
    # Use first non-empty line as a candidate title
    rel = 0
    for ln in block.splitlines(keepends=True):
        ln_stripped = ln.strip()
        ln_len = len(ln)
        if ln_stripped:
            line_abs_start = s + rel
            line_abs_end   = s + rel + ln_len
            title_m = TITLE_RE.match(ln_stripped)
            if title_m:
                # title is this line; body starts after it
                body_start = line_abs_end
                body_span = (body_start, e)
                return ( (line_abs_start, line_abs_end),
                         body_span,
                         ln_stripped,
                         title_m )
            # Not a formal title; we’ll treat the whole block as “just text”
            break
        rel += ln_len

    # No formal title; treat full block as body
    return (None, (s, e), None, None)


def detect_instrument_series(
    full_text: str,
    org_key: Optional[str],
    region_span: Tuple[int, int],
    next_heading_starts: set,
) -> Optional[InstrSeries]:
    """
    Build an instrument series inside [region_span), using your existing item segmentation.
    Returns None if no instrument items were found.
    """
    region_start, region_end = region_span

    items: List[InstrItem] = []
    for it_s, it_e in find_item_char_spans(full_text, region_start, region_end, next_heading_starts):
        span_title, span_body, title_surface, m = _split_title_body(full_text, it_s, it_e)

        if m:
            # Explicit title → high confidence, parse fields
            instr_type = m.group("type")
            num = int(m.group("number"))
            yr_raw = m.group("year")
            year = int(yr_raw) if len(yr_raw) == 4 else (2000 + int(yr_raw) if int(yr_raw) <= 50 else 1900 + int(yr_raw))
            conf = 1.0
        else:
            # No formal title; infer from first sentence/keywords
            snippet = clean_item_text(full_text[span_body[0]:min(span_body[0]+160, span_body[1])]).lower()
            instr_type = None
            for kw in INSTR_KEYWORDS:
                if kw in snippet:
                    instr_type = kw  # lightweight label; you can map to canonical later
                    break
            num = None
            year = None
            conf = 0.7 if instr_type else 0.6  # slight bump if keyword found

        items.append(InstrItem(
            start_char=it_s,
            end_char=it_e,
            span_title=span_title,
            span_body=span_body,
            title_surface=title_surface,
            instrument_type=instr_type,
            number=num,
            year=year,
            confidence=conf,
        ))

    if not items:
        return None

    series = InstrSeries(
        start_char=items[0].start_char,
        end_char=items[-1].end_char,
        org_key=org_key,
        items=items,
    )
    return series

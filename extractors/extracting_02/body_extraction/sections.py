# body_extraction/sections.py

from typing import List, Tuple, Dict, Iterable
from spacy.tokens import Doc
from spacy.matcher import PhraseMatcher
from .body_taxonomy import BODY_SECTIONS
from .matchers import get_header_matcher


def _label_variants(section_label_surface: str) -> List[str]:
    """
    Build minimal robust variants from the printed section label:
    - strip trailing colon
    - include with and without colon
    """
    label = (section_label_surface or "").strip()
    if label.endswith(":"):
        label = label[:-1].strip()
    variants = []
    if label:
        variants.append(label)
        variants.append(f"{label}:")
    return variants


def find_section_cuts_in_band(
    nlp,
    body_text: str,
    band_start: int,
    band_end: int,
    section_keys: Iterable[str],
) -> List[Tuple[int, str]]:
    """
    Return list of (abs_start_pos, section_key) where a header alias for that section
    appears inside the band.
    """
    text = body_text[band_start:band_end]
    doc: Doc = nlp.make_doc(text)

    pm = get_header_matcher(nlp, section_keys)
    hits: List[Tuple[int, str]] = []

    for match_id, s, _ in pm(doc):
        label = doc.vocab.strings[match_id]  # e.g. "HDR_Convencoes"
        # label format is "HDR_<canonical>"
        _, key = label.split("_", 1)
        abs_pos = band_start + doc[s].idx
        hits.append((abs_pos, key))

    hits.sort(key=lambda x: x[0])
    return hits


def build_section_blocks_in_band(
    nlp,
    body_text: str,
    band_start: int,
    band_end: int,
    section_keys_in_org: List[str],
) -> Dict[str, Tuple[int, int]]:
    """
    Partition [band_start, band_end) into ordered blocks per found header.
    Returns mapping {section_key: (start, end)} for keys that had a header hit.
    """
    cuts = find_section_cuts_in_band(nlp, body_text, band_start, band_end, section_keys_in_org)
    blocks: Dict[str, Tuple[int, int]] = {}

    if not cuts:
        return blocks

    for i, (pos, key) in enumerate(cuts):
        start = pos
        end = cuts[i + 1][0] if i + 1 < len(cuts) else band_end
        blocks[key] = (start, end)

    return blocks


def find_section_block_in_band_spacy(
    nlp,
    body_text: str,
    band_start: int,
    band_end: int,
    section_label_surface: str,
    section_key: str,
) -> Tuple[int, int]:
    """
    Fallback: find the start of a single section inside [band_start, band_end)
    using (1) label variants and (2) taxonomy aliases (header_aliases) for that key.
    Returns (section_start_abs, band_end). If nothing matches, returns (band_start, band_end).
    """
    band_text = body_text[band_start:band_end]
    doc = nlp.make_doc(band_text)

    # Build patterns: label variants first (most specific), then taxonomy header aliases
    variants = set(_label_variants(section_label_surface))
    sec = BODY_SECTIONS.get(section_key)
    if sec:
        for alias in sec.header_aliases:
            if alias:
                variants.add(alias)

    if not variants:
        return band_start, band_end

    matcher = PhraseMatcher(doc.vocab, attr="LOWER")
    for i, v in enumerate(sorted(variants, key=len, reverse=True)):
        matcher.add(f"SEC_{i}", [nlp.make_doc(v)])

    hits = matcher(doc)
    if not hits:
        return band_start, band_end

    # Earliest occurrence by character index
    best = min(hits, key=lambda h: doc[h[1]].idx)
    _, s, _ = best
    start_char_in_band = doc[s].idx
    return band_start + start_char_in_band, band_end

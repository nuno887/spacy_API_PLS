# body_extraction/sections.py

from typing import Tuple, List
from spacy.matcher import PhraseMatcher

# ---------------------------------------------------------------------
# Visual cue lists to anchor sections even if the literal header is missing
# Keys = canonical section names from your taxonomy (e.g., "Despachos")
# ---------------------------------------------------------------------
SECTION_CUE_VARIANTS = {
    "Despachos": {
        "Despachos", "Despacho", "Despacho Conjunto",
        "Despacho N.º", "Despacho Nº", "Despacho No", "Despacho n.º", "Despacho nº",
        "Despacho Conjunto N.º", "Despacho Conjunto Nº", "Despacho Conjunto n.º", "Despacho Conjunto nº",
    },

    "PortariasCondições": {
        "Portarias de Condições de Trabalho", "Portarias de Condições", "Portaria de Condições",
        "Portaria", "Portaria N.º", "Portaria Nº", "Portaria n.º", "Portaria nº",
    },

    "PortariasExtensao": {
        "Portarias de Extensão", "Portaria de Extensão",
        "Portaria de Extensão N.º", "Portaria de Extensão Nº", "Portaria de Extensão n.º", "Portaria de Extensão nº",
        "Portaria de Extensao", "Portaria de Extensao N.º", "Portaria de Extensao Nº", "Portaria de Extensao n.º",
    },

    "Convencoes": {
        "Convenção Coletiva de Trabalho", "Convenção Colectiva de Trabalho",
        "Contrato Coletivo de Trabalho", "Contrato Colectivo de Trabalho", "CCT",
        "Acordo de Empresa", "Acordo de Empresa N.º", "Acordo de Empresa nº",
    },

    "RegulamentosCondicoesMinimas": {
        "Regulamentos de Condições Mínimas", "Regulamentos de Condicoes Minimas",
        "Regulamento de Condições Mínimas", "Regulamento de Condicoes Minimas",
    },

    "RegulamentosExtensao": {
        "Regulamentos de Extensão", "Regulamentos de Extensao",
        "Regulamento de Extensão", "Regulamento de Extensao",
        "Regulamento de Extensão N.º", "Regulamento de Extensao N.º",
    },
}


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


def find_section_block_in_band_spacy(
    nlp,
    body_text: str,
    band_start: int,
    band_end: int,
    section_label_surface: str,
    section_key: str,
) -> Tuple[int, int]:
    """
    Use spaCy + PhraseMatcher to find the start of the section inside [band_start, band_end).
    Strategy:
      1) Try exact section label (with/without colon).
      2) If not found, try section-specific title/cue variants (e.g., "Despacho Conjunto N.º").
    Returns (section_start_abs, band_end). If nothing matches, returns (band_start, band_end).
    """
    # Slice the band
    band_text = body_text[band_start:band_end]
    doc = nlp.make_doc(band_text)

    # Build patterns: label variants first, then cue variants
    variants = set(_label_variants(section_label_surface))
    variants |= SECTION_CUE_VARIANTS.get(section_key, set())

    # If nothing to match, return entire band
    if not variants:
        return band_start, band_end

    # Build PhraseMatcher
    matcher = PhraseMatcher(doc.vocab, attr="LOWER")
    for i, v in enumerate(sorted(variants, key=len, reverse=True)):
        if not v:
            continue
        matcher.add(f"SEC_{i}", [nlp.make_doc(v)])

    hits = matcher(doc)
    if not hits:
        # No header/cue found → whole band
        return band_start, band_end

    # Pick earliest occurrence by raw character index
    best = min(hits, key=lambda h: doc[h[1]].idx)
    _, s, _ = best
    start_char_in_band = doc[s].idx
    return band_start + start_char_in_band, band_end

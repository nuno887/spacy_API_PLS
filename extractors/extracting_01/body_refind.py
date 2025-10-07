# body_refind.py
# Robust re-anchoring using normalized shadow text + regex over BODY-ONLY doc.
# Handles line breaks, NBSPs, diacritics, and common ordinal glyph variants, while
# preserving strict ALL-CAPS validation for ORG/SUBORG on the *original* text.

import re
import unicodedata
from typing import Dict, List, Tuple, Callable, Optional
from spacy.tokens import Doc
from .models import BodyItem

__all__ = ["build_body_via_sumario_spacy"]

# -------------------- ALL-CAPS gate (for ORG / ORG_SECUNDARIA) --------------------

_caps_token_rx = re.compile(r"[A-Za-zÀ-ÿ]")

def _is_all_caps_token(tok: str) -> bool:
    has_alpha = False
    for ch in tok:
        if ch.isalpha():
            has_alpha = True
            if ch != ch.upper():
                return False
    return has_alpha

def _passes_all_caps_gate(text: str) -> bool:
    # reject spans containing a blank line
    if re.search(r"\n\s*\n", text):
        return False
    for tok in re.split(r"\s+", text.strip()):
        if _caps_token_rx.search(tok) and not _is_all_caps_token(tok):
            return False
    return True

# -------------------- Normalization helpers --------------------

_WHITESPACE_RX = re.compile(r"\s+")

def _strip_diacritics(s: str) -> str:
    # Convert to NFKD then drop combining marks
    nk = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nk if not unicodedata.combining(ch))

def _canonical_glyphs(s: str) -> str:
    # Map ordinal/degree glyphs to ASCII-ish (for matching only)
    # Also heal common OCR variants globally (safe for matching)
    s = s.replace("\u00A0", " ")  # NBSP -> space
    s = s.replace("º", "o").replace("°", "o").replace("ª", "a")
    return s

def _heal_hyphen_linebreak_pairs(original: str, i: int) -> Tuple[bool, int]:
    """
    If we see a discretionary hyphen at EOL like '-\\n' or '-\\r\\n', signal to skip both
    the hyphen and the line break (joining words without space). Return (skip, new_i).
    """
    if original[i] != "-":
        return (False, i)
    # Look ahead for CRLF or LF
    if i + 1 < len(original) and original[i + 1] == "\n":
        return (True, i + 2)  # skip '-' and '\n'
    if i + 2 < len(original) and original[i + 1] == "\r" and original[i + 2] == "\n":
        return (True, i + 3)  # skip '-' + CRLF
    return (False, i)

def _build_normalized_with_map(original: str) -> Tuple[str, List[int]]:
    """
    Build a shadow 'normalized' string used only for matching, plus a map norm_idx->orig_idx.
    Normalization steps:
      - NFKC
      - heal hyphen+linebreak joins (CULTU-\\nRA -> CULTURA)
      - fold ordinal glyphs (º, °, ª) to ASCII
      - strip diacritics
      - lowercase
      - collapse any whitespace (incl. newlines, tabs, NBSP) to a single space
    """
    # First pass: NFKC
    src = unicodedata.normalize("NFKC", original)

    norm_chars: List[str] = []
    idx_map: List[int] = []

    i = 0
    prev_was_space = False
    while i < len(src):
        ch = src[i]

        # Heal hyphen + line break (join words, no space)
        skip, new_i = _heal_hyphen_linebreak_pairs(src, i)
        if skip:
            i = new_i
            prev_was_space = False  # we're joining words
            continue

        # Normalize whitespace to a single space
        if ch.isspace():
            if not prev_was_space and len(norm_chars) > 0:
                norm_chars.append(" ")
                # map this normalized space to the current original index
                idx_map.append(i)
            elif not prev_was_space and len(norm_chars) == 0:
                # avoid leading space; also map it consistently
                norm_chars.append(" ")
                idx_map.append(i)
            prev_was_space = True
            i += 1
            continue

        prev_was_space = False

        # Canonicalize glyphs
        ch2 = _canonical_glyphs(ch)

        # Strip diacritics
        ch3 = _strip_diacritics(ch2)

        # Lowercase
        for out_ch in ch3.lower():
            norm_chars.append(out_ch)
            idx_map.append(i)

        i += 1

    # Collapse any multiple spaces created at the very start/end
    norm = "".join(norm_chars)
    norm = _WHITESPACE_RX.sub(" ", norm).strip()

    # Re-trim idx_map to match trim in norm
    # Find the first and last non-space in the constructed norm_chars to approximate mapping
    # (Since we collapsed whitespace during construction, leading/trailing spaces are minimal.)
    if norm:
        # nothing extra to trim because we already stripped. idx_map length equals len(norm_chars before strip)
        # To be safe, rebuild again with collapse+strip mirrored on idx_map:
        # We'll rebuild norm and idx_map together in a final pass.
        final_norm_chars: List[str] = []
        final_idx_map: List[int] = []
        prev_space = False
        for ch, oi in zip("".join(norm_chars), idx_map):
            if ch.isspace():
                if not prev_space and len(final_norm_chars) > 0:
                    final_norm_chars.append(" ")
                    final_idx_map.append(oi)
                prev_space = True
            else:
                final_norm_chars.append(ch)
                final_idx_map.append(oi)
                prev_space = False
        # strip leading/trailing space
        if final_norm_chars and final_norm_chars[0] == " ":
            final_norm_chars.pop(0)
            final_idx_map.pop(0)
        if final_norm_chars and final_norm_chars[-1] == " ":
            final_norm_chars.pop()
            final_idx_map.pop()

        norm = "".join(final_norm_chars)
        idx_map = final_idx_map
    else:
        idx_map = []

    return norm, idx_map

def _normalize_phrase_for_regex(s: str) -> str:
    """
    Same normalization as the body (conceptually), but without building a map:
      - NFKC -> glyph canonicalization -> strip diacritics -> lowercase -> collapse whitespace
    Then escape regex metachars and replace internal spaces with '\\s+'.
    """
    s = unicodedata.normalize("NFKC", s)
    s = _canonical_glyphs(s)
    s = _strip_diacritics(s).lower()
    s = _WHITESPACE_RX.sub(" ", s).strip()
    # Escape regex meta chars
    parts = [re.escape(p) for p in s.split(" ") if p]
    if not parts:
        return ""
    return r"\s+".join(parts)

# -------------------- Matching over normalized text --------------------

def _gather_regex_candidates(
    norm_body: str,
    idx_map: List[int],
    phrases: List[str],
    *,
    enforce_caps_on_original: bool,
    original_text: str,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    For each phrase, compile a regex over the normalized body that tolerates any whitespace.
    Map normalized match spans back to original text using idx_map. Optionally enforce ALL-CAPS on original.
    Returns: phrase -> list[(orig_start, orig_end)]
    """
    out: Dict[str, List[Tuple[int, int]]] = {}
    for p in phrases:
        pat = _normalize_phrase_for_regex(p)
        if not pat:
            continue
        rx = re.compile(pat)
        hits: List[Tuple[int, int]] = []
        for m in rx.finditer(norm_body):
            nst, nen = m.start(), m.end()
            if nst >= len(idx_map) or nen - 1 >= len(idx_map):
                continue
            # Map normalized positions to original indices
            orig_start = idx_map[nst]
            orig_end = idx_map[nen - 1] + 1  # end-exclusive
            if enforce_caps_on_original:
                # run gate on original slice
                if not _passes_all_caps_gate(original_text[orig_start:orig_end]):
                    continue
            hits.append((orig_start, orig_end))
        if hits:
            hits.sort(key=lambda t: (t[0], -t[1]))
            out[p] = hits
    return out

# -------------------- Main --------------------

def build_body_via_sumario_spacy(
    doc: Doc,
    roster: Dict[str, object],
    pattern_tokenizer=None,            # kept for compatibility; not used in regex approach
    include_local_details: bool = False
) -> List[BodyItem]:
    """
    Match the roster's ORG / ORG_SECUNDARIA / DOC strings in the BODY-ONLY `doc` using a
    normalized shadow text + regex that tolerates whitespace, diacritics, and common glyph variants.
    ORG/SUBORG matches are still validated via ALL-CAPS on the original text.
    Assign in roster order and slice sections by DOC anchors. If no DOCs are found for an ORG,
    fall back to slicing by ORG_SECUNDARIA.
    """
    full_text = doc.text

    # Build simple blueprint directly from roster (strings-only, ordered)
    blueprint = [
        {
            "org_text": o.get("org_text", ""),
            "suborgs": [{"text": t} for t in o.get("suborg_texts", [])],
            "docs":    [{"text": t} for t in o.get("doc_texts",    [])],
        }
        for o in roster.get("orgs", [])
    ]

    # 1) Build normalized shadow body + index map
    norm_body, idx_map = _build_normalized_with_map(full_text)

    # 2) Collect phrases from roster in Sumário order
    org_phrases = [b["org_text"] for b in blueprint]
    sub_phrases = [s["text"] for b in blueprint for s in b["suborgs"]]
    doc_phrases = [d["text"] for b in blueprint for d in b["docs"]]

    # 3) Gather candidates via regex on normalized body; validate caps for ORG/SUBORG
    org_cands = _gather_regex_candidates(
        norm_body, idx_map, org_phrases,
        enforce_caps_on_original=True,
        original_text=full_text,
    )
    sub_cands = _gather_regex_candidates(
        norm_body, idx_map, sub_phrases,
        enforce_caps_on_original=True,
        original_text=full_text,
    )
    doc_cands = _gather_regex_candidates(
        norm_body, idx_map, doc_phrases,
        enforce_caps_on_original=False,
        original_text=full_text,
    )

    # 4) Assign ORGs in roster order with a moving cursor (left-to-right, non-overlapping)
    assigned_orgs: List[Dict] = []
    cursor = 0

    def next_org_start(i: int) -> int:
        for j in range(i + 1, len(assigned_orgs)):
            if assigned_orgs[j].get("assigned"):
                return assigned_orgs[j]["assigned"][0]
        return len(full_text)

    for b in blueprint:
        phrase = b["org_text"]
        cands = org_cands.get(phrase, [])
        chosen = None
        for st, en in cands:
            if st >= cursor:
                chosen = (st, en)
                cursor = en
                break
        assigned_orgs.append({**b, "assigned": chosen})

    # 5) For each ORG section, assign SUBORGs and DOCs; slice sections using DOC anchors,
    #    or fall back to slicing by SUBORGs if no DOCs are present.
    body_items: List[BodyItem] = []
    order_idx = 1

    for i, org_entry in enumerate(assigned_orgs):
        org_span = org_entry["assigned"]
        if org_span is None:
            continue
        org_st, org_en = org_span
        section_end = next_org_start(i)

        # SUBORGs (collect chosen hits for fallback slicing)
        sub_assignments: List[Tuple[int, int, str]] = []
        sub_cursor = org_st
        for sub in org_entry["suborgs"]:
            phrase = sub["text"]
            hits = [h for h in sub_cands.get(phrase, []) if org_st <= h[0] < section_end]
            chosen = None
            for st, en in hits:
                if st >= sub_cursor:
                    chosen = (st, en, phrase)
                    sub_cursor = en
                    break
            if chosen:
                sub_assignments.append(chosen)

        # DOCs drive slicing (primary mode)
        doc_cursor = org_en
        doc_assignments: List[Tuple[int, int]] = []
        for d in org_entry["docs"]:
            phrase = d["text"]
            hits = [h for h in doc_cands.get(phrase, []) if org_en <= h[0] < section_end]
            chosen = None
            for st, en in hits:
                if st >= doc_cursor:
                    chosen = (st, en)
                    doc_cursor = en
                    break
            if chosen:
                doc_assignments.append(chosen)

        # Tiny safety net: if no DOCs found inside section, try a short look-back window
        if not doc_assignments:
            lookback = max(0, org_en - 120)
            for d in org_entry["docs"]:
                phrase = d["text"]
                for st, en in doc_cands.get(phrase, []):
                    if lookback <= st < org_en:
                        doc_assignments = [(st, en)]
                        break
                if doc_assignments:
                    break

        # Fallback: slice by SUBORGs if there are no DOCs
        if not doc_assignments:
            if sub_assignments:
                for k, (sub_st, sub_en, sub_phrase) in enumerate(sub_assignments):
                    seg_end = sub_assignments[k + 1][0] if k + 1 < len(sub_assignments) else section_end
                    body_items.append(BodyItem(
                        org_text=" ".join(org_entry["org_text"].split()),
                        org_start=org_st,
                        org_end=org_en,
                        section_id=org_st,
                        doc_title=sub_phrase,               # identify the chunk by its suborg
                        doc_start=sub_st,
                        doc_end=sub_en,
                        relation="CONTAINS",                # driver is suborg here
                        slice_text=full_text[sub_st:seg_end].strip(),
                        slice_start=sub_st,
                        slice_end=seg_end,
                        order_index=order_idx,
                    ))
                    order_idx += 1
            continue  # move to next ORG

        # Primary: slice by DOCs
        first_doc_st, first_doc_en = doc_assignments[0]
        first_end = (doc_assignments[1][0] if len(doc_assignments) >= 2 else section_end)
        body_items.append(BodyItem(
            org_text=" ".join(org_entry["org_text"].split()),
            org_start=org_st,
            org_end=org_en,
            section_id=org_st,
            doc_title=full_text[first_doc_st:first_doc_en],
            doc_start=first_doc_st,
            doc_end=first_doc_en,
            relation="SECTION_ITEM",
            slice_text=full_text[org_st:first_end].strip(),
            slice_start=org_st,
            slice_end=first_end,
            order_index=order_idx,
        ))
        order_idx += 1

        for j in range(1, len(doc_assignments) - 1):
            cur_st, cur_en = doc_assignments[j]
            nxt_st, _ = doc_assignments[j + 1]
            body_items.append(BodyItem(
                org_text=" ".join(org_entry["org_text"].split()),
                org_start=org_st,
                org_end=org_en,
                section_id=org_st,
                doc_title=full_text[cur_st:cur_en],
                doc_start=cur_st,
                doc_end=cur_en,
                relation="SECTION_ITEM",
                slice_text=full_text[cur_st:nxt_st].strip(),
                slice_start=cur_st,
                slice_end=nxt_st,
                order_index=order_idx,
            ))
            order_idx += 1

        if len(doc_assignments) >= 2:
            last_st, last_en = doc_assignments[-1]
            body_items.append(BodyItem(
                org_text=" ".join(org_entry["org_text"].split()),
                org_start=org_st,
                org_end=org_en,
                section_id=org_st,
                doc_title=full_text[last_st:last_en],
                doc_start=last_st,
                doc_end=last_en,
                relation="SECTION_ITEM",
                slice_text=full_text[last_st:section_end].strip(),
                slice_start=last_st,
                slice_end=section_end,
                order_index=order_idx,
            ))
            order_idx += 1

    return body_items

# body_extraction/anchors.py

from typing import List, Tuple, Dict, Any

def build_body_org_bands_from_relations(
    body_len: int,
    relations_org_to_org: List[Dict[str, Any]],
    body_offset: int,
) -> List[Tuple[int, int, Dict[str, Any]]]:
    """
    Build BODY-relative ORG bands from relations_org_to_org.
    - body_len: length of body_text
    - relations_org_to_org: payload["relations_org_to_org"]
      each item may have r["body"]["span"]["start"/"end"]
    - body_offset: payload["body"]["span"]["start"], to convert full-text
      absolute coords into body_text-relative coords.

    Returns a list of (start, end, meta) where start/end are BODY-relative.
    If no relations, returns a single band covering the whole body.
    """
    # Collect unique body hits with spans in full-text coords
    hits = []
    seen = set()
    for r in relations_org_to_org or []:
        b = r.get("body")
        if not b:
            continue
        sp = b.get("span") or {}
        s_abs = sp.get("start")
        e_abs = sp.get("end")
        if s_abs is None or e_abs is None:
            continue
        key = (s_abs, e_abs)
        if key in seen:
            continue
        seen.add(key)
        # Convert to body-relative
        s_rel = max(0, s_abs - body_offset)
        # We only need the anchor start to cut bands; keep meta for context
        hits.append({
            "start_rel": s_rel,
            "meta": b,  # keep the original 'body' object as meta
        })

    # No anchors → whole body as a single band
    if not hits:
        return [(0, body_len, {"span": {"start": 0, "end": body_len}, "surface_raw": ""})]

    # Sort by start and create bands [start_i, start_{i+1})
    hits.sort(key=lambda h: h["start_rel"])
    bands: List[Tuple[int, int, Dict[str, Any]]] = []
    for i, h in enumerate(hits):
        a = min(max(0, h["start_rel"]), body_len)
        z = hits[i + 1]["start_rel"] if i + 1 < len(hits) else body_len
        z = min(max(a, z), body_len)
        bands.append((a, z, h["meta"]))
    return bands


def pick_band_for_section(
    bands: List[Tuple[int, int, Dict[str, Any]]],
    section_start_body_rel: int,
    default_band: Tuple[int, int, Dict[str, Any]],
) -> Tuple[int, int, Dict[str, Any]]:
    """
    Given body-relative ORG bands and a section heading start (also body-relative),
    return the band that contains it; else default_band.
    """
    for a, z, meta in bands:
        if a <= section_start_body_rel < z:
            return (a, z, meta)
    return default_band

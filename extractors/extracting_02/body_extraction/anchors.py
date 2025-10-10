from typing import List, Tuple, Dict, Any

def build_body_org_bands(payload: Dict[str, Any]) -> List[Tuple[int, int, Dict]]:
    """Produce BODY bands [start, next_start) from relations body spans; fallback to whole body."""
    body = payload["body"]
    body_start, body_end = body["span"]["start"], body["span"]["end"]
    rels = payload.get("relations_org_to_org", [])

    bodys = []
    seen = set()
    for r in rels:
        b = r.get("body")
        if not b: continue
        key = (b["span"]["start"], b["span"]["end"])
        if key in seen: continue
        seen.add(key)
        bodys.append(b)

    bodys.sort(key=lambda b: b["span"]["start"])
    if not bodys:
        return [(body_start, body_end, {"span": {"start": body_start, "end": body_end}, "surface_raw": ""})]

    bands = []
    for i, b in enumerate(bodys):
        a = b["span"]["start"]
        z = bodys[i+1]["span"]["start"] if i+1 < len(bodys) else body_end
        bands.append((a, z, b))
    return bands

def pick_band_for_section(bands: List[tuple], section_start: int, default_band: tuple) -> tuple:
    for a, z, obj in bands:
        if a <= section_start < z:
            return (a, z, obj)
    return default_band

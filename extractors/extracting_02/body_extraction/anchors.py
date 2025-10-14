# body_extraction/anchors.py

from typing import List, Dict, Any, Tuple, Optional
import re
from entities.normalization import canonical_org_key


# --- Spaced ALL-CAPS banner collapse -----------------------------------------
# e.g., "S E C R E T A R I A  R E G I O N A L" -> "SECRETARIA REGIONAL"
_SPACED_CAPS_RUN = re.compile(r"(?:\b[A-ZÀ-Ý]\b[ \u00A0])+[A-ZÀ-Ý]\b")

def _collapse_spaced_caps(s: str) -> str:
    """
    Collapse runs of single uppercase letters separated by spaces/NBSP into a word.
    Keeps normal words intact. Also squeezes redundant spaces.
    """
    if not s:
        return ""
    def _join_run(m: re.Match) -> str:
        letters = re.findall(r"[A-ZÀ-Ý]", m.group(0))
        return "".join(letters)
    out = _SPACED_CAPS_RUN.sub(_join_run, s)
    # squeeze multiple spaces/nbsp into single space
    out = re.sub(r"[ \u00A0]{2,}", " ", out)
    return out.strip()


def build_body_org_bands_from_relations(
    body_len: int,
    relations_org_to_org: List[Dict[str, Any]],
    body_offset: int,
) -> List[Dict[str, Any]]:
    """
    Build BODY-relative ORG bands from relations_org_to_org.

    Output (list of dicts):
      [
        {
          "key": <canonical org key from relation["key"]>,
          "band": (start_body, end_body),
          "meta": {
            "body_org":    {"surface_raw": <str>, "span": {"start": <int>, "end": <int>}},  # BODY-relative
            "sumario_org": {"surface_raw": <str>, "span": {"start": <int>, "end": <int>}},  # Sumário coords
            "relation":    <original relation dict>,
          },
        },
        ...
      ]
    """
    anchors: List[Dict[str, Any]] = []
    seen_starts: set[int] = set()

    for rel in relations_org_to_org or []:
        key = rel.get("key")
        body_side = rel.get("body") or {}
        sumario_side = rel.get("sumario") or {}

        bspan_abs = (body_side.get("span") or {})
        s_abs = bspan_abs.get("start")
        e_abs = bspan_abs.get("end")
        if s_abs is None or e_abs is None:
            continue

        # Convert to BODY-relative
        s_rel = max(0, int(s_abs) - int(body_offset))
        e_rel = max(s_rel, int(e_abs) - int(body_offset))
        s_rel = min(s_rel, body_len)
        e_rel = min(e_rel, body_len)

        if s_rel in seen_starts:
            continue
        seen_starts.add(s_rel)

        anchors.append({
            "key": key,
            "start_rel": s_rel,
            "body_org": {
                "surface_raw": body_side.get("surface_raw", ""),
                "span": {"start": s_rel, "end": e_rel},
            },
            "sumario_org": {
                "surface_raw": sumario_side.get("surface_raw", ""),
                "span": sumario_side.get("span") or {},
            },
            "relation": rel,
        })

    if not anchors:
        return [{
            "key": None,
            "band": (0, body_len),
            "meta": {
                "body_org":   {"surface_raw": "", "span": {"start": 0, "end": body_len}},
                "sumario_org": {"surface_raw": "", "span": {"start": 0, "end": 0}},
                "relation": None,
            }
        }]

    anchors.sort(key=lambda a: a["start_rel"])
    bands: List[Dict[str, Any]] = []
    for i, a in enumerate(anchors):
        a_start = a["start_rel"]
        a_end = anchors[i + 1]["start_rel"] if i + 1 < len(anchors) else body_len
        a_end = max(a_start, min(a_end, body_len))

        bands.append({
            "key": a["key"],
            "band": (a_start, a_end),
            "meta": {
                "body_org": a["body_org"],
                "sumario_org": a["sumario_org"],
                "relation": a["relation"],
            }
        })

    return bands


def _normalize_bands(body_bands: Any) -> List[Dict[str, Any]]:
    """
    Normalize various shapes into a list of {'key','band','meta'} dicts.

    Accepts:
      - dict: {key: {'band': (a,z), 'meta': {...}}, ...}
      - list of dicts: [{'key': key, 'band': (a,z), 'meta': {...}}, ...]
      - list of tuples: [(a, z, meta), ...]  # meta should have surface to derive key
    """
    if isinstance(body_bands, dict):
        out = []
        for k, v in body_bands.items():
            out.append({
                "key": k,
                "band": v.get("band"),
                "meta": v.get("meta") or {},
            })
        return out

    out: List[Dict[str, Any]] = []
    for b in list(body_bands or []):
        if isinstance(b, dict):
            k = b.get("key")
            band = b.get("band")
            meta = b.get("meta") or {}
            if not k:
                # Try derive from body or sumário surface
                surf = (
                    meta.get("body_org", {}).get("surface_raw")
                    or meta.get("sumario_org", {}).get("surface_raw")
                    or meta.get("surface_raw")
                    or ""
                ).strip()
                if surf:
                    surf = _collapse_spaced_caps(surf)
                    k = canonical_org_key(surf)
            out.append({"key": k, "band": band, "meta": meta})
        elif isinstance(b, (tuple, list)) and len(b) >= 2:
            a, z = b[0], b[1]
            meta = b[2] if len(b) > 2 and isinstance(b[2], dict) else {}
            surf = (
                meta.get("surface_raw")
                or meta.get("body_org", {}).get("surface_raw")
                or meta.get("sumario_org", {}).get("surface_raw")
                or ""
            ).strip()
            if surf:
                surf = _collapse_spaced_caps(surf)
                k = canonical_org_key(surf)
            else:
                k = None
            out.append({"key": k, "band": (a, z), "meta": meta})
        # else: unknown shape → skip
    return out


def pick_band_for_section(
    body_bands: Any,
    sumario_pos: Optional[int],
    default_band: Tuple[int, int, Dict[str, Any]],
    org_key: Optional[str] = None,
    org_surface: Optional[str] = None,
) -> Tuple[int, int, Dict[str, Any]]:
    """
    Choose the body band for a section.

    Priority:
      1) org_key lookup
      2) org_surface canonicalized (with spaced-caps collapse)
      3) nearest by sumário position (if provided and available)
      4) default_band
    """
    bands = _normalize_bands(body_bands)
    index = {b.get("key"): b for b in bands if b.get("key")}

    # 1) exact by canonical key
    if org_key and org_key in index:
        b = index[org_key]
        a, z = b["band"]
        return a, z, {"via": "key", **(b.get("meta") or {})}

    # 2) fuzzy by surface → canonical (with spaced-caps collapse)
    if org_surface:
        try:
            cleaned = _collapse_spaced_caps(str(org_surface))
            k = canonical_org_key(cleaned)
        except Exception:
            k = None
        if k and k in index:
            b = index[k]
            a, z = b["band"]
            return a, z, {"via": "surface", **(b.get("meta") or {})}

    # 3) nearest by sumário position (optional; ignored if None)
    if sumario_pos is not None:
        best = None
        best_dist = None
        for b in bands:
            s_org = (((b.get("meta") or {}).get("sumario_org") or {}).get("span") or {}).get("start")
            if s_org is None:
                continue
            d = abs(int(s_org) - int(sumario_pos))
            if best is None or d < best_dist:
                best, best_dist = b, d
        if best:
            a, z = best["band"]
            return a, z, {"via": "pos", **(best.get("meta") or {})}

    # 4) fallback
    a, z, meta = default_band
    return a, z, {"via": "default", **(meta or {})}

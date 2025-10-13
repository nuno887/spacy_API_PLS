# body_extraction/extract.py

from typing import List, Dict, Any, Optional, Tuple
from spacy.tokens import Doc
from entities.normalization import canonical_org_key

from .anchors import build_body_org_bands_from_relations, pick_band_for_section
from .align import locate_candidates_in_window
from .spans import expand_to_sentence
from .scoring import pick_best_candidate
from .sections import find_section_block_in_band_spacy

def _first_nonblank_line(s: str) -> str:
    for ln in s.splitlines():
        t = ln.strip()
        if t:
            return t
    return ""

def _title_anchor(item_text: str) -> str:
    """First non-blank line up to the first colon (if any)."""
    t = _first_nonblank_line(item_text)
    if not t:
        return ""
    return t.split(":", 1)[0].strip() or t

def _pad_band(a: int, z: int, total: int, pad: int = 300) -> Tuple[int, int]:
    """Widen tiny bands to improve hit rate; clamp to [0,total)."""
    if z - a >= 120:
        return a, z
    return max(0, a - pad), min(total, z + pad)

def run_extraction(
    body_text: str,
    sections: List[Dict[str, Any]],
    relations_org_to_org: Optional[List[Dict[str, Any]]],
    nlp,
    body_offset: int = 0,
) -> Dict[str, Any]:
    """
    Extract body spans for each Sumário item using ORG bands, section blocks, and title/anchor matching.
    If matching fails, fall back to the entire section block within the ORG band (no clipping).
    """
    # 1) Build ORG bands (body-relative)
    body_bands = build_body_org_bands_from_relations(
        len(body_text),
        relations_org_to_org or [],
        body_offset,
    )

    results: List[Dict[str, Any]] = []

    # 2) Ensure sentence boundaries exist
    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer", first=True, config={"punct_chars": [".", "!", "?", ";", ":"]})
    body_doc: Doc = nlp(body_text)

    for s in sections:
        sec_path = s.get("path", [])
        sec_key = sec_path[-1] if sec_path else "(unknown)"
        sec_span = s.get("span", {"start": 0, "end": 0})
        sec_start_in_sumario = int(sec_span.get("start", 0))

        # 3) Resolve ORG identity for the section (if available)
        org_surface = None
        if "org_span" in s:
            # ORG-series: use the header text from Sumário
            os = s["org_span"]
            org_surface = (s.get("surface") or [sec_key])[0]
        elif s.get("org_context", {}).get("sumario_org", {}).get("surface_raw"):
            org_surface = s["org_context"]["sumario_org"]["surface_raw"]

        org_key = canonical_org_key(org_surface) if org_surface else None

        # 4) Pick band (key → surface → position → default) and pad if tiny
        default_band = (0, len(body_text), {"surface_raw": "", "span": {"start": 0, "end": len(body_text)}})
        print(f"[BANDS] type={type(body_bands).__name__} sample={body_bands[0] if isinstance(body_bands, list) and body_bands else 'n/a'}")
        a, z, band_meta = pick_band_for_section(
            body_bands=body_bands,
            sumario_pos=sec_start_in_sumario - body_offset,
            default_band=default_band,
            org_key=org_key,
            org_surface=org_surface,
        )
        win_start, win_end = _pad_band(a, z, len(body_text))

        # 5) Iterate items with title-first anchoring
        for it in s.get("items", []):
            title = _title_anchor(it.get("text") or "")
            tried = []
            best = {"method": "none"}

            # (a) title-first
            if title:
                cands = locate_candidates_in_window(nlp, body_text, win_start, win_end, title)
                tried.append(("title", len(cands)))
                b1 = pick_best_candidate(cands)
                if b1.get("method", "none") != "none":
                    best = b1
                    best["anchor_used"] = "title"

            # (b) fallback to full text if needed
            if best.get("method", "none") == "none":
                cands = locate_candidates_in_window(nlp, body_text, win_start, win_end, it["text"])
                tried.append(("full", len(cands)))
                b2 = pick_best_candidate(cands)
                if b2.get("method", "none") != "none":
                    best = b2
                    best["anchor_used"] = "full"

            # 6) Finalize span (or fallback to section block)
            if best.get("method", "none") != "none":
                s0, e0 = int(best["start"]), int(best["end"])
                exp_s, exp_e = expand_to_sentence(body_doc, s0, e0)
                body_span = {"start": exp_s, "end": exp_e}
                method = best.get("method", "exact")
                confidence = float(best.get("confidence", 1.0))
            else:
                # Section-scoped fallback (still inside the band)
                sec_label_surface = (s.get("surface") or [sec_key])[-1]
                sec_start_abs, sec_end_abs = find_section_block_in_band_spacy(
                    nlp,
                    body_text,
                    win_start,
                    win_end,
                    sec_label_surface,
                    section_key=sec_key,
                )
                # If degenerate, return the whole (padded) band with tiny confidence
                if sec_start_abs >= sec_end_abs:
                    sec_start_abs, sec_end_abs = win_start, win_end
                body_span = {"start": sec_start_abs, "end": sec_end_abs}
                method = "section_fallback"
                confidence = 0.10

            results.append({
                "section_path": sec_path,
                "section_name": sec_key,
                "section_span": sec_span,
                "item_text": it["text"],
                "item_span_sumario": it.get("span"),
                "org_context": band_meta,
                "body_span": body_span,
                "confidence": confidence,
                "method": method,
                "diagnostics": {
                    "window": {"start": win_start, "end": win_end, "via": band_meta.get("via")},
                    "anchor_used": best.get("anchor_used", "none"),
                    "tried": tried,
                    "num_candidates": best.get("num_candidates") if "num_candidates" in best else None,
                },
            })

    summary = {
        "found": sum(1 for r in results if r["method"] != "none"),
        "not_found": sum(1 for r in results if r["method"] == "none"),
        "avg_confidence": round(sum(r["confidence"] for r in results) / max(1, len(results)), 4),
    }
    return {"summary": summary, "results": results}

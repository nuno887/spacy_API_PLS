# body_extraction/extract.py

from typing import List, Dict, Any, Optional
from spacy.tokens import Doc

from .anchors import build_body_org_bands_from_relations, pick_band_for_section
from .align import locate_candidates_in_window
from .spans import expand_to_sentence
from .scoring import pick_best_candidate
from .sections import find_section_block_in_band_spacy


def run_extraction(
    body_text: str,
    sections: List[Dict[str, Any]],
    relations_org_to_org: Optional[List[Dict[str, Any]]],
    nlp,
    body_offset: int = 0,  # default keeps backward-compat with older calls
) -> Dict[str, Any]:
    """
    Extract body spans for each Sumário item using ORG bands, section blocks, and title/anchor matching.
    If matching fails, fall back to the entire section block within the ORG band (no clipping).
    """

    # Build ORG bands (body-relative) from relations
    body_bands = build_body_org_bands_from_relations(
        len(body_text),
        relations_org_to_org or [],
        body_offset,
    )

    results: List[Dict[str, Any]] = []

    # Pre-make a lightweight doc for sentence expansion
    body_doc: Doc = nlp.make_doc(body_text)

    for s in sections:
        # Section metadata
        sec_path = s.get("path", [])
        sec_key = sec_path[-1] if sec_path else "(unknown)"
        sec_span = s.get("span", {"start": 0, "end": 0})
        sec_start_in_sumario = int(sec_span.get("start", 0))

        # Pick the band where this section likely lives (by section heading start)
        default_band = (0, len(body_text), {"surface_raw": "", "span": {"start": 0, "end": len(body_text)}})
        a, z, band_meta = pick_band_for_section(body_bands, sec_start_in_sumario - body_offset, default_band)

        # Define a search window inside the band (here we use full band; you can narrow if desired)
        win_start, win_end = a, z

        # Iterate items for this section
        for it in s.get("items", []):
            # Find candidates in the window
            cands = locate_candidates_in_window(nlp, body_text, win_start, win_end, it["text"])
            best = pick_best_candidate(cands)

            # Decide final span
            if best.get("method", "none") != "none":
                s0, e0 = int(best["start"]), int(best["end"])
                exp_s, exp_e = expand_to_sentence(body_doc, s0, e0)
                body_span = {"start": exp_s, "end": exp_e}
                method = best.get("method", "exact")
                confidence = float(best.get("confidence", 1.0))
            else:
                # === SECTION FALLBACK (spaCy-based) ===
                # Prefer surface label from Sumário, else canonical key from path
                sec_label_surface = (s.get("surface_path") or [sec_key])[-1]
                sec_start_abs, sec_end_abs = find_section_block_in_band_spacy(
                    nlp,
                    body_text,
                    a,
                    z,
                    sec_label_surface,
                    section_key=sec_key,
                )
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
                    "window": {"start": win_start, "end": win_end},
                    "num_candidates": len(cands),
                    "first_anchor": (cands[0].get("anchor_used") if cands else None),
                },
            })

    summary = {
        "found": sum(1 for r in results if r["method"] != "none"),
        "not_found": sum(1 for r in results if r["method"] == "none"),
        "avg_confidence": round(sum(r["confidence"] for r in results) / max(1, len(results)), 4),
    }

    return {"summary": summary, "results": results}

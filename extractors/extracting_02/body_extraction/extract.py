from typing import Dict, Any, List, Tuple
import spacy
from spacy.tokens import Doc
from .schemas import ExtractionReport, ExtractionResult
from .anchors import build_body_org_bands, pick_band_for_section
from .align import locate_candidates_in_window
from .scorer import pick_best_candidate
from .postprocess import expand_to_sentence
from .config import ANCHOR_WINDOW_CHARS

def _doc_for_body(nlp, full_text: str, body_span: Dict[str, int]) -> Tuple[Doc, int]:
    """Create a spaCy Doc for the body slice; return doc and its absolute start offset."""
    a, b = body_span["start"], body_span["end"]
    body_text = full_text[a:b]
    # fast tokenizer-only is fine (your pt_core_news_lg with disabled pipes is OK)
    return nlp.make_doc(body_text), a

def run_extraction(text_raw: str, payload: Dict[str, Any], nlp) -> ExtractionReport:
    """
    Use spaCy for tokenization and sentence boundaries (from tokenizer-only doc).
    """
    # Build bands and body doc
    body_bands = build_body_org_bands(payload)
    default_band = body_bands[0]
    body_doc, body_abs0 = _doc_for_body(nlp, text_raw, payload["body"]["span"])

    results: List[ExtractionResult] = []

    sections = payload["sumario"]["sections"]
    for s in sections:
        # Decide band: prefer section's attached body_org when present
        if "org_context" in s and s["org_context"].get("body_org"):
            bobj = s["org_context"]["body_org"]
            sect_band = pick_band_for_section(body_bands, bobj["span"]["start"], default_band)
        else:
            sect_band = pick_band_for_section(body_bands, s["span"]["start"], default_band)

        a, z, _ = sect_band
        band_len = z - a
        if band_len > ANCHOR_WINDOW_CHARS * 2:
            win_start, win_end = a, min(z, a + ANCHOR_WINDOW_CHARS)
        else:
            win_start, win_end = a, z

        for it in s.get("items", []):
            # Locate candidates inside window using spaCy
            cands = locate_candidates_in_window(nlp, text_raw, win_start, win_end, it["text"])
            best = pick_best_candidate(nlp, it["text"], text_raw, cands)

            if best.get("method", "none") != "none":
                # Expand to sentence boundaries using body doc (offset-aware)
                abs_s, abs_e = best["start"], best["end"]
                # body_doc is relative to body_abs0; we need char positions within body_doc
                s_local = abs_s - body_abs0
                e_local = abs_e - body_abs0
                # Guard bounds
                s_local = max(0, min(len(body_doc.text), s_local))
                e_local = max(0, min(len(body_doc.text), e_local))
                # Expand within the body_doc, then remap back to absolute
                exp_s_local, exp_e_local = expand_to_sentence(body_doc, s_local, e_local)
                body_span = {"start": body_abs0 + exp_s_local, "end": body_abs0 + exp_e_local}
            else:
                body_span = {"start": win_start, "end": win_start}  # empty/default

            results.append({
                "section_path": s["path"],
                "section_span": s["span"],
                "item_text": it["text"],
                "item_span_sumario": it["span"],
                "org_context": s.get("org_context", {}),
                "body_span": body_span,
                "confidence": float(best.get("confidence", 0.0)),
                "method": best.get("method", "none"),
                "diagnostics": {
                    "window": {"start": win_start, "end": win_end},
                    "num_candidates": len(cands),
                    "first_anchor": cands[0]["anchor_used"] if cands else None,
                }
            })

    found = sum(1 for r in results if r["method"] != "none")
    not_found = len(results) - found
    avg_conf = round(sum(r["confidence"] for r in results) / len(results), 4) if results else 0.0

    return {
        "summary": {"found": found, "not_found": not_found, "avg_confidence": avg_conf},
        "results": results,
    }

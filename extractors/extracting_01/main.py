# main.py
from pathlib import Path
import json
import spacy

from .entities import normalize_text, detect_entities, print_output
from .relations import build_relations
from . import segmenter

from .body_refind import build_body_via_sumario_spacy


def run_pipeline(raw_text: str, show_debug: bool = False):
    """
    Two-pass pipeline WITHOUT assembler.py:
      1) Sumário + roster + body_text (segmenter)
      2) Re-anchor roster strings in BODY-ONLY doc (body_refind) -> BodyItem slices
    Returns:
      - body_doc: spaCy Doc built from body_text
      - sumario: Sumário dataclass (text, ents, relations)
      - roster: dict with cut_index and orgs
      - body_items: list[BodyItem]
      - full_text: normalized full document text
      - body_text: text after cut_index
    """
    # 0) Normalize + tokenizer-only pipeline
    full_text = normalize_text(raw_text)
    nlp = spacy.load("pt_core_news_lg", disable=["ner", "tagger", "parser", "lemmatizer"])
    full_doc = nlp.make_doc(full_text)

    # 1) Entities (rule-based) + relations (rule-based)
    full_doc.ents = detect_entities(full_doc)
    build_relations(full_doc)

    # 2) Sumário + roster + body text
    sumario, roster, body_text, _file_relations = segmenter.build_sumario_and_body(
        full_doc, include_local_details=False
    )

    # 3) Body-only doc + re-anchoring into slices
    body_doc = nlp.make_doc(body_text)
    body_items = build_body_via_sumario_spacy(body_doc, roster, include_local_details=False)

    if show_debug:
        print_output(full_doc)
    #Build the bundle here and return it
    bundle = _preview_bundle(sumario, roster, body_items, full_text, body_text)

    print("body_items: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>", body_items)
    print("roster: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>", roster)
    print("body_doc:>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", body_doc)
    print("body_text:>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", body_text)

    return bundle


def _preview_bundle(sumario, roster, body_items, full_text: str, body_text: str):
    """
    Build and print a JSON bundle (no assembler.py).
    Includes:
      - sumario (text/ents/relations + relations' head/tail text)
      - roster
      - sections (grouped body slices per ORG)
      - body_relations (with absolute offsets and head/tail text)
      - raw text blocks (full_raw, sumario_raw, body_raw)
    """
    cut = int(roster.get("cut_index", 0))

    # Sumário with explicit head/tail text for each relation
    sumario_relations_text = []
    s_text = sumario.text or ""
    for r in (sumario.relations or []):
        hs, he = r["head_offsets"]["start"], r["head_offsets"]["end"]
        ts, te = r["tail_offsets"]["start"], r["tail_offsets"]["end"]
        sumario_relations_text.append({
            "relation": r.get("relation"),
            "head_offsets": r["head_offsets"],
            "tail_offsets": r["tail_offsets"],
            "head_text": s_text[hs:he],
            "tail_text": s_text[ts:te],
        })

    # Group BodyItem slices by section (ORG start)
    by_section = {}
    for it in body_items:
        by_section.setdefault(it.section_id, []).append(it)

    # Build sections (simple: org info + slices)
    sections = []
    body_relations = []
    for section_id in sorted(by_section.keys()):
        items = sorted(by_section[section_id], key=lambda x: x.order_index)
        if not items:
            continue

        org_text = " ".join((items[0].org_text or "").split())
        org_start = min(it.org_start for it in items)
        org_end   = max(it.org_end   for it in items)

        # slices
        slices = []
        for it in items:
            driver = (
                "DOC" if it.relation == "SECTION_ITEM"
                else "SUBORG" if it.relation == "CONTAINS"
                else "UNANCHORED"
            )
            slices.append({
                "driver": driver,
                "title": it.doc_title,
                "start_body": it.slice_start,
                "end_body": it.slice_end,
                "start_abs": cut + it.slice_start,
                "end_abs": cut + it.slice_end,
                "text": it.slice_text,
                "order_index": it.order_index,
            })

            # derive body relations with head/tail text (absolute offsets)
            if it.relation in ("SECTION_ITEM", "CONTAINS"):
                rel = {
                    "relation": it.relation,
                    "head": {"label": "ORG", "text": org_text},
                    "tail": {
                        "label": "DOC" if it.relation == "SECTION_ITEM" else "ORG_SECUNDARIA",
                        "text": it.doc_title,
                    },
                    "head_offsets": {"start": cut + it.org_start, "end": cut + it.org_end},
                    "tail_offsets": {"start": cut + it.doc_start, "end": cut + it.doc_end},
                    "section_id": cut + it.section_id,
                    "source": "body_slices",
                }
                hs, he = rel["head_offsets"]["start"], rel["head_offsets"]["end"]
                ts, te = rel["tail_offsets"]["start"], rel["tail_offsets"]["end"]
                rel["head_text"] = full_text[hs:he]
                rel["tail_text"] = full_text[ts:te]
                body_relations.append(rel)

        sections.append({
            "org": {
                "text": org_text,
                "start_body": org_start,
                "end_body": org_end,
                "start_abs": cut + org_start,
                "end_abs": cut + org_end,
            },
            "slices": slices,
        })

    bundle = {
        "cut_index": cut,
        "sumario": {
            "text": sumario.text,
            "ents": sumario.ents or {},
            "relations": sumario.relations or [],
            "relations_text": sumario_relations_text,
        },
        "roster": roster,
        "sections": sections,
        "body_relations": body_relations,
        "full_raw": full_text,
        "sumario_raw": sumario.text,
        "body_raw": body_text,
    }

    print(json.dumps(bundle, ensure_ascii=False, indent=2))
    return bundle


if __name__ == "__main__":
    # Read input from a file in the repo (edit path as needed)
    input_path = Path("file_01.txt")
    raw = input_path.read_text(encoding="utf-8")

    body_doc, bundle = run_pipeline(raw_text=raw, show_debug=False)
    print(json.dumps(bundle, ensure_ascii=False, indent=2))

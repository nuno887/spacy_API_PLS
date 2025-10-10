from typing import Tuple

from .parser import parse
from .org_detection import find_org_spans
from .split_sumario import split_sumario_body
from .linking import (
    collect_org_hits_from_spans,
    link_orgs,
    find_sumario_anchor,
    choose_body_start_by_second_org,
)

def _build_sumario_struct_from_tree(sections_tree, offset: int, sumario_len: int):
    """
    sections_tree: output of parse(sumario_text, nlp)
    offset: start char of sumário within full text_raw
    sumario_len: len(sumario_text)
    """
    # 1) sections (adjust spans to full-text coordinates)
    sections = []
    for leaf in sections_tree:
        adj_heading_span = {
            "start": leaf["span"]["start"] + offset,
            "end":   leaf["span"]["end"]   + offset,
        }
        items = []
        for it in leaf["items"]:
            items.append({
                "text": it["text"],
                "span": {
                    "start": it["span"]["start"] + offset,
                    "end":   it["span"]["end"]   + offset,
                },
            })
        sections.append({
            "path": leaf["path"],
            "surface_path": leaf["surface"],
            "span": adj_heading_span,
            "items": items,
        })

    # 1b) Dedupe sections by (path, span) and merge items (stable order)
    deduped = []
    seen = {}
    for s in sections:
        key = (tuple(s["path"]), s["span"]["start"], s["span"]["end"])
        if key in seen:
            seen[key]["items"].extend(s["items"])
        else:
            seen[key] = {**s, "items": list(s["items"])}
            deduped.append(seen[key])
    sections = deduped

    # 2) relations_section_item (Section → Item)
    relations_section_item = []
    for s in sections:
        for it in s["items"]:
            relations_section_item.append({
                "section_key": s["path"][-1],
                "section_path": s["path"],
                "surface_path": s["surface_path"],
                "section_span": s["span"],
                "item_span": it["span"],
                "item_text": it["text"],
            })

    # 3) section_ranges (heading → content range inside sumário)
    sections_sorted = sorted(sections, key=lambda x: x["span"]["start"])
    section_ranges = []
    for i, s in enumerate(sections_sorted):
        heading_end = s["span"]["end"]
        next_start = sections_sorted[i + 1]["span"]["start"] if i + 1 < len(sections_sorted) else (offset + sumario_len)
        section_ranges.append({
            "section_key": s["path"][-1],
            "section_path": s["path"],
            "surface_path": s["surface_path"],
            "heading_span": s["span"],
            "content_range": {"start": heading_end, "end": next_start}
        })

    # Final hard dedupe at payload level (even if upstream emitted a duplicate)
    uniq = {}
    for s in sections:
        key = (tuple(s["path"]), s["span"]["start"], s["span"]["end"])
        if key in uniq:
            uniq[key]["items"].extend(s.get("items", []))
        else:
            uniq[key] = s
    sections = list(uniq.values())

    return sections, relations_section_item, section_ranges


def parse_sumario_and_body_bundle(text_raw: str, nlp):
    doc_full = nlp(text_raw)
    org_spans_full = find_org_spans(doc_full, text_raw)

    sum_span, body_span = split_sumario_body(text_raw, org_spans_full)
    sum_start, sum_end = sum_span
    body_start, body_end = body_span

    sumario_text = text_raw[sum_start:sum_end]
    body_text    = text_raw[body_start:body_end]

    _, sections_tree = parse(sumario_text, nlp)
    sections, rel_section_item, section_ranges = _build_sumario_struct_from_tree(
        sections_tree, offset=sum_start, sumario_len=len(sumario_text)
    )

    # ORG hits per slice + ORG↔ORG linking
    sum_orgs  = collect_org_hits_from_spans(org_spans_full, text_raw, sum_span, source="sumario")
    body_orgs = collect_org_hits_from_spans(org_spans_full, text_raw, body_span, source="body")
    relations, diag = link_orgs(sum_orgs, body_orgs)

    # Attach nearest preceding Sumário ORG (and linked Body ORG, if any) to each section
    sum_orgs_sorted = sorted(sum_orgs, key=lambda h: h["span"]["start"])
    org_bands = []
    for i, org in enumerate(sum_orgs_sorted):
        band_start = org["span"]["start"]
        band_end = sum_orgs_sorted[i + 1]["span"]["start"] if i + 1 < len(sum_orgs_sorted) else sum_end
        org_bands.append((band_start, band_end, org))

    # Map body org by canonical key via relations
    body_by_key = {r["key"]: r["body"] for r in relations}

    # Enrich sections with org_context
    for s in sections:
        hstart = s["span"]["start"]
        attached = None
        for a, b, org in org_bands:
            if a <= hstart < b:
                attached = org
                break
        if attached:
            s["org_context"] = {
                "sumario_org": {
                    "surface_raw": attached["surface_raw"],
                    "span": attached["span"],
                    "canonical_key": attached["canonical_key"],
                }
            }
            bod = body_by_key.get(attached["canonical_key"])
            if bod:
                s["org_context"]["body_org"] = {
                    "surface_raw": bod["surface_raw"],
                    "span": bod["span"],
                }

    second_org_pos = choose_body_start_by_second_org(org_spans_full, text_raw, find_sumario_anchor(text_raw))
    strategy = "second_org_pair" if (second_org_pos == body_start) else "fallback_first_l1_or_window"

    payload = {
        "version": "sumario_body_linker@1.0.0",
        "text_raw": text_raw,
        "sumario": {
            "span": {"start": sum_start, "end": sum_end},
            "text_raw": sumario_text,
            "sections": sections,
            "relations_section_item": rel_section_item,
            "section_ranges": section_ranges,
        },
        "body": {
            "span": {"start": body_start, "end": body_end},
            "text_raw": body_text,
        },
        "relations_org_to_org": relations,
        "diagnostics": {
            "strategy": strategy,
            "split_anchor": {
                "key": relations[0]["key"],
                "body_start": body_start
            } if strategy == "second_org_pair" and relations else None,
            "unmatched_sumario_orgs": diag.get("unmatched_sumario_orgs", []),
            "unmatched_body_orgs": diag.get("unmatched_body_orgs", []),
        },
    }
    return payload, sumario_text, body_text, text_raw

import spacy
from entities import parse, parse_sumario_and_body_bundle
from entities.debug_print import print_results, print_payload_summary
from body_extraction import run_extraction
from body_extraction.debug_print import print_report  



# Para apagar depois -------------------------------------------------------------------------------------------------

import json

def pretty_sections(sections):
    print("\n=== SECTIONS (as passed to run_extraction) ===")
    for i, s in enumerate(sections, 1):
        path = " > ".join(s.get("path", []))
        span = s.get("span", {})
        org_span = s.get("org_span")
        items = s.get("items", [])
        org_ctx = s.get("org_context", {})
        org_name = None
        if org_ctx and org_ctx.get("sumario_org"):
            org_name = org_ctx["sumario_org"]["surface_raw"].splitlines()[0].strip()

        print(f"[{i:02d}] PATH: {path}")
        if org_name:
            print(f"     ORG: {org_name}")
        if org_span:
            print(f"  org_span: {org_span['start']}..{org_span['end']}")
        print(f"     span: {span.get('start','?')}..{span.get('end','?')}")
        print(f"     items: {len(items)}")
        # show first line of each item, truncated
        for j, it in enumerate(items, 1):
            txt = (it.get("text") or "").strip().splitlines()[0]
            if len(txt) > 120:
                txt = txt[:117] + "..."
            istart = it.get("span", {}).get("start", "?")
            iend = it.get("span", {}).get("end", "?")
            print(f"        {j:02d}. {txt}  @{istart}..{iend}")
        print()

def pretty_rels(rels):
    print("\n=== ORG -> ORG RELATIONS (as passed to run_extraction) ===")
    if not rels:
        print("(none)")
        return
    for r in rels:
        key = r.get("key")
        s_s = r.get("sumario", {}).get("span", {})
        s_txt = r.get("sumario", {}).get("surface_raw", "").splitlines()[0].strip()
        b_s = r.get("body", {}).get("span", {})
        b_txt = r.get("body", {}).get("surface_raw", "").splitlines()[0].strip()
        conf = r.get("confidence")
        print(f"- key: {key}")
        print(f"  sumário: '{s_txt}' @{s_s.get('start','?')}..{s_s.get('end','?')}")
        print(f"  body   : '{b_txt}' @{b_s.get('start','?')}..{b_s.get('end','?')}  (body-relative expected)")
        print(f"  conf   : {conf}")



# --------------------------------------------------------------------------------------------------------------------

def build_nlp():

    nlp = spacy.load("pt_core_news_lg", disable=["ner", "tagger", "parser", "lemmatizer"])
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer", first=True, config={"punct_chars": [".", "!", "?", ";", ":"]})
    return nlp


if __name__ == "__main__":
    with open("sample_input_02.txt", "r", encoding="utf-8") as f:
        text = f.read()

    nlp = build_nlp()

    # -------- Split first (Sumário / Body) --------
    payload, sumario_text, body_text, _ = parse_sumario_and_body_bundle(text, nlp)

    # -------- Debug: sections only on Sumário slice (print absolute offsets via 'offset') --------
    sum_start = payload["sumario"]["span"]["start"]
    sum_doc, sum_sections = parse(sumario_text, nlp)
    print_results(sum_doc, sum_sections, offset=sum_start)

    # -------- Bundle summary (Sumário + Body + relations) --------
    print_payload_summary(payload, save_path="sumario_body_payload.json")

    # -------- Body extraction / alignment --------
    sections = payload["sumario"]["sections"]
    rels = payload.get("relations_org_to_org", [])


    pretty_sections(sections)
    pretty_rels(rels)

    # Important: create body_doc via nlp(...) so sentencizer runs
    report = run_extraction(body_text, sections, rels, nlp)

    print_report(report, body_text, show_full=True)



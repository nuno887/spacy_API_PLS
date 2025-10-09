import json



def print_results(doc, sections_tree):
    print("\n=== ORG HEADERS ===")
    for ent in doc.ents:
        if ent.label_ == "ORG":
            print(f"[ORG] '{ent.text}' @{ent.start_char}:{ent.end_char}")

    print("\n=== SECTIONS (leaf nodes) ===")
    for sect in sections_tree:
        path = " > ".join(sect["path"])
        surface = " > ".join(sect["surface"])
        span = sect["span"]
        print(f"- PATH: {path}")
        print(f"  SURF: {surface}")
        print(f"  SPAN: {span['start']}..{span['end']}")
        if sect["items"]:
            print(f"  ITEMS ({len(sect['items'])}):")
            for i, it in enumerate(sect["items"], 1):
                print(f"    {i:02d}. {it['text']}  @{it['span']['start']}..{it['span']['end']}")
        else:
            print("  ITEMS (0)")
        print()
"""
    print("\n=== ALL ENTITY SPANS (debug) ===")
    for ent in doc.ents:
        print(f"{ent.label_:<20} @{ent.start_char:>5}-{ent.end_char:<5} | {repr(ent.text)}")
"""

def print_payload_summary(payload: dict, save_path: str | None = None) -> None:
    sum_span = payload["sumario"]["span"]
    body_span = payload["body"]["span"]

  
    print("\n=== SPLIT ===")
    print(f"Sumário: {sum_span['start']}..{sum_span['end']} | len={sum_span['end']-sum_span['start']}")
    print(f"Body   : {body_span['start']}..{body_span['end']} | len={body_span['end']-body_span['start']}")
    print(f"Strategy: {payload['diagnostics']['strategy']}")

    # Sections & items
    print("\n=== SUMÁRIO SECTIONS ===")
    seen_sec = set()
    secs = payload["sumario"]["sections"]
    for idx, s in enumerate(secs):
        sec_key = (tuple(s["path"]), s["span"]["start"], s["span"]["end"])
        if sec_key in seen_sec:
            continue
        seen_sec.add(sec_key)
        path = " > ".join(s["path"])
        print(f"- {path}  @ {s['span']['start']}..{s['span']['end']}  [#{idx}]")
        for it in s["items"]:
            print(f"   • {it['text']}  @ {it['span']['start']}..{it['span']['end']}")

    # Section → Item relations
    print("\n=== SUMÁRIO RELATIONS (Section -> Item) ===")
    for r in payload["sumario"]["relations_section_item"]:
        print(f"{' > '.join(r['section_path'])}  ::  {r['item_text']}")

    # Section ranges
    print("\n=== SUMÁRIO SECTION RANGES ===")
    for sr in payload["sumario"]["section_ranges"]:
        print(f"{' > '.join(sr['section_path'])}  ::  content {sr['content_range']['start']}..{sr['content_range']['end']}")

    # ORG → ORG relations
    print("\n=== ORG -> ORG RELATIONS ===")
    rels = payload.get("relations_org_to_org", [])
    if not rels:
        print("(none)")
    else:
        for r in rels:
            print(f"- {r['key']}")
            print(f"  sumário: '{r['sumario']['surface_raw']}' @{r['sumario']['span']['start']}..{r['sumario']['span']['end']}")
            print(f"  body   : '{r['body']['surface_raw']}' @{r['body']['span']['start']}..{r['body']['span']['end']}")
            print(f"  conf   : {r['confidence']}")

    # Diagnostics
    diag = payload.get("diagnostics", {})
    if diag.get("unmatched_sumario_orgs") or diag.get("unmatched_body_orgs") or diag.get("split_anchor"):
        print("\n=== DIAGNOSTICS ===")
        if diag.get("split_anchor"):
            print(f"Split anchor: {diag['split_anchor']}")
        if diag.get("unmatched_sumario_orgs"):
            print("Unmatched Sumário ORGs:")
            for h in diag["unmatched_sumario_orgs"]:
                print(f"  - '{h['surface_raw']}' @{h['span']['start']}..{h['span']['end']} | key={h['canonical_key']}")
        if diag.get("unmatched_body_orgs"):
            print("Unmatched Body ORGs:")
            for h in diag["unmatched_body_orgs"]:
                print(f"  - '{h['surface_raw']}' @{h['span']['start']}..{h['span']['end']} | key={h['canonical_key']}")

    # Optional save
    if save_path:
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"\nSaved payload to {save_path}")
        except Exception as e:
            print(f"\nCouldn’t save payload to {save_path}: {e}")


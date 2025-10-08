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

    print("\n=== ALL ENTITY SPANS (debug) ===")
    for ent in doc.ents:
        print(f"{ent.label_:<20} @{ent.start_char:>5}-{ent.end_char:<5} | {repr(ent.text)}")

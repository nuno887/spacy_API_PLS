def print_report(report: dict, text_raw: str, preview_len: int = 160) -> None:
    s = report["summary"]
    print("\n=== BODY EXTRACTION SUMMARY ===")
    print(f"Found: {s['found']} | Not found: {s['not_found']} | Avg conf: {s['avg_confidence']}")

    print("\n=== BODY EXTRACTION RESULTS ===")
    for i, r in enumerate(report["results"], 1):
        path = " > ".join(r["section_path"])
        bs = r["body_span"]
        preview = text_raw[bs["start"]:bs["end"]][:preview_len].replace("\n", " ")
        print(f"{i:02d}. [{r['method']} | conf={r['confidence']:.2f}] {path}")
        print(f"    item  : {r['item_text'][:preview_len]}")
        print(f"    body  : @{bs['start']}..{bs['end']}  | {preview!r}")

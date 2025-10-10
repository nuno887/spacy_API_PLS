# body_extraction/debug_print.py
from typing import Dict

def print_report(report: Dict, body_text: str, show_full: bool = False, preview_chars: int = 220) -> None:
    """
    Pretty-print the extraction report.
    - show_full: if True, prints the entire matched section text instead of a truncated preview.
    - preview_chars: how many characters to show if show_full=False.
    """
    found = sum(1 for r in report["results"] if r["method"] != "none")
    total = len(report["results"])
    avg_conf = (
        sum(r["confidence"] for r in report["results"]) / total if total > 0 else 0.0
    )

    print("\n=== BODY EXTRACTION SUMMARY ===")
    print(f"Found: {found} | Not found: {total - found} | Avg conf: {avg_conf:.2f}")

    print("\n=== BODY EXTRACTION RESULTS ===")
    for i, r in enumerate(report["results"], 1):
        sec_name = r.get("section_name", "(unknown)")
        method = r.get("method", "none")
        conf = r.get("confidence", 0.0)
        item_text = r.get("item_text", "").replace("\n", " ")

        s = r["body_span"]["start"]
        e = r["body_span"]["end"]

        # Decide how much body text to show
        if show_full:
            excerpt = body_text[s:e]
        else:
            excerpt = body_text[s:min(e, s + preview_chars)]
        excerpt_clean = excerpt.replace("\r\n", " ").replace("\n", " ")

        print(f"{i:02d}. [{method} | conf={conf:.2f}] {sec_name}")
        print(f"    item  : {item_text[:180]}")
        print(f"    body  : @{s}..{e}  | '{excerpt_clean}'")

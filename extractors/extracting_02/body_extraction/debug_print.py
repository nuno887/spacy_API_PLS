# body_extraction/debug_print.py
from typing import Dict, Any


def _first_nonblank_line(s: str) -> str:
    """Return the first non-empty line of s, with internal whitespace collapsed."""
    for ln in (s or "").splitlines():
        t = ln.strip()
        if t:
            return " ".join(t.split())
    return ""


def _clean_one_line(s: str) -> str:
    """Normalize CR/LF to spaces for compact previews."""
    return (s or "").replace("\r\n", " ").replace("\n", " ").strip()


def print_report(report: Dict[str, Any],
                 body_text: str,
                 show_full: bool = False,
                 preview_chars: int = 220) -> None:
    """
    Pretty-print the extraction report.

    - show_full: if True, prints the entire matched body text instead of a truncated preview.
    - preview_chars: how many characters of body to show if show_full=False.
    """
    results = report.get("results", [])
    total = len(results)
    found = sum(1 for r in results if r.get("method") not in (None, "none"))
    avg_conf = (sum(float(r.get("confidence", 0.0)) for r in results) / total) if total > 0 else 0.0

    print("\n=== BODY EXTRACTION SUMMARY ===")
    print(f"Found: {found} | Not found: {total - found} | Avg conf: {avg_conf:.2f}")

    print("\n=== BODY EXTRACTION RESULTS ===")
    for i, r in enumerate(results, 1):
        sec_name = r.get("section_name", "(unknown)")
        method = r.get("method", "none")
        conf = float(r.get("confidence", 0.0))

        # Full, untruncated first item line
        item_text_full_line = _first_nonblank_line(r.get("item_text", ""))
        print(f"{i:02d}. [{method} | conf={conf:.2f}] {sec_name}")
        print(f"    item  : {item_text_full_line}")

        # Body excerpt
        span = r.get("body_span") or {}
        s = int(span.get("start", 0))
        e = int(span.get("end", 0))

        if show_full:
            excerpt = body_text[s:e]
        else:
            excerpt = body_text[s:min(e, s + max(0, int(preview_chars)))]

        print(f"    body  : @{s}..{e}  | '{_clean_one_line(excerpt)}'")


def render_report(report: Dict[str, Any],
                  body_text: str,
                  show_full: bool = False,
                  preview_chars: int = 220) -> str:
    """Return the report as a single string (same content as print_report)."""
    results = report.get("results", [])
    total = len(results)
    found = sum(1 for r in results if r.get("method") not in (None, "none"))
    avg_conf = (sum(float(r.get("confidence", 0.0)) for r in results) / total) if total > 0 else 0.0

    lines = []
    lines.append("\n=== BODY EXTRACTION SUMMARY ===")
    lines.append(f"Found: {found} | Not found: {total - found} | Avg conf: {avg_conf:.2f}")

    lines.append("\n=== BODY EXTRACTION RESULTS ===")
    for i, r in enumerate(results, 1):
        sec_name = r.get("section_name", "(unknown)")
        method = r.get("method", "none")
        conf = float(r.get("confidence", 0.0))
        item_text_full_line = _first_nonblank_line(r.get("item_text", ""))

        span = r.get("body_span") or {}
        s = int(span.get("start", 0))
        e = int(span.get("end", 0))

        if show_full:
            excerpt = body_text[s:e]
        else:
            excerpt = body_text[s:min(e, s + max(0, int(preview_chars)))]

        excerpt_1line = (excerpt or "").replace("\r\n", " ").replace("\n", " ").strip()

        lines.append(f"{i:02d}. [{method} | conf={conf:.2f}] {sec_name}")
        lines.append(f"    item  : {item_text_full_line}")
        lines.append(f"    body  : @{s}..{e}  | '{excerpt_1line}'")

    return "\n".join(lines)


def save_report_txt(path: str,
                    report: Dict[str, Any],
                    body_text: str,
                    show_full: bool = False,
                    preview_chars: int = 220) -> None:
    """Write the rendered report to a UTF-8 text file."""
    txt = render_report(report, body_text, show_full=show_full, preview_chars=preview_chars)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)

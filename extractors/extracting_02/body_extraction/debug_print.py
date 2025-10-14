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


# --- DEBUG: dump a section window line-by-line and show potential title matches ---
TITLE_PREFIXES = {
    "PortariasExtensao": ["Aviso de Projeto"],
    "Convencoes": ["Contrato Coletivo", "Acordo de adesão"],
}

def _collapse_ws_dbg(s: str) -> str:
    # keep \n for line positions, but remove common noise
    s = s.replace("\u00AD", "")          # soft hyphen
    s = s.replace("-\n", "")             # hyphen + newline wrap
    s = s.replace("\u00A0", " ")         # NBSP
    return s

def dbg_dump_window_lines(sec_key: str, body_text: str, win_start: int, win_end: int, max_lines: int = 50):
    print(f"\n[WIN-DBG] {sec_key} window @{win_start}..{win_end} (len={win_end-win_start})")
    raw = body_text[win_start:win_end]
    norm = _collapse_ws_dbg(raw)

    # show first few non-blank raw lines (context)
    print("[WIN-DBG] first raw lines:")
    shown = 0
    for ln in raw.splitlines():
        if ln.strip():
            shown += 1
            preview = ln.strip()
            if len(preview) > 120: preview = preview[:117] + "..."
            print(f"  · {preview!r}")
            if shown >= 5: break

    # now enumerate normalized lines with absolute offsets and prefix hit markers
    print("[WIN-DBG] scan normalized lines for prefixes:")
    prefixes = TITLE_PREFIXES.get(sec_key, [])
    offset = 0
    line_no = 1
    for line in norm.splitlines(keepends=True):
        if line_no > max_lines:
            print(f"  ... (truncated after {max_lines} lines)")
            break
        abs_line_start = win_start + offset
        stripped = line.lstrip()
        near_bol = (len(line) - len(stripped)) <= 2
        prefix_hit = any(stripped.startswith(p) for p in prefixes) if near_bol else False
        mark = "<<< TITLE?" if prefix_hit else ""
        show = stripped.strip().replace("\t", " ")
        if len(show) > 120: show = show[:117] + "..."
        print(f"  {line_no:03d} @{abs_line_start:>5} bol_ok={near_bol!s:5} {mark:10} | {show!r}")
        offset += len(line)
        line_no += 1


def dbg_print_spacy_title_hits(nlp, sec_key: str, win_text: str, win_start: int, body_text: str, index_block_titles_fn):
    rel_hits = index_block_titles_fn(nlp, win_text, sec_key)
    abs_hits = [win_start + r for r in sorted(rel_hits)]
    print(f"[TITLE-DBG] spaCy title hits for {sec_key}: {len(abs_hits)}")
    for p in abs_hits:
        L = body_text.rfind("\n", 0, p) + 1
        R = body_text.find("\n", p)
        if R == -1: R = len(body_text)
        line = body_text[L:R].strip()
        if len(line) > 160: line = line[:157] + "..."
        print(f"  - @{p:>5} | line='{line}'")
    return abs_hits

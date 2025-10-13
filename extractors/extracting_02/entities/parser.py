from typing import List, Dict, Tuple
from collections import defaultdict
import re

from spacy.tokens import Span
from spacy.util import filter_spans

from .taxonomy import build_heading_matcher
from .headings import scan_headings
from .org_detection import find_org_spans, _is_all_caps_line, _starts_with_starter
from .items import find_item_char_spans, clean_item_text, _find_inline_item_boundaries, build_title_matcher, is_numeric_only_item
from .taxonomy import TAXONOMY
from .taxonomy import Node
from .normalization import normalize_heading_text, strip_diacritics


TITLE_LINE_RE = re.compile(
    r"""(?xi)                       # verbose, case-insensitive
    \b
    (?:acordo|contrato)             # Acordo|Contrato
    \s+coletiv\w+\s+de\s+trabalho   # Coletivo[a]? de Trabalho (OCR safe)
    .*?
    (?:n[\.\º°]?\s*o?)?             # optional n.º / nº / n. o
    \s*\d+\s*/\s*\d{2,4}            # 1/2014, 12/14, etc.
    \s*:?
    \s*\Z
    """
)


def _dedup_spans(spans: List[Span]) -> List[Span]:
    seen, out = set(), []
    for s in spans:
        key = (s.start, s.end, s.label_)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


# --- helpers: headings/leaves -------------------------------------------------

def _scan_heading_leaves(text: str, nlp, alias_to_nodes):
    """Return (leaves, heading_hits), where leaves carry path/surface/span/text_range."""
    hits = scan_headings(text, alias_to_nodes)
    hits.sort(key=lambda h: h.start_char)

    # stack entries: (canonical, level, start, end, surface)
    stack, leaves, leaf_seen = [], [], set()

    def close_leaf_if_any(next_start: int):
        if not stack:
            return
        leaf = stack[-1]
        leaf_path = [s[0] for s in stack]
        leaf_surfs = [s[4] for s in stack]
        leaf_start, leaf_end = leaf[2], leaf[3]
        text_start, text_end = leaf_end, next_start
        key = (leaf_start, leaf_end, tuple(leaf_path))
        if text_start < text_end and key not in leaf_seen:
            leaf_seen.add(key)
            leaves.append({
                "path": leaf_path[:],
                "surface": leaf_surfs[:],
                "span": {"start": leaf_start, "end": leaf_end},
                "text_range": (text_start, text_end),
            })

    def allowed_by_parents(node, current_parent: str | None) -> bool:
        return True if node.parents is None else current_parent in node.parents

    i = 0
    while i < len(hits):
        hit = hits[i]
        key_norm = normalize_heading_text(hit.surface)
        candidates = alias_to_nodes.get(key_norm, [])
        chosen = None
        current_parent = stack[-1][0] if stack else None
        for node in sorted(candidates, key=lambda n: (-n.level, -len(n.canonical))):
            if allowed_by_parents(node, current_parent):
                chosen = node
                break
        if chosen is None:
            chosen = candidates[0] if candidates else None
            if chosen is None:
                i += 1
                continue

        if stack and stack[-1][0] == chosen.canonical and stack[-1][2] == hit.start_char:
            i += 1
            continue

        close_leaf_if_any(hit.start_char)
        while stack and stack[-1][1] >= chosen.level:
            stack.pop()
        stack.append((chosen.canonical, chosen.level, hit.start_char, hit.end_char, hit.surface))
        i += 1

    close_leaf_if_any(len(text))
    return leaves, hits


# --- helpers: entities from leaves/items -------------------------------------

def _collect_heading_leaf_spans(doc, leaves) -> list[Span]:
    out = []
    for leaf in leaves:
        ch = doc.char_span(leaf["span"]["start"], leaf["span"]["end"], alignment_mode="expand")
        if ch is None:
            continue
        label = leaf["path"][-1]
        out.append(Span(doc, ch.start, ch.end, label=label))
    return out


def _segment_items_in_leaves(text: str, leaves, nlp) -> list[Span]:
    """Return raw item spans (as spaCy Spans) labeled Item{LeafName}."""
    doc = nlp.make_doc(text)  # cheap tokenization for alignment_mode
    heading_starts = {l["span"]["start"] for l in leaves}
    item_spans: list[Span] = []
    for leaf in leaves:
        sc, ec = leaf["text_range"]
        for s_char, e_char in find_item_char_spans(text, sc, ec, heading_starts):
            ch = doc.char_span(s_char, e_char, alignment_mode="expand")
            if ch is None:
                continue
            item_spans.append(Span(doc, ch.start, ch.end, label=f"Item{leaf['path'][-1]}"))
    return item_spans


def _refine_item_spans(text: str, raw_item_spans: list[Span], nlp) -> list[Span]:
    """Apply inline splits (e.g., dot-leaders, multiple titles on one line)."""
    doc = nlp.make_doc(text)
    refined: list[Span] = []
    for it_sp in raw_item_spans:
        raw = text[it_sp.start_char:it_sp.end_char]
        cuts = _find_inline_item_boundaries(raw, it_sp.start_char, nlp)
        if not cuts:
            refined.append(it_sp); continue
        prev = it_sp.start_char
        for cut in cuts:
            ch = doc.char_span(prev, cut, alignment_mode="expand")
            if ch is not None:
                refined.append(Span(doc, ch.start, ch.end, label=it_sp.label_))
            prev = cut + 1
        ch = doc.char_span(prev, it_sp.end_char, alignment_mode="expand")
        if ch is not None:
            refined.append(Span(doc, ch.start, ch.end, label=it_sp.label_))
    return refined


def _assemble_sections_tree_from_leaves(text: str, doc, leaves, ents: list[Span]) -> list[dict]:
    """Build sections_tree (taxonomy leaves only), cleaning and dropping numeric-only items."""
    sections_tree: list[dict] = []
    items_per_leaf: dict[int, list[dict]] = defaultdict(list)
    seen_item_spans_per_leaf: dict[int, set] = defaultdict(set)

    for leaf in leaves:
        sc, ec = leaf["text_range"]
        lid = id(leaf)
        for sp in ents:
            if sp.label_.startswith("Item") and sc <= sp.start_char and sp.end_char <= ec:
                key = (sp.start_char, sp.end_char)
                if key in seen_item_spans_per_leaf[lid]:
                    continue
                seen_item_spans_per_leaf[lid].add(key)
                txt_clean = clean_item_text(sp.text)
                if is_numeric_only_item(txt_clean):
                    continue
                items_per_leaf[lid].append({
                    "text": txt_clean,
                    "span": {"start": sp.start_char, "end": sp.end_char}
                })

    tmp = []
    for leaf in leaves:
        tmp.append({
            "path": leaf["path"],
            "surface": leaf["surface"],
            "span": leaf["span"],
            "items": items_per_leaf.get(id(leaf), []),
        })

    deduped, seen = [], {}
    for s in tmp:
        key = (tuple(s["path"]), s["span"]["start"], s["span"]["end"])
        if key in seen:
            seen[key]["items"].extend(s["items"])
        else:
            seen[key] = {**s, "items": list(s["items"])}
            deduped.append(seen[key])
    return deduped


def _detect_series_sections_under_orgs(
    text: str,
    nlp,
    doc,
    leaves,
    org_spans: list[Span],
    alias_to_nodes: dict,
) -> list[dict]:
    """
    Detect sections that are a series of instruments directly under each ORG block:
      ORG HEADER
      Title 1:
      Title 2:
      ...
    Returns a list of section dicts shaped like taxonomy leaves, but with path/surface = [ORG label].
    """
    # Boundaries that terminate an instrument-series region
    heading_starts = {l["span"]["start"] for l in leaves}
    boundary_starts = sorted(set(heading_starts) | {sp.start_char for sp in org_spans} | {len(text)})

    def _next_boundary_after(pos: int) -> int:
        for b in boundary_starts:
            if b > pos:
                return b
        return len(text)

    def _org_display_name(raw: str) -> str:
        # First non-blank line, trim trailing punctuation/dashes/colon
        for ln in raw.splitlines():
            t = ln.strip()
            if t:
                return t.rstrip(" :—–-").strip()
        return raw.strip()

    # Reject obvious heading blocks (not items) by inspecting the first non-blank line
    def _is_heading_like_line(line: str) -> bool:
        # (a) ORG header style (starter + ALL CAPS)
        if _starts_with_starter(line) and _is_all_caps_line(line):
            return True
        # (b) Known taxonomy heading alias
        norm = normalize_heading_text(line.rstrip(":"))
        if norm in alias_to_nodes:
            return True
        # (c) Colon but no number/year token on that line
        if ":" in line and not re.search(r'n[\.\º°]?\s*o?\s*\d+\s*/\s*\d{2,4}', line, re.IGNORECASE):
            return True
        # (d) Very specific denylist (per your examples)
        dl = {"direcao regional do trabalho", "direção regional do trabalho",
              "regulamentacao do trabalho", "regulamentação do trabalho"}
        base = strip_diacritics(line).strip().lower().rstrip(":")
        if base in dl:
            return True
        return False

    def _first_nonblank_line(full: str, s: int, e: int) -> str:
        frag = full[s:e]
        for ln in frag.splitlines():
            if ln.strip():
                return ln.strip()
        return ""

    # Title detection: spaCy matcher + regex fallback
    title_matcher = build_title_matcher(nlp)

    series_sections: list[dict] = []
    # ---- Debug (optional)
    print(f"[SERIES DBG] boundary_starts={sorted(boundary_starts)[:10]}... len={len(boundary_starts)}")

    for osp in org_spans:
        # Region AFTER the org header block, up to next boundary (next org/heading/EOF)
        region_start = osp.end_char
        region_end = _next_boundary_after(osp.end_char)
        print(f"[SERIES DBG] visit ORG {osp.start_char}-{osp.end_char} -> region {region_start}-{region_end} (len={region_end-region_start})")
        if region_start >= region_end:
            print(f"[SERIES DBG] skip ORG {osp.start_char}-{osp.end_char}: empty region")
            continue

        region_text = text[region_start:region_end]
        proto_lines = [ln.strip() for ln in region_text.splitlines() if ln.strip()]
        print(f"[SERIES DBG] region first lines: {proto_lines[:2]}")

        # Pre-check: require at least one explicit title line in the region
        has_title = False
        dbg_matches = []
        for idx, ln in enumerate(region_text.splitlines()):
            s_line = ln.strip()
            if not s_line:
                continue
            d = nlp.make_doc(s_line)
            hit_matcher = any(i == 0 for i, j, k in title_matcher(d))
            hit_regex = bool(TITLE_LINE_RE.search(s_line))
            if hit_matcher or hit_regex:
                has_title = True
                dbg_matches.append((idx, s_line))

        print(f"[SERIES CHECK] ORG {osp.start_char}-{osp.end_char}: titles_found={len(dbg_matches)}")
        for i, s_line in dbg_matches[:6]:
            print(f"   [title@line {i}] {s_line}")

        if not has_title:
            # No titles → treat this ORG as classic taxonomy only (no series)
            continue

        # Segment candidates in this org-scoped gap
        candidates: list[tuple[int, int]] = []
        for s_char, e_char in find_item_char_spans(text, region_start, region_end, set(boundary_starts)):
            if s_char < e_char:
                candidates.append((s_char, e_char))
        print(f"[SERIES BUILD] ORG {osp.start_char}-{osp.end_char}: candidates_found={len(candidates)}")

        # Filter out heading-like blocks unless first line is a title
        kept_blocks: list[tuple[int, int]] = []
        for s_char, e_char in candidates:
            first_ln = _first_nonblank_line(text, s_char, e_char)
            if not first_ln:
                continue
            d_first = nlp.make_doc(first_ln)
            is_title = any(i == 0 for i, j, k in title_matcher(d_first)) or bool(TITLE_LINE_RE.search(first_ln))
            if (not is_title) and _is_heading_like_line(first_ln):
                continue
            kept_blocks.append((s_char, e_char))

        # Inline-split each kept block on new titles (prevents merging consecutive items)
        final_items: list[dict] = []
        for s_char, e_char in kept_blocks:
            raw = text[s_char:e_char]
            cuts = _find_inline_item_boundaries(raw, s_char, nlp)
            if not cuts:
                txt = clean_item_text(raw)
                if not is_numeric_only_item(txt):
                    final_items.append({"text": txt, "span": {"start": s_char, "end": e_char}})
                continue
            prev = s_char
            for cut in cuts:
                if prev < cut:
                    txt = clean_item_text(text[prev:cut])
                    if not is_numeric_only_item(txt):
                        final_items.append({"text": txt, "span": {"start": prev, "end": cut}})
                prev = cut + 1
            if prev < e_char:
                txt = clean_item_text(text[prev:e_char])
                if not is_numeric_only_item(txt):
                    final_items.append({"text": txt, "span": {"start": prev, "end": e_char}})

        print(f"[SERIES BUILD] ORG {osp.start_char}-{osp.end_char}: items_kept={len(final_items)}")

        if final_items:
            org_header_text = text[osp.start_char:osp.end_char]
            org_label = _org_display_name(org_header_text)
            print(f"[SERIES APPEND] ORG {osp.start_char}-{osp.end_char}: label='{org_label}', items={len(final_items)}")
            series_sections.append({
                "path": [org_label],
                "surface": [org_label],
                "span": {"start": region_start, "end": region_end},
                "items": final_items,
                "org_span": {"start": osp.start_char, "end": osp.end_char},
            })

    return series_sections




def parse(text: str, nlp):
    doc = nlp(text)
    _, alias_to_nodes = build_heading_matcher(nlp)

    # 1) headings → leaves
    leaves, _ = _scan_heading_leaves(text, nlp, alias_to_nodes)

    # 2) ORG spans
    org_spans = find_org_spans(doc, text)
    org_spans.sort(key=lambda sp: sp.start_char)

    # 3) heading leaf spans (entities)
    heading_leaf_spans = _collect_heading_leaf_spans(doc, leaves)

    # 4) items inside leaves → refine-inline
    raw_item_spans = _segment_items_in_leaves(text, leaves, nlp)
    refined_item_spans = _refine_item_spans(text, raw_item_spans, nlp)

    # 5) finalize ents (ORG + headings + items)
    all_spans = _dedup_spans(org_spans + heading_leaf_spans + refined_item_spans)
    all_spans = filter_spans(all_spans)
    doc.ents = tuple(_dedup_spans(all_spans))

    # 6) sections tree (taxonomy leaves)
    sections_tree = _assemble_sections_tree_from_leaves(text, doc, leaves, list(doc.ents))

    # 7) ORG-series sections (add-on)
    series_sections = _detect_series_sections_under_orgs(
        text, nlp, doc, leaves, org_spans, alias_to_nodes
    )
    sections_tree.extend(series_sections)

    return doc, sections_tree

from typing import List, Dict, Tuple
from collections import defaultdict
from spacy.tokens import Span
from spacy.util import filter_spans
from .taxonomy import build_heading_matcher
from .headings import scan_headings
from .org_detection import find_org_spans
from .items import find_item_char_spans, clean_item_text, _find_inline_item_boundaries
from .taxonomy import TAXONOMY
from .taxonomy import Node
from .normalization import normalize_heading_text


def _dedup_spans(spans: List[Span]) -> List[Span]:
    seen, out = set(), []
    for s in spans:
        key = (s.start, s.end, s.label_)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def parse(text: str, nlp):
    doc = nlp(text)
    _, alias_to_nodes = build_heading_matcher(nlp)

    hits = scan_headings(text, alias_to_nodes)
    hits.sort(key=lambda h: h.start_char)

    # stack entries: (canonical, level, start, end, surface)
    stack: List[Tuple[str, int, int, int, str]] = []
    leaves: List[Dict] = []
    leaf_seen = set()

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
        # Using the same normalization that built alias_to_nodes (lower + strip deacritics + trim colon + squeeze spaces)
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

        # guard: avoid pushing the same heading at the same position twice
        if stack and stack[-1][0] == chosen.canonical and stack[-1][2] == hit.start_char:
            i += 1
            continue

        close_leaf_if_any(hit.start_char)
        while stack and stack[-1][1] >= chosen.level:
            stack.pop()
        stack.append((chosen.canonical, chosen.level, hit.start_char, hit.end_char, hit.surface))
        i += 1

    # close the last open leaf to EOF
    close_leaf_if_any(len(text))

    # ORG spans across the whole text
    org_spans = find_org_spans(doc, text)

    # heading leaf spans (for labeling)
    heading_leaf_spans: List[Span] = []
    for leaf in leaves:
        ch = doc.char_span(leaf["span"]["start"], leaf["span"]["end"], alignment_mode="expand")
        if ch is None:
            continue
        label = leaf["path"][-1]
        heading_leaf_spans.append(Span(doc, ch.start, ch.end, label=label))

    # collect items within each leaf's text range
    heading_starts = {l["span"]["start"] for l in leaves}
    item_spans: List[Span] = []
    for leaf in leaves:
        sc, ec = leaf["text_range"]
        for s_char, e_char in find_item_char_spans(text, sc, ec, heading_starts):
            ch = doc.char_span(s_char, e_char, alignment_mode="expand")
            if ch is None:
                continue
            item_spans.append(Span(doc, ch.start, ch.end, label=f"Item{leaf['path'][-1]}"))

    # refine items with inline splits
    refined_item_spans: List[Span] = []
    for it_sp in item_spans:
        raw = text[it_sp.start_char:it_sp.end_char]
        splits = _find_inline_item_boundaries(raw, it_sp.start_char)
        if not splits:
            refined_item_spans.append(it_sp)
            continue
        prev = it_sp.start_char
        for cut in splits:
            ch = doc.char_span(prev, cut, alignment_mode="expand")
            if ch is not None:
                refined_item_spans.append(Span(doc, ch.start, ch.end, label=it_sp.label_))
            prev = cut + 1
        ch = doc.char_span(prev, it_sp.end_char, alignment_mode="expand")
        if ch is not None:
            refined_item_spans.append(Span(doc, ch.start, ch.end, label=it_sp.label_))

    # finalize entities
    all_spans = _dedup_spans(org_spans + heading_leaf_spans + refined_item_spans)
    all_spans = filter_spans(all_spans)
    doc.ents = tuple(_dedup_spans(all_spans))

    # build sections_tree with items
    sections_tree: List[Dict] = []
    items_per_leaf: Dict[int, List[Dict]] = defaultdict(list)
    seen_item_spans_per_leaf: Dict[int, set] = defaultdict(set)

    for leaf in leaves:
        sc, ec = leaf["text_range"]
        lid = id(leaf)
        for sp in doc.ents:
            if sp.label_.startswith("Item") and sc <= sp.start_char and sp.end_char <= ec:
                key = (sp.start_char, sp.end_char)
                if key in seen_item_spans_per_leaf[lid]:
                    continue
                seen_item_spans_per_leaf[lid].add(key)
                items_per_leaf[lid].append({
                    "text": clean_item_text(sp.text),
                    "span": {"start": sp.start_char, "end": sp.end_char}
                })

    # dedupe leaves by (path, heading span) and merge items
    tmp = []
    for leaf in leaves:
        tmp.append({
            "path": leaf["path"],
            "surface": leaf["surface"],
            "span": leaf["span"],
            "items": items_per_leaf.get(id(leaf), [])
        })

    deduped = []
    seen = {}
    for s in tmp:
        key = (tuple(s["path"]), s["span"]["start"], s["span"]["end"])
        if key in seen:
            seen[key]["items"].extend(s["items"])
        else:
            seen[key] = {**s, "items": list(s["items"])}
            deduped.append(seen[key])

    sections_tree = deduped
    return doc, sections_tree

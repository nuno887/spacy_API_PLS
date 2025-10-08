# entities.py
import re
import sys
import unicodedata
import spacy
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.util import filter_spans

# -----------------------------------------------------------------------------
# Pipeline: tokenizer-only (fast; avoids built-in NER conflicts)
# -----------------------------------------------------------------------------
nlp = spacy.load("pt_core_news_lg", disable=["ner", "tagger", "parser", "lemmatizer"])

# -----------------------------------------------------------------------------
# ORG header starters (ALL-CAPS, can span multiple lines)
# -----------------------------------------------------------------------------
HEADER_STARTERS = {
    "SECRETARIA", "SECRETARIAS", "VICE-PRESIDÊNCIA", "VICE-PRESIDENCIA",
    "PRESIDÊNCIA", "PRESIDENCIA", "DIREÇÃO", "DIRECÇÃO",
    "ASSEMBLEIA", "CÂMARA", "CAMARA", "MUNICIPIO",
    "TRIBUNAL", "CONSERVATÓRIA", "CONSERVATORIA",
    "ADMINISTRAÇÃO"
}

# -----------------------------------------------------------------------------
# Heading taxonomy with levels, canonical names, and aliasing (accent-insensitive)
# Levels: 1=top, 2=sub, 3=sub-sub
# "parents" restricts where a node is valid (for context-sensitive aliases like "Alterações")
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Node:
    canonical: str
    level: int
    aliases: List[str]
    parents: Optional[List[str]] = None  # canonical names of allowed parents (None = top/any)

# Known L1 headings (existing + new "Organizações do Trabalho")
L1_NODES = [
    Node("Despachos", 1, ["Despachos", "Despachos:"]),
    Node("PortariasCondições", 1, ["Portarias de Condições de Trabalho", "Portarias de Condições de Trabalho:"]),
    Node("PortariasExtensao", 1, ["Portarias de Extensão", "Portarias de Extensao", "Portarias de Extensão:", "Portarias de Extensao:"]),
    Node("Convencoes", 1, ["Convenções Coletivas de Trabalho", "Convencoes Coletivas de Trabalho",
                            "Convenções Colectivas de Trabalho", "Convenções Coletivas de Trabalho:",
                            "Convencoes Coletivas de Trabalho:", "Convenções Colectivas de Trabalho:"]),
    Node("Organizações do Trabalho", 1, ["Organizações do Trabalho", "Organizacoes do Trabalho",
                                         "Organizações do Trabalho:", "Organizacoes do Trabalho:"]),
    Node(
    "RegulamentosCondicoesMinimas", 1,
    [
        "Regulamentos de Condições Mínimas",
        "Regulamentos de Condições Mínimas:",
        "Regulamentos de Condicoes Minimas",
        "Regulamentos de Condicoes Minimas:"
    ],),
     Node(
        "RegulamentosExtensao", 1,
        [
            "Regulamentos de Extensão:",
            "Regulamentos de Extensão",
            "Regulamentos de Extensao:"
            "Regulamentos de Extensao"
        ]
    )
]

# L2 under "Organizações do Trabalho"
L2_NODES = [
    Node("Associações Sindicais", 2, ["Associações Sindicais", "Associacoes Sindicais",
                                      "Associações Sindicais:", "Associacoes Sindicais:"],
         parents=["Organizações do Trabalho"]),
    Node("Associações de Empregadores", 2, ["Associações de Empregadores", "Associacoes de Empregadores",
                                            "Associações de Empregadores:", "Associacoes de Empregadores:"],
         parents=["Organizações do Trabalho"]),
]

# L3 under the two L2s: "Estatutos" (aka "Alterações/Alteracoes")
L3_NODES = [
    Node("Estatutos", 3, ["Estatutos", "Estatutos:", "Alterações", "Alteracoes", "Alterações:", "Alteracoes:"],
         parents=["Associações Sindicais", "Associações de Empregadores"]),
]

TAXONOMY: List[Node] = L1_NODES + L2_NODES + L3_NODES

# -----------------------------------------------------------------------------
# Helpers (regexes + utils)
# -----------------------------------------------------------------------------
DOT_LEADER_LINE_RE = re.compile(r'^\s*\.{5,}\s*$')   # line that is only dots
DOT_LEADER_TAIL_RE = re.compile(r'\.{5,}\s*$')       # dots at end of the line
BLANK_RE = re.compile(r'^\s*$')

def _strip_diacritics(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")

def _normalize_heading_text(s: str) -> str:
    # lower, strip diacritics, remove trailing colon/spaces, compress spaces
    s = s.strip()
    s = s[:-1] if s.endswith(":") else s
    s = _strip_diacritics(s).lower()
    s = re.sub(r'\s+', ' ', s)
    return s

def _normalize_aliases(aliases: List[str]) -> List[str]:
    out = set()
    for a in aliases:
        variants = {a, a[:-1] if a.endswith(":") else a}
        for v in variants:
            out.add(_normalize_heading_text(v))
    return sorted(out, key=len, reverse=True)  # longer first

def _is_all_caps_line(ln: str) -> bool:
    t = ln.strip()
    if not t:
        return False
    letters = [ch for ch in t if ch.isalpha()]
    if not letters:
        return False
    return all(ch == ch.upper() for ch in letters)

def _starts_with_starter(ln: str) -> bool:
    t = ln.strip()
    if not t:
        return False
    first = re.split(r'[\s\-–—:,;./]+', t, 1)[0]
    return first.upper() in HEADER_STARTERS

def clean_item_text(raw: str) -> str:
    raw = raw.replace("-\n", "").replace("­\n", "")
    raw = re.sub(r'\s*\n\s*', ' ', raw).strip()
    raw = re.sub(r'\.*\s*$', '', raw).strip()
    return raw

def _dedup_spans(spans: List[Span]) -> List[Span]:
    seen = set()
    out = []
    for s in spans:
        key = (s.start, s.end, s.label_)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out

def canonical_org_key(s: str) -> str:
    """Uppercase, strip diacritics, drop all non-alphanumerics.
    This collapses 'S E C R E T A R I A' and 'Direcção/Dir e c c ã o' to stable keys."""
    t = _strip_diacritics(s).upper()
    return re.sub(r'[^A-Z0-9]+', '', t)

_SUMARIO_PAT = re.compile(r'\bS[UÚ]M[ÁA]RIO\b', re.IGNORECASE)

def find_sumario_anchor(text: str) -> Optional[int]:
    m = _SUMARIO_PAT.search(text)
    return m.start() if m else None

def find_first_l1_heading_after(text: str, start_pos: int) -> Optional[int]:
    """Light hint for 'body' start if no ORG is found right away."""
    # Use your L1 taxonomy aliases (accent/colon tolerant)
    aliases = []
    for node in L1_NODES:
        aliases.extend(node.aliases)
    pats = [re.compile(r'\b' + re.escape(a).replace(r'\:', r':?') + r'\b', re.IGNORECASE) for a in set(aliases)]
    best = None
    for pat in pats:
        m = pat.search(text, pos=start_pos)
        if m:
            if best is None or m.start() < best:
                best = m.start()
    return best


def split_sumario_body(text: str, org_spans_fulltext: List[Span]) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    """
    Return (sumario_span, body_span) as (start,end) into full text.

    Primary: Body starts at the 'second ORG' of the first valid ORG→ORG pair (earliest second occurrence).
    Fallback: first L1 heading after the Sumário anchor (if any), else after start; last resort: a conservative window.
    """
    S = find_sumario_anchor(text)  # may be None

    # 1) Try the 'second ORG' rule
    body_start = _choose_body_start_by_second_org(org_spans_fulltext, text, S)

    # 2) Fallbacks
    if body_start is None:
        # first L1 heading after anchor (or after 0 if no anchor)
        search_from = S if S is not None else 0
        body_start = find_first_l1_heading_after(text, search_from)

    if body_start is None:
        # conservative cap (avoid putting entire doc in Sumário)
        base = S if S is not None else 0
        body_start = min(len(text), base + 20000)

    # 3) Sumário starts at anchor if present; else from start (unchanged behavior)
    sum_start = S if S is not None else 0
    return (sum_start, body_start), (body_start, len(text))


def collect_org_hits_in_span(doc, text: str, span: Tuple[int,int], source: str) -> List[dict]:
    """Filter already-found ORG spans to this [start,end) and return originals + canonical keys."""
    start, end = span
    hits = []
    for sp in doc.ents:
        if sp.label_ != "ORG":
            continue
        if sp.start_char >= start and sp.end_char <= end:
            hits.append({
                "source": source,  # "sumario" or "body"
                "surface_raw": text[sp.start_char:sp.end_char],
                "span": {"start": sp.start_char, "end": sp.end_char},
                "canonical_key": canonical_org_key(text[sp.start_char:sp.end_char]),
            })
    return hits


def link_orgs(sumario_hits: List[dict], body_hits: List[dict]) -> Tuple[List[dict], dict]:
    """Return (relations, diagnostics). Greedy 1–1 pairing by first seen per canonical key."""
    # Index body hits by canonical key, preserve order
    body_by_key: Dict[str, List[dict]] = defaultdict(list)
    for h in body_hits:
        body_by_key[h["canonical_key"]].append(h)

    relations = []
    unmatched_sumario = []
    for h in sumario_hits:
        key = h["canonical_key"]
        lst = body_by_key.get(key, [])
        if lst:
            b = lst.pop(0)  # greedy 1–1
            relations.append({
                "key": key,
                "sumario": {"surface_raw": h["surface_raw"], "span": h["span"]},
                "body": {"surface_raw": b["surface_raw"], "span": b["span"]},
                "confidence": 1.0
            })
        else:
            unmatched_sumario.append(h)

    # Remaining body hits with no pair
    unmatched_body = [b for hits in body_by_key.values() for b in hits]

    diagnostics = {
        "unmatched_sumario_orgs": unmatched_sumario,
        "unmatched_body_orgs": unmatched_body
    }
    return relations, diagnostics

def _choose_body_start_by_second_org(org_spans_fulltext: List[Span],
                                     text: str,
                                     sumario_anchor: Optional[int]) -> Optional[int]:
    """
    Returns the earliest start_char among all 'second occurrences' of any ORG canonical key.
    If sumario_anchor is given, only consider pairs where the 2nd occurrence is after the anchor.
    """
    from collections import defaultdict

    # Build ordered occurrences per canonical key
    occ_by_key = defaultdict(list)  # key -> [start_char, ...] sorted
    for sp in sorted(org_spans_fulltext, key=lambda s: s.start_char):
        surf = text[sp.start_char:sp.end_char]
        key = canonical_org_key(surf)
        occ_by_key[key].append(sp.start_char)

    # Gather candidate 2nd-occurrence starts (optionally filtered by anchor)
    candidates = []
    for key, starts in occ_by_key.items():
        if len(starts) >= 2:
            second = starts[1]
            if sumario_anchor is None or second > sumario_anchor:
                candidates.append(second)

    if not candidates:
        return None
    return min(candidates)


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
    #    Order by heading start; content ends at next heading start (or sumário end)
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

    return sections, relations_section_item, section_ranges

# --- slice-aware ORG collector from spans list ---------------------------
def _collect_org_hits_from_spans(org_spans, text: str, span_range, source: str):
    start, end = span_range
    hits = []
    for sp in org_spans:
        if sp.start_char >= start and sp.end_char <= end:
            surf = text[sp.start_char:sp.end_char]
            hits.append({
                "source": source,  # "sumario" or "body"
                "surface_raw": surf,
                "span": {"start": sp.start_char, "end": sp.end_char},
                "canonical_key": canonical_org_key(surf),
            })
    return hits






# -----------------------------------------------------------------------------
# Build a matcher over ALL aliases (we'll use alias_to_nodes in a line scanner)
# -----------------------------------------------------------------------------
def build_heading_matcher(nlp) -> Tuple[PhraseMatcher, Dict[str, List[Node]]]:
    alias_to_nodes: Dict[str, List[Node]] = defaultdict(list)
    for node in TAXONOMY:
        for norm_alias in _normalize_aliases(node.aliases):
            # prevent duplicate nodes per normalized alias (by canonical)
            if node.canonical not in {n.canonical for n in alias_to_nodes[norm_alias]}:
                alias_to_nodes[norm_alias].append(node)
    return PhraseMatcher(nlp.vocab), alias_to_nodes

# -----------------------------------------------------------------------------
# Heading detection via line scanning (allows diacritic-insensitive matching)
# -----------------------------------------------------------------------------
@dataclass
class HeadingHit:
    canonical: str
    surface: str
    level: int
    start_char: int
    end_char: int

def scan_headings(text: str, alias_to_nodes: Dict[str, List[Node]]) -> List[HeadingHit]:
    lines = text.splitlines(keepends=True)
    # absolute starts for each line
    line_starts = []
    pos = 0
    for ln in lines:
        line_starts.append(pos)
        pos += len(ln)

    hits: List[HeadingHit] = []
    seen_hits = set()  # (start_char, end_char, canonical)

    for i, ln in enumerate(lines):
        surface = ln.strip()
        if not surface:
            continue
        norm = _normalize_heading_text(surface)
        if not norm:
            continue
        nodes = alias_to_nodes.get(norm)
        if not nodes:
            continue
        start_char = line_starts[i]
        end_char = line_starts[i] + len(lines[i])
        for node in nodes:
            key = (start_char, end_char, node.canonical)
            if key in seen_hits:
                continue
            seen_hits.add(key)
            hits.append(HeadingHit(
                node.canonical,
                surface if surface.endswith(":") else surface + ":",
                node.level,
                start_char,
                end_char
            ))
    return hits

# -----------------------------------------------------------------------------
# ORG detector (multi-line ALL-CAPS that starts with a starter token)
# -----------------------------------------------------------------------------
def find_org_spans(doc, text: str) -> List[Span]:
    org_spans = []
    lines = text.splitlines(keepends=True)
    line_starts = []
    pos = 0
    for ln in lines:
        line_starts.append(pos)
        pos += len(ln)

    i = 0
    while i < len(lines):
        if _starts_with_starter(lines[i]) and _is_all_caps_line(lines[i]):
            start_i = i
            j = i + 1
            while j < len(lines) and _is_all_caps_line(lines[j]):
                j += 1
            start_char = line_starts[start_i]
            end_char = line_starts[j - 1] + len(lines[j - 1])
            chspan = doc.char_span(start_char, end_char, alignment_mode="expand")
            if chspan is not None:
                org_spans.append(Span(doc, chspan.start, chspan.end, label="ORG"))
            i = j
        else:
            i += 1
    return org_spans

# -----------------------------------------------------------------------------
# Parse: builds hierarchy with stack + items; returns (doc, sections_tree)
# -----------------------------------------------------------------------------
def parse(text: str, nlp):
    """
    Returns:
      doc               : spaCy Doc with entities (ORG, section leaf spans, items)
      sections_tree     : list of dicts with {path, surface, span, items}
    """
    doc = nlp(text)
    _, alias_to_nodes = build_heading_matcher(nlp)

    # 1) find all heading line hits (may include ambiguous aliases)
    hits = scan_headings(text, alias_to_nodes)
    hits.sort(key=lambda h: h.start_char)

    # 2) resolve ambiguity contextually using a stack (parents)
    stack: List[Tuple[str, int, int, int, str]] = []
    leaves: List[Dict] = []
    leaf_seen = set()  # (leaf_heading_start, leaf_heading_end, tuple(path))

    def close_leaf_if_any(next_start: int):
        """Close current leaf’s text range at next_start (exclusive) and push to leaves."""
        if not stack:
            return
        leaf = stack[-1]
        leaf_path = [s[0] for s in stack]              # canonicals
        leaf_surfaces = [s[4] for s in stack]          # surfaces including colon
        leaf_heading_start = leaf[2]
        leaf_heading_end = leaf[3]
        text_start = leaf_heading_end
        text_end = next_start
        key = (leaf_heading_start, leaf_heading_end, tuple(leaf_path))
        if text_start < text_end and key not in leaf_seen:
            leaf_seen.add(key)
            leaves.append({
                "path": leaf_path[:],
                "surface": leaf_surfaces[:],
                "span": {"start": leaf_heading_start, "end": leaf_heading_end},
                "text_range": (text_start, text_end),
            })

    def allowed_by_parents(node: Node, current_parent: Optional[str]) -> bool:
        if node.parents is None:
            return True
        return current_parent in node.parents

    i = 0
    while i < len(hits):
        hit = hits[i]
        norm = _normalize_heading_text(hit.surface)
        candidates = alias_to_nodes.get(norm, [])
        # choose by allowed parents
        chosen: Optional[Node] = None
        current_parent = stack[-1][0] if stack else None

        for node in sorted(candidates, key=lambda n: (-n.level, -len(_normalize_heading_text(n.canonical)))):
            if allowed_by_parents(node, current_parent):
                chosen = node
                break
        if chosen is None:
            chosen = next((n for n in candidates if n.parents is None), candidates[0])

        # close current leaf up to this heading start
        close_leaf_if_any(hit.start_char)

        # pop until parent fits
        while stack and stack[-1][1] >= chosen.level:
            stack.pop()

        # push chosen heading
        stack.append((chosen.canonical, chosen.level, hit.start_char, hit.end_char, hit.surface))
        i += 1

    # close the last leaf to EOF
    close_leaf_if_any(len(text))

    # 3) Build entity spans: ORG + section leaf spans (canonical labels)
    org_spans = find_org_spans(doc, text)

    heading_leaf_spans: List[Span] = []
    for leaf in leaves:
        start = leaf["span"]["start"]
        end = leaf["span"]["end"]
        chspan = doc.char_span(start, end, alignment_mode="expand")
        if chspan is None:
            continue
        label = leaf["path"][-1]  # canonical leaf label
        heading_leaf_spans.append(Span(doc, chspan.start, chspan.end, label=label))

    # 4) Extract items inside each leaf’s text_range with 3-tier rule
    def find_item_char_spans(full_text: str, start_char: int, end_char: int, next_heading_starts: set):
        """Yield (start_char, end_char) for items that end with:
           - dots-only line,
           - trailing dot leaders,
           - or just before the next heading (fallback)."""
        segment = full_text[start_char:end_char]
        seg_lines = segment.splitlines(keepends=True)

        offs = []
        p = start_char
        for ln in seg_lines:
            offs.append(p)
            p += len(ln)

        block_start = 0
        for i, ln in enumerate(seg_lines):
            # Case 1: pure dots line → close previous block
            if DOT_LEADER_LINE_RE.match(ln):
                s = block_start
                e = i
                while s < e and BLANK_RE.match(seg_lines[s]): s += 1
                j = e - 1
                while j >= s and BLANK_RE.match(seg_lines[j]): j -= 1
                if j >= s:
                    yield offs[s], offs[j] + len(seg_lines[j])
                block_start = i + 1
                continue

            # Case 2: trailing dots on the same line
            m = DOT_LEADER_TAIL_RE.search(ln)
            if m:
                s = block_start
                while s <= i and BLANK_RE.match(seg_lines[s]): s += 1
                if s <= i:
                    end_char_abs = offs[i] + m.start()
                    yield offs[s], end_char_abs
                block_start = i + 1
                continue

            # Case 3: fallback — if next line starts a heading, close before it
            next_line_start = offs[i + 1] if i + 1 < len(seg_lines) else None
            if next_line_start is not None and next_line_start in next_heading_starts:
                s = block_start
                e = i
                while s < e and BLANK_RE.match(seg_lines[s]): s += 1
                j = e
                while j >= s and BLANK_RE.match(seg_lines[j]): j -= 1
                if j >= s:
                    yield offs[s], offs[j] + len(seg_lines[j])
                block_start = i + 1

    # Precompute the set of all heading starts for fallback close
    heading_starts = {leaf["span"]["start"] for leaf in leaves}

    item_spans: List[Span] = []
    for leaf in leaves:
        sc, ec = leaf["text_range"]
        for s_char, e_char in find_item_char_spans(text, sc, ec, heading_starts):
            chspan = doc.char_span(s_char, e_char, alignment_mode="expand")
            if chspan is None:
                continue
            item_spans.append(Span(doc, chspan.start, chspan.end, label=f"Item{leaf['path'][-1]}"))

    # 5) Finalize doc.ents without overlaps or duplicates
    all_spans = org_spans + heading_leaf_spans + item_spans
    all_spans = _dedup_spans(all_spans)          # remove exact duplicates first
    all_spans = filter_spans(all_spans)          # resolve overlaps
    doc.ents = tuple(_dedup_spans(all_spans))    # defensive dedup

    # 6) Build a clean sections_tree with items (dedup items per leaf)
    sections_tree = []
    items_per_leaf: Dict[int, List[Dict]] = defaultdict(list)
    seen_item_spans_per_leaf: Dict[int, set] = defaultdict(set)

    for leaf in leaves:
        sc, ec = leaf["text_range"]
        lid = id(leaf)
        for sp in doc.ents:
            if sp.label_.startswith("Item") and sp.start_char >= sc and sp.end_char <= ec:
                key = (sp.start_char, sp.end_char)
                if key in seen_item_spans_per_leaf[lid]:
                    continue
                seen_item_spans_per_leaf[lid].add(key)
                items_per_leaf[lid].append({
                    "text": clean_item_text(sp.text),
                    "span": {"start": sp.start_char, "end": sp.end_char}
                })

    for leaf in leaves:
        sections_tree.append({
            "path": leaf["path"],
            "surface": leaf["surface"],
            "span": leaf["span"],
            "items": items_per_leaf.get(id(leaf), [])
        })

    return doc, sections_tree

# --- main entry point you can call --------------------------------------
def parse_sumario_and_body_bundle(text_raw: str, nlp):
    """
    Returns: (payload_dict, sumario_text, body_text, text_raw)
    The payload contains sumário structure, section→item relations, ORG↔ORG links, diagnostics, and raw slices.
    """
    # A) ORG scan over the full text (for split + linking)
    doc_full = nlp(text_raw)
    org_spans_full = find_org_spans(doc_full, text_raw)  # existing function

    # B) Split by SECOND-ORG rule (with fallbacks already inside)
    sum_span, body_span = split_sumario_body(text_raw, org_spans_full)
    sum_start, sum_end = sum_span
    body_start, body_end = body_span

    sumario_text = text_raw[sum_start:sum_end]
    body_text    = text_raw[body_start:body_end]

    # C) Parse ONLY the sumário to build its structure
    doc_sum, sections_tree = parse(sumario_text, nlp)

    # D) Assemble sections, relations_section_item, section_ranges (spans → full-text coords)
    sections, rel_section_item, section_ranges = _build_sumario_struct_from_tree(
        sections_tree, offset=sum_start, sumario_len=len(sumario_text)
    )

    # E) ORG hits per slice + ORG↔ORG linking
    sum_orgs  = _collect_org_hits_from_spans(org_spans_full, text_raw, sum_span, source="sumario")
    body_orgs = _collect_org_hits_from_spans(org_spans_full, text_raw, body_span, source="body")
    relations, diag = link_orgs(sum_orgs, body_orgs)  # existing helper

    # F) Diagnostics: how split was chosen
    second_org_pos = _choose_body_start_by_second_org(org_spans_full, text_raw, find_sumario_anchor(text_raw))
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







































# -----------------------------------------------------------------------------
# Pretty-printer for quick testing
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# CLI / quick test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            _text = f.read()
    else:
        _text = (
            """
SECRETARIAREGIONAL DOS RECURSOS HUMANOS
Direcção Regional do Trabalho
Regulamentação do Trabalho
Despachos:
"Teixeira Duarte - Engenharia e Construções, S.A.", - Autorização para adopção de 
período de laboração com amplitude superior aos limites normais..................................
Regulamentos de Condições Mínimas:
Portaria que Aprova o Regulamento de Condições Mínimas para o sector da Indústria
Hoteleira da Região Autónoma da Madeira.....................................................................
Regulamentos de Extensão:
Portaria n.º 22/RE/2008 - Aprova o Regulamento de Extensão do CCT entre a ANF -
Associação Nacional das Farmácias e o SINPROFARM - Sindicato Nacional dos
Profissionais de Farmácia - Alteração Salarial e Outras.................................................
Portaria n.º 23/RE/2008 - Aprova o Regulamento de Extensão do ACTentre a Empresa
de Navegação Madeirense, Ld.ª, e Outras e a FESMAR - Federação de Sindicatos dos
Trabalhadores do Mar - Alteração Salarial e Outras........................................................
Aviso de Projecto de Portaria que Aprova o Regulamento de Extensão do Contrato
Colectivo de Trabalho Vertical entre a ACIF - CCIM - Associação Comercial e
Industrial do Funchal - Câmara de Comércio e Indústria da Madeira e o SITAM -
Sindicato dos Trabalhadores de Escritório, Comércio e Serviços da R.A.M. - Para o
Sector de Armazenamento, Engarrafamento, Comércio por grosso e Exportação do
Vinho da Madeira na Região Autónoma da Madeira - Revisão Salarial.........................
Aviso de Projecto de Portaria que Aprova o Regulamento de Extensão do Contrato
Colectivo de Trabalho entre a Associação Comercial e Industrial do Funchal - Câmara
de Comércio e Indústria da Madeira e o Sindicato das Indústrias Eléctricas do Sul e
Ilhas - Revisão Salarial..................................................................................................
Aviso de Projecto de Portaria que Aprova o Regulamento de Extensão do CCT entre a
AEEP - Associação dos Estabelecimentos de Ensino Particular e Cooperativo e a
FENPROF - Federação Nacional dos Professores e Outros - Alteração Salarial e
Outras.................................................................................................................................
Aviso de Projecto de Portaria que Aprova o Regulamento de Extensão do Contrato
Colectivo de Trabalho entre a ATMARAM - Associação de Transportes de Mercadorias
em Aluguer da Região Autónoma da Madeira e o Sindicato dos Trabalhadores de
Transportes Rodoviários da Região Autónoma da Madeira - Tabelas Salariais e
Outras.................................................................................................................................
Convenções Colectivas de Trabalho:
Contrato Colectivo de Trabalho Vertical entre a ACIF - CCIM - Associação Comercial
e Industrial do Funchal - Câmara de Comércio e Indústria da Madeira e o SITAM -
Sindicato dos Trabalhadores de Escritório, Comércio e Serviços da R.A.M. - Para o
Sector de Armazenamento, Engarrafamento, Comércio por grosso e Exportação do
Vinho da Madeira na Região Autónoma da Madeira - Revisão
Salarial...............................................................................................................................
Contrato Colectivo de Trabalho entre a Associação Comercial e Industrial do Funchal -
Câmara de Comércio e Indústria da Madeira e o Sindicato das Indústrias Eléctricas do
Sul e Ilhas - Revisão Salarial.............................................................................................. 
C C T entre a A E E P - Associação dos Estabelecimentos de Ensino Particular e
Cooperativo e a FENPROF - Federação Nacional dos Professores e Outros - Alteração
Salarial e Outras.................................................................................................................
CCTentre a APEB - Associação Portuguesa das Empresas de Betão Pronto e a FETESE
- Federação dos Sindicatos dos Trabalhadores de Serviços e outros (revisão global) -
Rectificação.......................................................................................................................
Contrato Colectivo de Trabalho entre a ATMARAM - Associação de Transportes de
Mercadorias em Aluguer da Região Autónoma da Madeira e o Sindicato dos
Trabalhadores de Transportes Rodoviários da Região Autónoma da Madeira -Tabelas
Salariais e Outras. ..............................................................................................................

SECRETARIAREGIONAL DOS RECURSOS HUMANOS

"""
        )

    doc, sections_tree = parse(_text, nlp)
    print_results(doc, sections_tree)

    doc = nlp(_text)
    org_full = find_org_spans(doc, _text)                  # existing function
    doc.ents = tuple(filter_spans(org_full))               # keep only ORG for this flow

    # 2) Split Sumário / Body (layout-free)
    sum_span, body_span = split_sumario_body(_text, org_full)

    # 3) Collect ORGs per slice (preserving originals + offsets + keys)
    sum_orgs = collect_org_hits_in_span(doc, _text, sum_span, source="sumario")
    body_orgs = collect_org_hits_in_span(doc, _text, body_span, source="body")

    # 4) Link them by canonical key
    relations, diag = link_orgs(sum_orgs, body_orgs)

    # 5) Quick prints
    print("\n=== SPLIT ===")
    print(f"Sumário: {sum_span[0]}..{sum_span[1]}  | len={sum_span[1]-sum_span[0]}")
    print(f"Body   : {body_span[0]}..{body_span[1]} | len={body_span[1]-body_span[0]}")

    print("\n=== ORG → ORG RELATIONS ===")
    for r in relations:
        print(f"- {r['key']}")
        print(f"  sumário: '{r['sumario']['surface_raw']}' @{r['sumario']['span']['start']}..{r['sumario']['span']['end']}")
        print(f"  body   : '{r['body']['surface_raw']}' @{r['body']['span']['start']}..{r['body']['span']['end']}")
        print(f"  conf   : {r['confidence']}")

    if diag["unmatched_sumario_orgs"] or diag["unmatched_body_orgs"]:
        print("\n=== DIAGNOSTICS ===")
        if diag["unmatched_sumario_orgs"]:
            print("Unmatched Sumário ORGs:")
            for h in diag["unmatched_sumario_orgs"]:
                print(f"  - '{h['surface_raw']}' @{h['span']['start']}..{h['span']['end']} | key={h['canonical_key']}")
        if diag["unmatched_body_orgs"]:
            print("Unmatched Body ORGs:")
            for h in diag["unmatched_body_orgs"]:
                print(f"  - '{h['surface_raw']}' @{h['span']['start']}..{h['span']['end']} | key={h['canonical_key']}")

    
# Derive a quick label for how body_start was chosen (debug only)
    strategy = "second_org_pair" if _choose_body_start_by_second_org(org_full, _text, find_sumario_anchor(_text)) == body_span[0] \
            else "fallback_first_l1_or_window"
    print(f"\n=== SPLIT ===  (strategy: {strategy})")
    print(f"Sumário: {sum_span[0]}..{sum_span[1]}  | len={sum_span[1]-sum_span[0]}")
    print(f"Body   : {body_span[0]}..{body_span[1]} | len={body_span[1]-body_span[0]}")

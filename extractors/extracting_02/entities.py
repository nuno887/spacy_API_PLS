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
        # add both as-is and without trailing colon
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

def clean_item_text(raw: str) -> str:
    raw = raw.replace("-\n", "").replace("­\n", "")
    raw = re.sub(r'\s*\n\s*', ' ', raw).strip()
    raw = re.sub(r'\.*\s*$', '', raw).strip()
    return raw

# -----------------------------------------------------------------------------
# Build a matcher over ALL aliases (accents and no-accents); keep map → Node
# -----------------------------------------------------------------------------
# --- CHANGED (inside build_heading_matcher) ---
def build_heading_matcher(nlp) -> Tuple[PhraseMatcher, Dict[str, List[Node]]]:
    alias_to_nodes: Dict[str, List[Node]] = defaultdict(list)
    patterns = []
    for node in TAXONOMY:
        for norm_alias in _normalize_aliases(node.aliases):
            # prevent duplicate nodes per normalized alias
            if node.canonical not in {n.canonical for n in alias_to_nodes[norm_alias]}:
                alias_to_nodes[norm_alias].append(node)
            # (we don't actually use 'patterns' later; leave as-is)
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
    seen_hits = set() # start and end char
    for i, ln in enumerate(lines):
        surface = ln.strip()
        if not surface:
            continue
        norm = _normalize_heading_text(surface)
        if not norm:
            continue
        # try to match full line to alias
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
# sections_tree is a list of leaf nodes with path/surface/span/items
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
    # sort by position
    hits.sort(key=lambda h: h.start_char)

    # 2) resolve ambiguity contextually using a stack (parents)
    # Stack holds tuples: (canonical, level, start_char_of_heading, end_char_of_heading, surface)
    stack: List[Tuple[str, int, int, int, str]] = []
    # The leaves we collect
    leaves: List[Dict] = []
    leaf_seen = set() # leaf_heading_start, leaf_eading_end, tuple(path)

    def close_leaf_if_any(next_start: int):
        if not stack:
            return
        leaf = stack[-1]
        leaf_path = [s[0] for s in stack]
        leaf_surfaces = [s[4] for s in stack]
        leaf_heading_start = leaf[2]
        leaf_heading_end = leaf[3]
        key = (leaf_heading_start, leaf_heading_end, tuple(leaf_path))
        text_start = leaf_heading_end
        text_end = next_start
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
        # for this line, there may be multiple candidate nodes (same alias); choose one by context
        # gather all nodes that produce this hit (same normalized alias); we must recompute from alias map:
        norm = _normalize_heading_text(hit.surface)
        candidates = alias_to_nodes.get(norm, [])
        # choose by allowed parents + level rules
        chosen: Optional[Node] = None
        # current parent canonical on stack (None if empty)
        current_parent = stack[-1][0] if stack else None
        current_level = stack[-1][1] if stack else 0

        # priority: (1) allowed_by_parents, (2) highest level > current_level? if equal or lower, still valid but will pop appropriately.
        for node in sorted(candidates, key=lambda n: (-n.level, -len(_normalize_heading_text(n.canonical)))):  # prefer deeper, more specific
            if allowed_by_parents(node, current_parent):
                chosen = node
                break
        if chosen is None:
            # fallback: take the one with no parent restrictions or the first
            chosen = next((n for n in candidates if n.parents is None), candidates[0])

        # close leaf up to this heading start
        close_leaf_if_any(hit.start_char)

        # pop until stack parent fits the chosen node
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
            abs_line_start = offs[i]

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

    # 5) Finalize doc.ents without overlaps
    all_spans = org_spans + heading_leaf_spans + item_spans
    all_spans = _dedup_spans(all_spans)
    all_spans = filter_spans(all_spans)
    doc.ents = tuple(_dedup_spans(all_spans)) # aranoic dedup, delete later...

    # 6) Build a clean sections_tree with items
    sections_tree = []
    # index items by range for quick lookup
    item_by_leaf: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
    
    for sp in doc.ents:
        if sp.label_.startswith("Item"):
            item_by_leaf[(sp.start_char, sp.end_char)]  # placeholder to avoid lint noise
    # simpler: collect items per leaf by containment
    items_per_leaf: Dict[int, List[Dict]] = defaultdict(list)  # key by id(leaves[i])
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
            # Sample combining your earlier and newer examples
            "ADMINISTRAÇÃO PÚBLICA REGIONAL - RELAÇÕES COLETIVAS\n"
            "DE TRABALHO\n"
            "Acordo Coletivo n.º 9/2014 - Acordo Coletivo de Entidade Empregadora Pública\n"
            "celebrado entre a Assembleia Legislativa da Madeira, o Sindicato dos Trabalhadores\n"
            "da Função Pública da Região Autónoma da Madeira e o Sindicato dos Trabalhadores\n"
            "Administração Pública e de Entidades com Fins Públicos. .........................................\n"
            "Acordo Coletivo n.º 10/2014 - Acordo Coletivo de Empregador Público celebrado\n"
            "entre a Secretaria dos Assuntos Sociais - SRAS, a Secretaria Regional do Plano e\n"
            "Finanças - SRPF, a Vice-Presidência do Governo da Região Autónoma da Madeira -\n"
            "VP, o Serviço de Saúde da Região Autónoma da Madeira, E.P.E. - SESARAM, a\n"
            "Federação dos Sindicatos da Administração Pública - FESAP, o Sindicato dos\n"
            "Trabalhadores da Função Pública da Região Autónoma da Madeira - STFP, RAM e\n"
            "o Sindicato Nacional dos Técncos Superiores de Saúde das Áreas de Diagnóstico e\n"
            "Terapêutica - SNTSSDT. ..............................................................................................\n"
            "SECRETARIA REGIONAL DA EDUCAÇÃO E RECURSOS HUMANOS\n"
            "Direção Regional do Trabalho\n"
            "Regulamentação do Trabalho\n"
            "Despachos:\n"
            "“Capio - Consultoria e Comércio, Lda” - Autorização para Adoção de Período de\n"
            "Laboração com Amplitude Superior aos Limites Normais. .........................................\n"
            "Portarias de Condições de Trabalho:\n"
            "Portarias de Extensão:\n"
            "Aviso de Projeto de Portaria de Extensão do Acordo de Empresa celebrado entre o\n"
            "Serviço de Saúde da Região Autónoma da Madeira, E.P.E. - SESARAM, a Federação\n"
            "dos Sindicatos da Administração Pública - FESAP, o Sindicato dos Trabalhadores da\n"
            "Função Pública da Região Autónoma da Madeira - STFP, RAM e o Sindicato Nacional\n"
            "dos Técnicos Superiores de Saúde das Áreas de Diagnóstico e Terapêutica - SNTSSDT.\n"
            "Organizações do Trabalho:\n"
            "Associações Sindicais:\n"
            "Estatutos:\n"
            "Sindicato Democrático dos Professores da Madeira - Alteração. ...............................\n"
            "Associações de Empregadores:\n"
            "Alterações:\n"
            "Associação Comercial e Industrial do Funchal - Câmara de Comércio e Indústria da\n"
            "Madeira - Alteração. ..............................................................................................\n"
        )

    doc, sections_tree = parse(_text, nlp)
    print_results(doc, sections_tree)

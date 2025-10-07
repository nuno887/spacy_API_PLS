import re
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.util import filter_spans

import sys

#------------------------------HELPERS--------------------------

def print_results(doc, sections, section_to_items):
    print("\n=== HEADINGS ===")
    # Heading entities are those whose labels match your mapped section labels
    heading_labels = set(HEADING_TO_LABEL.values())
    for ent in doc.ents:
        if ent.label_ in heading_labels:
            print(f"[{ent.label_}] '{ent.text}' @{ent.start_char}:{ent.end_char}")

    print("\n=== ITEMS PER SECTION ===")
    for sect, items in section_to_items.items():
        if not items:
            continue
        print(f"\n{sect}: ({len(items)} item(s))")
        for i, it in enumerate(items, 1):
            print(f"  {i:02d}. {it}")

    print("\n=== ALL ENTITY SPANS (debug) ===")
    for ent in doc.ents:
        print(f"{ent.label_:<18} @{ent.start_char:>5}-{ent.end_char:<5} | {repr(ent.text)}")



def _is_all_caps_line(ln: str) -> bool:
    """True if the line has at least one letter and all letters are uppercase (Unicode-aware)."""
    t = ln.strip()
    if not t:
        return False
    letters = [ch for ch in t if ch.isalpha()]
    if not letters:
        return False
    return all(ch == ch.upper() for ch in letters)

def _starts_with_starter(ln: str) -> bool:
    """True if the line begins with a known starter token (after trimming)."""
    import re
    t = ln.strip()
    if not t:
        return False
    # first token (split by whitespace or common punctuation)
    first = re.split(r'[\s\-–—:,;./]+', t, 1)[0]
    return first.upper() in HEADER_STARTERS

#---------------------------------------------------------------

# Your pipeline (tokenizer only)
nlp = spacy.load("pt_core_news_lg", disable=["ner", "tagger", "parser", "lemmatizer"])

HEADING_TO_LABEL = {
    "Despachos:": "Despachos",
    "Portarias de Condições de Trabalho:": "PortariasCondições",
    "Portarias de Extensão:": "PortariasExtensao",
    "Convenções Coletivas de Trabalho:": "Convencoes",
}

HEADER_STARTERS = {
    "SECRETARIA", "SECRETARIAS", "VICE-PRESIDÊNCIA", "VICE-PRESIDENCIA",
    "PRESIDÊNCIA", "PRESIDENCIA", "DIREÇÃO", "DIRECÇÃO",
    "ASSEMBLEIA", "CÂMARA", "CAMARA", "MUNICIPIO",
    "TRIBUNAL", "CONSERVATÓRIA", "CONSERVATORIA",
    "ADMINISTRAÇÃO"
}

# Build variants (with/without colon)
VARIANTS = {}
for k, v in HEADING_TO_LABEL.items():
    VARIANTS[k] = v
    VARIANTS[k[:-1] if k.endswith(":") else k] = v

matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
matcher.add("HEADINGS", [nlp.make_doc(h) for h in VARIANTS.keys()])

DOT_LEADER_RE = re.compile(r'^\s*\.{5,}\s*$', re.M)
BLANK_RE = re.compile(r'^\s*$', re.M)

def clean_item_text(raw: str) -> str:
    raw = raw.replace("-\n", "").replace("­\n", "")
    raw = re.sub(r'\s*\n\s*', ' ', raw).strip()
    raw = re.sub(r'\.*\s*$', '', raw).strip()
    return raw

def find_item_char_spans(full_text: str, start_char: int, end_char: int):
    """Yield (start_char, end_char) of blocks ending with a dot-leader line inside [start_char, end_char)."""
    segment = full_text[start_char:end_char]
    lines = segment.splitlines(keepends=True)
    # absolute offsets of each line start
    offs = []
    p = start_char
    for ln in lines:
        offs.append(p)
        p += len(ln)

    block_start = 0
    for i, ln in enumerate(lines):
        if DOT_LEADER_RE.match(ln):
            s = block_start
            e = i  # exclude dots line
            # trim blank lines inside block
            while s < e and BLANK_RE.match(lines[s]): s += 1
            j = e - 1
            while j >= s and BLANK_RE.match(lines[j]): j -= 1
            if j >= s:
                yield offs[s], offs[j] + len(lines[j])
            block_start = i + 1

def parse(text: str, nlp):
    """
    Parse a document to extract:
      - Heading entities (custom labels from HEADING_TO_LABEL)
      - Item entities inside each section (label = "Item{SectionLabel}")
      - ORG entities: multi-line ALL-CAPS headers starting with HEADER_STARTERS
    Returns: (doc, sections, section_to_items)
    """
    import re
    from spacy.matcher import PhraseMatcher
    from spacy.tokens import Span
    from spacy.util import filter_spans

    # ---- Config expected to exist in caller's module ------------------------
    # HEADING_TO_LABEL: dict like {"Despachos:": "Despachos", ...}
    # HEADER_STARTERS: set like {"ADMINISTRAÇÃO", "SECRETARIA", ...}
    # -------------------------------------------------------------------------

    # Helpers (self-contained)
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

    DOT_LEADER_RE = re.compile(r'^\s*\.{5,}\s*$', re.M)
    BLANK_RE = re.compile(r'^\s*$', re.M)

    def clean_item_text(raw: str) -> str:
        raw = raw.replace("-\n", "").replace("­\n", "")
        raw = re.sub(r'\s*\n\s*', ' ', raw).strip()
        raw = re.sub(r'\.*\s*$', '', raw).strip()
        return raw

    def find_item_char_spans(full_text: str, start_char: int, end_char: int):
        """Yield (start_char, end_char) for blocks ending with a dot-leader line in [start_char, end_char)."""
        segment = full_text[start_char:end_char]
        lines = segment.splitlines(keepends=True)
        offs = []
        p = start_char
        for ln in lines:
            offs.append(p)
            p += len(ln)

        block_start = 0
        for i, ln in enumerate(lines):
            if DOT_LEADER_RE.match(ln):
                s = block_start
                e = i  # exclude the dots line
                while s < e and BLANK_RE.match(lines[s]): s += 1
                j = e - 1
                while j >= s and BLANK_RE.match(lines[j]): j -= 1
                if j >= s:
                    yield offs[s], offs[j] + len(lines[j])
                block_start = i + 1

    # ---------- Build matcher for headings (with and without colon) ----------
    VARIANTS = {}
    for k, v in HEADING_TO_LABEL.items():
        VARIANTS[k] = v
        if k.endswith(":"):
            VARIANTS[k[:-1]] = v  # no-colon variant

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher.add("HEADINGS", [nlp.make_doc(h) for h in VARIANTS.keys()])

    # ---------- Tokenize only (pipeline is already disabled outside) ----------
    doc = nlp(text)

    # ---------- Headings as entities ----------
    heading_spans = []
    for _, start, end in matcher(doc):
        txt = doc[start:end].text
        key = txt if txt in VARIANTS else txt.rstrip(":")
        label = VARIANTS.get(key)
        if label:
            heading_spans.append(Span(doc, start, end, label=label))
    heading_spans = sorted(filter_spans(heading_spans), key=lambda s: s.start)

    # ---------- ORG multi-line ALL-CAPS headers ----------
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

    # ---------- Section slicing helpers ----------
    def section_bounds(i):
        sc = heading_spans[i].end_char
        ec = heading_spans[i+1].start_char if i+1 < len(heading_spans) else len(doc.text)
        return sc, ec

    # ---------- Items inside each section ----------
    item_spans = []
    for i, h in enumerate(heading_spans):
        sc, ec = section_bounds(i)
        for s_char, e_char in find_item_char_spans(doc.text, sc, ec):
            chspan = doc.char_span(s_char, e_char, alignment_mode="expand")
            if chspan is None:
                continue
            item_spans.append(Span(doc, chspan.start, chspan.end, label=f"Item{h.label_}"))

    # ---------- Finalize entities (de-overlap) ----------
    all_spans = heading_spans + item_spans + org_spans
    doc.ents = tuple(filter_spans(all_spans))

    # ---------- Build structured outputs ----------
    sections = {h.label_: [] for h in heading_spans}
    for i, h in enumerate(heading_spans):
        sc, ec = section_bounds(i)
        block = doc.text[sc:ec]
        for ln in block.splitlines():
            t = ln.strip()
            if not t or DOT_LEADER_RE.fullmatch(t):
                continue
            sections[h.label_].append(t.strip(".·• ").rstrip(".·• "))

    section_to_items = {h.label_: [] for h in heading_spans}
    for ent in doc.ents:
        if ent.label_.startswith("Item"):
            parent = ent.label_.replace("Item", "", 1)
            section_to_items.setdefault(parent, []).append(clean_item_text(ent.text))

    return doc, sections, section_to_items


if __name__ == "__main__":
    # If a file path is provided, read from it; else use a small inline sample
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            _text = f.read()
    else:
        _text = (
            """
SECRETARIA REGIONAL DA EDUCAÇÃO E RECURSOS HUMANOS
Direção Regional do Trabalho
Regulamentação do Trabalho
Despachos:
...
Portarias de Condições de Trabalho:
...
Portarias de Extensão:
Portaria de Extensão n.º 5/2014 - Portaria de Extensão do Acordo Coletivo de Trabalho
celerado entre a SIM - Sociedade Insular de Moagens (Sociedade Unipessoal), S.A. e
Outras e o Sindicato dos Trabalhadores na Hotelaria, Turismo, Alimentação, Serviços
e Similares da R.A.M. - Revisão Salarial e Outras. ........................................................
Aviso de Projeto de Portaria de Extensão do Contrato Coletivo entre a Associação
Portuguesa das Empresas do Setor Elétrico e Eletrónico e a FETESE - Federação dos
Sindicatos da Indústria e Serviços e Outros - Alteração Salarial. ...................................
Convenções Coletivas de Trabalho:
Contrato Coletivo entre a Associação Portuguesa das Empresas do Setor Elétrico e
Eletrónico e a FETESE - Federação dos Sindicatos da Indústria e Serviços e Outros -
Alteração Salarial.
....................................................................................................
2
2
3
"""
        )

    doc, sections, section_to_items = parse(_text, nlp)
    print_results(doc, sections, section_to_items)
# === END ADDITIONS ============================================================
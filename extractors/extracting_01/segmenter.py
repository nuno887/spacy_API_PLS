from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from spacy.tokens import Doc, Span
from .models import Sumario

import unicodedata

# -------- helpers --------

# ADD near helpers
def _norm_org_tokens(s: str) -> List[str]:
    # collapse ws, strip diacritics, uppercase, then split to tokens
    return _strip_diacritics(_collapse_ws(s)).upper().strip(",.;:").split()

def _is_token_prefix(a: List[str], b: List[str], min_shared: int = 5) -> bool:
    """
    Return True if one list is a prefix of the other and the shared prefix
    has at least `min_shared` tokens.
    """
    if not a or not b:
        return False
    m = min(len(a), len(b))
    if m < min_shared:
        return False
    return a[:m] == b[:m]


def _strip_diacritics(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def _coalesce_split_orgs(roster_orgs: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    If an ORG has exactly one SUBORG and there exists a later ORG whose org_text equals
    the concatenation of ORG + SUBORG (whitespace-collapsed, punctuation-trimmed),
    then:
      - replace the earlier entry's org_text with the concatenation,
      - clear its suborg_texts,
      - merge doc_texts (dedupe, preserving order),
      - delete the later duplicate entry,
      - keep the earlier position.
    Repeat until no such pairs remain.
    """
    def norm_key(s: str) -> str:
        return _norm_org(s)  # collapse ws, upper, strip trailing ,.;:

    i = 0
    while i < len(roster_orgs):
        cur = roster_orgs[i]
        subs = cur.get("suborg_texts") or []
        if len(subs) == 1:
            combined = _collapse_ws(f'{cur.get("org_text","")} {subs[0]}')
            key_combined = norm_key(combined)

            # find a *later* org with exactly this combined text
            j = i + 1
            found = -1
            while j < len(roster_orgs):
                if norm_key(roster_orgs[j].get("org_text","")) == key_combined:
                    found = j
                    break
                j += 1

            if found != -1:
                # merge docs (dedupe, preserve order: earlier first)
                docs_i = cur.get("doc_texts") or []
                docs_j = roster_orgs[found].get("doc_texts") or []
                merged_docs = []
                for t in docs_i + docs_j:
                    if t and t not in merged_docs:
                        merged_docs.append(t)

                # mutate earlier entry
                cur["org_text"] = combined
                cur["suborg_texts"] = []
                cur["doc_texts"] = merged_docs

                # drop the later duplicate entry
                del roster_orgs[found]
                # do not advance i; there might be more to coalesce relative to this slot
                continue
        i += 1

    return roster_orgs


def _collapse_ws(s: str) -> str:
    return " ".join(s.split())

def _ents_in_order(doc: Doc) -> List[Span]:
    return sorted(doc.ents, key=lambda e: (e.start_char, -e.end_char))

def _next_entity_start_after(doc: Doc, pos: int) -> int:
    nxt = [e.start_char for e in doc.ents if e.start_char > pos]
    return min(nxt) if nxt else len(doc.text)

def _relations_of_type(doc: Doc, rel_label: str) -> List[dict]:
    return [r for r in getattr(doc._, "relations", []) if r.get("relation") == rel_label]

def _span_by_offsets(doc: Doc, start: int, end: int, label: Optional[str] = None) -> Optional[Span]:
    for e in doc.ents:
        if e.start_char == start and e.end_char == end and (label is None or e.label_ == label):
            return e
    return None

def _merge_orphan_orgs(roster_orgs: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    If an ORG has no suborgs and no docs, merge its org_text into the *next* ORG's org_text
    (as a prefix) and drop the orphan entry.
    """
    merged: List[Dict[str, object]] = []
    i = 0
    n = len(roster_orgs)
    while i < n:
        cur = roster_orgs[i]
        has_payload = bool(cur.get("suborg_texts")) or bool(cur.get("doc_texts"))
        if not has_payload and i + 1 < n:
            nxt = dict(roster_orgs[i + 1])  # shallow copy
            nxt["org_text"] = _collapse_ws(f'{cur.get("org_text","")} {nxt.get("org_text","")}')
            merged.append(nxt)
            i += 2  # skip current (orphan) and consume next
        else:
            merged.append(cur)
            i += 1
    return merged


def _norm_org(s: str) -> str:
    return _strip_diacritics(_collapse_ws(s)).upper().strip(",.;:")

def _filter_ents_in_span(doc: Doc, a: int, b: int) -> Dict[str, List[Tuple[int, int, str, str]]]:
    out: Dict[str, List[Tuple[int, int, str, str]]] = {"ORG": [], "DOC": [], "ORG_SECUNDARIA": []}
    for e in doc.ents:
        if a <= e.start_char and e.end_char <= b and e.label_ in out:
            out[e.label_].append((e.start_char, e.end_char, e.label_, e.text))
    for k in out:
        out[k].sort(key=lambda t: (t[0], -t[1]))
    return out

def _filter_relations_in_span(doc: Doc, a: int, b: int) -> List[dict]:
    rels = []
    for r in getattr(doc._, "relations", []):
        hs, he = r["head_offsets"]["start"], r["head_offsets"]["end"]
        ts, te = r["tail_offsets"]["start"], r["tail_offsets"]["end"]
        if a <= hs < b and a <= ts < b:
        #if a <= hs and he <= b and a <= ts and te <= b:
            rels.append(r)
    return rels

def _find_body_start_with_first_repeated_org(doc: Doc) -> int:
    """
    Second occurrence of an ORG marks the body start, where 'occurrence' allows
    an ORG later to be the earlier ORG plus appended tokens (e.g., ORG + SUBORG),
    matched accent-insensitively and with whitespace/punctuation normalized.
    """
    seen_tokens: List[Tuple[List[str], int]] = []
    for e in _ents_in_order(doc):
        if e.label_ != "ORG":
            continue
        cur = _norm_org_tokens(e.text)
        # check against all previously seen ORGs
        for prev_tokens, _ in seen_tokens:
            # treat as repeat if one is a prefix of the other with sufficient overlap
            if _is_token_prefix(cur, prev_tokens) or _is_token_prefix(prev_tokens, cur):
                return e.start_char
        seen_tokens.append((cur, e.start_char))
    return len(doc.text)

# -------- main API --------
def build_sumario_and_body(doc: Doc, include_local_details: bool = False) -> Tuple[Sumario, Dict[str, object], str, List[dict]]:
    """
    Returns:
      - sumario: Sumário block (text + ents + relations) for [0:cut)
      - roster:  minimal, strings-only hand-off for Pass 2:
                 {"cut_index": int, "orgs": [{"org_text": str,
                                              "suborg_texts": [str, ...],
                                              "doc_texts": [str, ...]}, ...]}
      - body_text: original text from cut_index to EOF
      - file_relations: ALL relations found in the whole file (doc._.relations), unfiltered
    """
    # --- Sumário (unchanged) ---
    ents = _ents_in_order(doc)
    cut = _find_body_start_with_first_repeated_org(doc)

    sum_text = doc.text[:cut]
    sum_ents = _filter_ents_in_span(doc, 0, cut)
    sum_rels = _filter_relations_in_span(doc, 0, cut)
    sumario = Sumario(text=sum_text, ents=sum_ents, relations=sum_rels)

    # --- Roster (strings-only, ordered) ---
    cut_index = len(sumario.text)
    ent_text_by_offsets = {
        (st, en, lab): txt
        for lab, lst in sum_ents.items()
        for (st, en, lab, txt) in lst
    }

    # Group Sumário relations by ORG offsets
    org_to_sub: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    org_to_doc: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for r in sum_rels:
        rel = r.get("relation")
        hs, he = r["head_offsets"]["start"], r["head_offsets"]["end"]
        ts, te = r["tail_offsets"]["start"], r["tail_offsets"]["end"]
        if rel == "CONTAINS":
            org_to_sub.setdefault((hs, he), []).append((ts, te))
        elif rel == "SECTION_ITEM":
            org_to_doc.setdefault((hs, he), []).append((ts, te))

    for k in org_to_sub:
        org_to_sub[k].sort(key=lambda p: p[0])
    for k in org_to_doc:
        org_to_doc[k].sort(key=lambda p: p[0])

    # ORG occurrences in Sumário order (use offsets only for sorting; not returned)
    orgs_ordered = sorted(sum_ents.get("ORG", []), key=lambda t: (t[0], -t[1]))

    roster_orgs: List[Dict[str, object]] = []
    for st, en, _, org_text in orgs_ordered:
        key = (st, en)
        sub_texts = [
            ent_text_by_offsets.get((ts, te, "ORG_SECUNDARIA"), "")
            for (ts, te) in org_to_sub.get(key, [])
        ]
        doc_texts = [
            ent_text_by_offsets.get((ts, te, "DOC"), "")
            for (ts, te) in org_to_doc.get(key, [])
            if "," not in ent_text_by_offsets.get((ts, te, "DOC"), "")
        ]
        roster_orgs.append(
            {"org_text": _collapse_ws(org_text),
             "suborg_texts":[_collapse_ws(t) for t in sub_texts], 
             "doc_texts": [_collapse_ws(t) for t in doc_texts],
             }
        )
    roster_orgs = _merge_orphan_orgs(roster_orgs)

    roster_orgs = _coalesce_split_orgs(roster_orgs)

    roster: Dict[str, object] = {"cut_index": cut_index, "orgs": roster_orgs}
    body_text = doc.text[cut_index:]

    print("segmenter.py.body_text:<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", body_text)
    # full, unfiltered relations from the entire document (Sumário + Body)
    file_relations: List[dict] = list(getattr(doc._, "relations", []))
    return sumario, roster, body_text, file_relations


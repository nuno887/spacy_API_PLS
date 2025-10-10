from typing import List, Tuple, Dict
from collections import defaultdict
from spacy.tokens import Span
from .normalization import canonical_org_key
from .constants import SUMARIO_PAT


def collect_org_hits_in_span(doc, text: str, span: Tuple[int,int], source: str) -> List[dict]:
    start, end = span
    hits = []
    DEBUG = True
    for sp in doc.ents:
        if sp.label_ != "ORG": continue
        if sp.start_char >= start and sp.end_char <= end:
            surf = text[sp.start_char:sp.end_char]
            # Prefer key set during detection; fallback to canonical_org_key
            key = getattr(sp._, "org_key", None) or canonical_org_key(surf)
            item = {
                "source": source,
                "surface_raw": surf,
                "span": {"start": sp.start_char, "end": sp.end_char},
                "canonical_key": key,
            }
            hits.append(item)
            if DEBUG:
                preview = surf[:100].replace("\n", " ")
                print(f"[COLLECT ents] {source} key={key} chars={sp.start_char}-{sp.end_char} text='{preview}'")
    return hits

def collect_org_hits_from_spans(org_spans: List[Span], text: str, span_range, source: str):
    start, end = span_range
    hits = []
    DEBUG = True
    for sp in org_spans:
        if sp.start_char >= start and sp.end_char <= end:
            surf = text[sp.start_char:sp.end_char]
            key = getattr(sp._, "org_key", None) or canonical_org_key(surf)
            item = {
                "source": source,
                "surface_raw": surf,
                "span": {"start": sp.start_char, "end": sp.end_char},
                "canonical_key": key,
            }
            hits.append(item)
            if DEBUG:
                preview = surf[:100].replace("\n", " ")
                print(f"[COLLECT spans] {source} key={key} chars={sp.start_char}-{sp.end_char} text='{preview}'")

    return hits

def link_orgs(sumario_hits: List[dict], body_hits: List[dict]) -> Tuple[List[dict], dict]:
    body_by_key: Dict[str, List[dict]] = defaultdict(list)
    for h in body_hits: 
        body_by_key[h["canonical_key"]].append(h)

    relations, unmatched_sumario = [], []
    DEBUG = True
    if DEBUG:
        print(f"[LINK] sumario_hits={len(sumario_hits)} body_hits={len(body_hits)} "
        f"unique_body_keys={len(body_by_key)}")
    for h in sumario_hits:
        key = h["canonical_key"]
        lst = body_by_key.get(key, [])
        if lst:
            b = lst.pop(0)
            relations.append({
                "key": key,
                "sumario": {"surface_raw": h["surface_raw"], "span": h["span"]},
                "body": {"surface_raw": b["surface_raw"], "span": b["span"]},
                "confidence": 1.0
            })
            if DEBUG:
                s_prev = h["surface_raw"][:80].replace("\n", " ")
                print(f"[LINK ?] no body match for key={key} SUM='{s_prev}'")
        else:
            unmatched_sumario.append(h)

    unmatched_body = [b for hits in body_by_key.values() for b in hits]
    return relations, {
        "unmatched_sumario_orgs": unmatched_sumario,
        "unmatched_body_orgs": unmatched_body
    }

def find_sumario_anchor(text: str):
    m = SUMARIO_PAT.search(text)
    return m.start() if m else None

def choose_body_start_by_second_org(org_spans_fulltext: List[Span], text: str, sumario_anchor: int | None):
    occ_by_key = defaultdict(list)
    DEBUG = True

    for sp in sorted(org_spans_fulltext, key=lambda s: s.start_char):
        surf = text[sp.start_char:sp.end_char]
        # Prefer the detector's  key to guarantee concistency
        key = getattr(sp._, "org_key", None) or canonical_org_key(surf)
        occ_by_key[key].append(sp.start_char)
        if DEBUG:
            prev = surf[:80].replace("\n", " ")
            print(f"[SECOND_ORG] key={key} at {sp.start_char} text='{prev}'")

    candidates = []
    for starts in occ_by_key.values():
        if len(starts) >= 2:
            second = starts[1]
            if sumario_anchor is None or second > sumario_anchor:
                candidates.append(second)
    if DEBUG:
        print(f"[SECOND_ORG] sumario_anchor={sumario_anchor} candidates={candidates}")
    return min(candidates) if candidates else None


from typing import Tuple, Optional, List
from spacy.tokens import Span
from .linking import choose_body_start_by_second_org, find_sumario_anchor
from .taxonomy import L1_NODES
import re

def find_first_l1_heading_after(text: str, start_pos: int) -> Optional[int]:
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
    S = find_sumario_anchor(text)
    body_start = choose_body_start_by_second_org(org_spans_fulltext, text, S)
    if body_start is None:
        #Fallback: use the first ORG that appears after the Sumário anchor (if any)
        after = [sp.start_char for sp in org_spans_fulltext if S is None or sp.start_char > S]
        body_start = min(after) if after else len(text)
    sum_start = S if S is not None else 0
    return (sum_start, body_start), (body_start, len(text))

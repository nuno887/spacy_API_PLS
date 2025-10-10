from typing import Dict, List
from spacy.tokens import Doc
from .features import normalize_for_match
from .config import MIN_CONF_TO_ACCEPT

def score_candidate(nlp, item_text: str, frag_text: str, method: str) -> float:
    """Simple spaCy-based similarity via token overlap proxy (robust + fast)."""
    # Basic normalized token overlap
    it = set(t.text.lower() for t in nlp.make_doc(normalize_for_match(item_text)) if t.is_alpha or t.is_digit)
    ft = set(t.text.lower() for t in nlp.make_doc(normalize_for_match(frag_text)) if t.is_alpha or t.is_digit)
    if not it or not ft:
        jac = 0.0
    else:
        jac = len(it & ft) / len(it | ft)

    base = 0.75 if method == "exact" else 0.60
    conf = min(1.0, base * 0.6 + jac * 0.6)
    return conf

def pick_best_candidate(nlp, item_text: str, full_text: str, candidates: List[Dict]) -> Dict:
    if not candidates:
        return {"method": "none", "confidence": 0.0}
    best, best_conf = None, -1.0
    for c in candidates:
        frag = full_text[c["start"]:c["end"]]
        conf = score_candidate(nlp, item_text, frag, c["method"])
        if conf > best_conf:
            best_conf = conf; best = c
    out = dict(best)
    out["confidence"] = best_conf
    if best_conf < MIN_CONF_TO_ACCEPT:
        out["method"] = "none"
    return out

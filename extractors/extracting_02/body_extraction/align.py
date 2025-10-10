from typing import List, Dict, Tuple
from spacy.tokens import Doc
from spacy.matcher import PhraseMatcher
from .features import normalize_for_match, item_anchor_phrases
from .config import MAX_CANDIDATES_PER_ITEM

def _phrase_match(doc: Doc, phrases: List[str]) -> List[Tuple[int, int, str]]:
    """spaCy PhraseMatcher over normalized text by using lower/diacritics-insensitive pipeline behavior."""
    # Build phrase docs (simple lowercased/normalized strings fed back into nlp)
    # We avoid adding to vocab; use on-the-fly docs.
    matcher = PhraseMatcher(doc.vocab, attr="LOWER")
    patterns = []
    labels = []
    for i, p in enumerate(phrases):
        labels.append(f"ANCHOR_{i}")
        patterns.append(doc.vocab.make_doc(p))
    for label, pat in zip(labels, patterns):
        matcher.add(label, [pat])

    spans = []
    for match_id, start, end in matcher(doc):
        label = doc.vocab.strings[match_id]
        span = doc[start:end]
        spans.append((span.start_char, span.end_char, label))
        if len(spans) >= MAX_CANDIDATES_PER_ITEM:
            break
    return spans

def locate_candidates_in_window(nlp, full_text: str, window_start: int, window_end: int, item_text: str) -> List[Dict]:
    """
    Use spaCy doc over the window; try phrase matches of anchor variants.
    Fallback: simple substring search on normalized strings to add a few more candidates.
    """
    window_text = full_text[window_start:window_end]
    doc_win = nlp.make_doc(window_text)  # fast tokenizer-only ok; your pipeline is tokenizer-only anyway
    anchors = item_anchor_phrases(item_text)
    # primary: phrase matcher (LOWER)
    spans = _phrase_match(doc_win, anchors)

    candidates: List[Dict] = []
    for s, e, lbl in spans:
        abs_s = window_start + s
        abs_e = window_start + e
        candidates.append({"start": abs_s, "end": abs_e, "method": "exact", "raw_score": 1.0, "anchor_used": lbl})
        if len(candidates) >= MAX_CANDIDATES_PER_ITEM:
            break

    # secondary: normalized substring (to catch diacritics/quotes variations)
    if len(candidates) < MAX_CANDIDATES_PER_ITEM and anchors:
        win_norm = normalize_for_match(window_text)
        for a in anchors:
            a_norm = normalize_for_match(a)
            pos = 0
            while pos < len(win_norm):
                j = win_norm.find(a_norm, pos)
                if j == -1: break
                # map j in normalized string back to raw text approx by proportional index
                # (safe enough since we only normalize case/diacritics/spaces)
                raw_start = max(0, min(len(window_text), j))
                abs_s = window_start + raw_start
                abs_e = min(window_start + raw_start + max(60, len(a) + 120), window_end)
                candidates.append({"start": abs_s, "end": abs_e, "method": "exact", "raw_score": 0.9, "anchor_used": "norm_substr"})
                if len(candidates) >= MAX_CANDIDATES_PER_ITEM: break
                pos = j + max(1, len(a_norm))
            if len(candidates) >= MAX_CANDIDATES_PER_ITEM: break

    # dedupe by (start,end)
    seen, out = set(), []
    for c in sorted(candidates, key=lambda x: (x["start"], -x["end"])):
        key = (c["start"], c["end"])
        if key in seen: continue
        seen.add(key); out.append(c)
    return out[:MAX_CANDIDATES_PER_ITEM]

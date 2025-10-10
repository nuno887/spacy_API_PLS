# body_extraction/align.py

from typing import List, Dict, Tuple
from spacy.tokens import Doc
from spacy.matcher import PhraseMatcher
from .features import normalize_for_match, item_anchor_phrases
from .config import MAX_CANDIDATES_PER_ITEM


def _phrase_match(nlp, doc: Doc, phrases: List[str]) -> List[Tuple[int, int, str]]:
    """
    Run a PhraseMatcher over the window doc using LOWER attr.
    Patterns are tokenized via the provided `nlp` (so tokenizer rules match).
    Returns list of (start_char, end_char, label).
    """
    matcher = PhraseMatcher(doc.vocab, attr="LOWER")
    patterns = []
    labels = []
    for i, p in enumerate(phrases):
        labels.append(f"ANCHOR_{i}")
        patterns.append(nlp.make_doc(p))  # tokenize patterns with the same nlp

    for label, pat in zip(labels, patterns):
        matcher.add(label, [pat])

    spans: List[Tuple[int, int, str]] = []
    for match_id, start, end in matcher(doc):
        label = doc.vocab.strings[match_id]
        span = doc[start:end]
        spans.append((span.start_char, span.end_char, label))
        if len(spans) >= MAX_CANDIDATES_PER_ITEM:
            break
    return spans


def locate_candidates_in_window(nlp, full_text: str, window_start: int, window_end: int, item_text: str) -> List[Dict]:
    """
    Use spaCy to find candidate locations for `item_text` within [window_start, window_end) of full_text.
    Primary: PhraseMatcher on anchor phrases.
    Secondary: normalized substring fallback (diacritics/whitespace robust).
    Returns list of candidate dicts: {start, end, method, raw_score, anchor_used}.
    """
    window_text = full_text[window_start:window_end]
    doc_win = nlp.make_doc(window_text)  # tokenizer-only is fine

    anchors = item_anchor_phrases(item_text)

    # 1) PhraseMatcher
    spans = _phrase_match(nlp, doc_win, anchors)

    candidates: List[Dict] = []
    for s, e, lbl in spans:
        abs_s = window_start + s
        abs_e = window_start + e
        candidates.append({"start": abs_s, "end": abs_e, "method": "exact", "raw_score": 1.0, "anchor_used": lbl})
        if len(candidates) >= MAX_CANDIDATES_PER_ITEM:
            break

    # 2) Normalized substring fallback (if still short on candidates)
    if len(candidates) < MAX_CANDIDATES_PER_ITEM and anchors:
        win_norm = normalize_for_match(window_text)
        for a in anchors:
            a_norm = normalize_for_match(a)
            pos = 0
            while pos < len(win_norm):
                j = win_norm.find(a_norm, pos)
                if j == -1:
                    break
                # Approximate raw mapping: since normalization only affects case/diacritics/spacing,
                # using the same index is typically close enough for a candidate region.
                raw_start = max(0, min(len(window_text), j))
                abs_s = window_start + raw_start
                abs_e = min(window_start + raw_start + max(60, len(a) + 120), window_end)
                candidates.append({
                    "start": abs_s,
                    "end": abs_e,
                    "method": "exact",
                    "raw_score": 0.9,
                    "anchor_used": "norm_substr",
                })
                if len(candidates) >= MAX_CANDIDATES_PER_ITEM:
                    break
                pos = j + max(1, len(a_norm))
            if len(candidates) >= MAX_CANDIDATES_PER_ITEM:
                break

    # Dedupe by (start, end) and cap
    seen = set()
    out: List[Dict] = []
    for c in sorted(candidates, key=lambda x: (x["start"], -x["end"])):
        key = (c["start"], c["end"])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
        if len(out) >= MAX_CANDIDATES_PER_ITEM:
            break

    return out

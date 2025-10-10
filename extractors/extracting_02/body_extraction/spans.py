# body_extraction/spans.py
from typing import Tuple
from spacy.tokens import Doc

def expand_to_sentence(doc: Doc, start: int, end: int) -> Tuple[int, int]:
    """
    Expand a [start, end) character span to full sentence boundaries using doc.sents.
    - If no sentencizer is present (no sents), returns (start, end).
    - If start==end, snaps to the sentence covering that point.
    - If span crosses multiple sentences, expands from the first to the last.
    """
    # Safety clamps
    start = max(0, min(len(doc.text), start))
    end   = max(start, min(len(doc.text), end))

    # If the pipeline has no sentence boundaries, just return as-is
    if not hasattr(doc, "sents"):
        return start, end

    # Find the first and last sentence covering the span
    first_sent_start = None
    last_sent_end = None

    # Fast path for point spans: pick the sentence containing `start`
    if start == end:
        for sent in doc.sents:
            s0 = sent.start_char
            s1 = sent.end_char
            if s0 <= start < s1:
                return s0, s1
        # If no sentence found (shouldn't happen), return as-is
        return start, end

    # General case: expand to cover all sentences intersecting the span
    for sent in doc.sents:
        s0 = sent.start_char
        s1 = sent.end_char
        # Intersects if there is any overlap between [start,end) and [s0,s1)
        if not (end <= s0 or s1 <= start):
            if first_sent_start is None:
                first_sent_start = s0
            last_sent_end = s1

    if first_sent_start is not None and last_sent_end is not None:
        return first_sent_start, last_sent_end

    # Fallback: if the span didn’t intersect any sentence (e.g., odd tokenization),
    # try snapping to the sentence containing `start`.
    for sent in doc.sents:
        s0 = sent.start_char
        s1 = sent.end_char
        if s0 <= start < s1:
            return s0, s1

    # Final fallback
    return start, end

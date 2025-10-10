from typing import Tuple
from spacy.tokens import Doc
from .features import sent_span_covering
from .config import MAX_SENTENCE_EXPAND

def expand_to_sentence(doc_body: Doc, abs_start: int, abs_end: int) -> Tuple[int, int]:
    return sent_span_covering(doc_body, abs_start, abs_end)

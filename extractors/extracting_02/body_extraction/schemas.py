from typing import TypedDict, List, Literal, Optional, Dict

Method = Literal["exact", "heuristic", "fuzzy", "none"]

class Span(TypedDict):
    start: int
    end: int

class OrgContext(TypedDict, total=False):
    sumario_org: Dict
    body_org: Dict

class ExtractionResult(TypedDict, total=False):
    section_path: List[str]
    section_span: Span
    item_text: str
    item_span_sumario: Span
    org_context: OrgContext
    body_span: Span
    confidence: float
    method: Method
    diagnostics: Dict

class ExtractionReport(TypedDict):
    summary: Dict
    results: List[ExtractionResult]

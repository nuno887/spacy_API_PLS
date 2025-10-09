from dataclasses import dataclass
from typing import List, Dict
from .taxonomy import Node
from .normalization import normalize_heading_text

@dataclass
class HeadingHit:
    canonical: str
    surface: str
    level: int
    start_char: int
    end_char: int

def scan_headings(text: str, alias_to_nodes: Dict[str, List[Node]]) -> List[HeadingHit]:
    lines = text.splitlines(keepends=True)
    line_starts, pos = [], 0
    for ln in lines:
        line_starts.append(pos); pos += len(ln)

    hits, seen = [], set()
    for i, ln in enumerate(lines):
        surface = ln.strip()
        if not surface: continue
        norm = normalize_heading_text(surface)
        if not norm: continue
        nodes = alias_to_nodes.get(norm)
        if not nodes: continue
        start_char = line_starts[i]
        end_char = line_starts[i] + len(lines[i])
        for node in nodes:
            #Dedupe per physical line + canonical, so alias variants on the same line don't double emit
            key = (start_char, node.canonical)
            if key in seen: continue
            seen.add(key)
            hits.append(HeadingHit(
                node.canonical,
                surface if surface.endswith(":") else surface + ":",
                node.level,
                start_char,
                end_char
            ))
    return hits

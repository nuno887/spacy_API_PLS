from dataclasses import dataclass
from typing import List, Dict
from .taxonomy import Node, TAXONOMY
from .normalization import normalize_heading_text, normalize_text_offsetsafe

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
        # normalize NBSP/quotes etc. before inspecting
        raw_line = normalize_text_offsetsafe(ln)
        surface = raw_line.strip()

   

        if not surface: continue
        norm = normalize_heading_text(surface)
        if not norm: continue
        nodes = alias_to_nodes.get(norm)
        # Fallback: heading + content on same line ("Heading: item...") -> match prefix up to the first colon
        if not nodes and ":" in surface:
            prefix_norm = normalize_heading_text(surface.split(":", 1)[0])
            if prefix_norm:
                nodes = alias_to_nodes.get(prefix_norm)
                if nodes:
                    # Treat the printed surface as just the heading label + ":" for consistency
                    surface = surface.split(":", 1)[0].strip() + ":"

        if not nodes: continue
        start_char = line_starts[i]
        end_char = line_starts[i] + len(lines[i])
        for node in nodes:
            # Dedupe per physical line + canonical, so alias variants on the same line don't double emit
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

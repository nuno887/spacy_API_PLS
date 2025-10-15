from dataclasses import dataclass, field
from typing import List

@dataclass
class VirtualTextFile:
    filename: str
    content: bytes
    text: str
    mimetype: str = "text/plain; charset=utf-8"
    log: List[str] = field(default_factory=list)

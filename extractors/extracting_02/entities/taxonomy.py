from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from collections import defaultdict
from spacy.matcher import PhraseMatcher
from .normalization import normalize_heading_text

@dataclass(frozen=True)
class Node:
    canonical: str
    level: int
    aliases: List[str]
    parents: Optional[List[str]] = None

L1_NODES = [
    Node("Despachos", 1, ["Despachos", "Despachos:"]),
    Node("PortariasCondições", 1, ["Portarias de Condições de Trabalho", "Portarias de Condições de Trabalho:"]),
    Node("PortariasExtensao", 1, ["Portarias de Extensão", "Portarias de Extensao", "Portarias de Extensão:", "Portarias de Extensao:"]),
    Node("Convencoes", 1, [
        "Convenções Coletivas de Trabalho", "Convencoes Coletivas de Trabalho",
        "Convenções Colectivas de Trabalho", "Convenções Coletivas de Trabalho:",
        "Convencoes Coletivas de Trabalho:", "Convenções Colectivas de Trabalho:"
    ]),
    Node("Organizações do Trabalho", 1, [
        "Organizações do Trabalho", "Organizacoes do Trabalho",
        "Organizações do Trabalho:", "Organizacoes do Trabalho:"
    ]),
    Node("RegulamentosCondicoesMinimas", 1, [
        "Regulamentos de Condições Mínimas", "Regulamentos de Condições Mínimas:",
        "Regulamentos de Condicoes Minimas", "Regulamentos de Condicoes Minimas:"
    ]),
    Node("RegulamentosExtensao", 1, [
        "Regulamentos de Extensão:", "Regulamentos de Extensão",
        "Regulamentos de Extensao:", "Regulamentos de Extensao"
    ]),
    Node("DespachosConjuntos", 1, ["Despachos Conjuntos:", "Despachos Conjuntos"]),
]

L2_NODES = [
    Node("Associações Sindicais", 2, [
        "Associações Sindicais", "Associacoes Sindicais",
        "Associações Sindicais:", "Associacoes Sindicais:"
    ], parents=["Organizações do Trabalho"]),
    Node("Associações de Empregadores", 2, [
        "Associações de Empregadores", "Associacoes de Empregadores",
        "Associações de Empregadores:", "Associacoes de Empregadores:"
    ], parents=["Organizações do Trabalho"]),
]

L3_NODES = [
    Node("Estatutos", 3, ["Estatutos", "Estatutos:", "Alterações", "Alteracoes", "Alterações:", "Alteracoes:"],
         parents=["Associações Sindicais", "Associações de Empregadores"]),
]

TAXONOMY: List[Node] = L1_NODES + L2_NODES + L3_NODES

def _normalize_aliases(aliases: List[str]) -> List[str]:
    out = set()
    for a in aliases:
        variants = {a, a[:-1] if a.endswith(":") else a}
        for v in variants:
            out.add(normalize_heading_text(v))
    return sorted(out, key=len, reverse=True)

def build_heading_matcher(nlp) -> Tuple[PhraseMatcher, Dict[str, List[Node]]]:
    alias_to_nodes: Dict[str, List[Node]] = defaultdict(list)
    for node in TAXONOMY:
        for norm_alias in _normalize_aliases(node.aliases):
            if node.canonical not in {n.canonical for n in alias_to_nodes[norm_alias]}:
                alias_to_nodes[norm_alias].append(node)
    return PhraseMatcher(nlp.vocab), alias_to_nodes

# body_extraction/body_taxonomy.py
from dataclasses import dataclass
from typing import List, Dict, Pattern, Optional
import re

@dataclass(frozen=True)
class BodySection:
    canonical: str
    header_aliases: List[str]                # how this section header appears in BODY
    item_title_patterns: List[List[dict]]    # spaCy Matcher token patterns (line-start titles)
    item_anchor_phrases: List[str]           # PhraseMatcher phrases (fast anchors)
    stop_cues: Optional[List[str]] = None    # optional extra hard stops while growing

def _p(pat: str) -> Pattern[str]:
    return re.compile(pat, re.IGNORECASE)

# Common number markers for "n.º / nº / n° / no / n."
NUM_MARKERS = ["n.º", "nº", "n°", "no", "n."]

BODY_SECTIONS: Dict[str, BodySection] = {
    "PortariasExtensao": BodySection(
        canonical="PortariasExtensao",
        header_aliases=[
            "Portarias de Extensão", "Portarias de Extensao",
            "Portaria de Extensão", "Portaria de Extensao",
            "Portarias de Extensão:", "Portaria de Extensão:",
        ],
        item_title_patterns=[
            # "Aviso de Projeto/Projecto de Portaria ..."
            [{"LOWER": "aviso"}, {"LOWER": "de"}, {"LOWER": {"IN": ["projeto", "projecto"]}},
             {"LOWER": "de"}, {"LOWER": "portaria"}],
            # "Portaria n.º 5/2020" (num marker optional)
            [{"LOWER": "portaria"}, {"LOWER": {"IN": NUM_MARKERS}, "OP": "?"},
             {"LIKE_NUM": True}, {"TEXT": "/"}, {"LIKE_NUM": True}],
            [{"LOWER": "portaria"}, {"LOWER": {"IN": NUM_MARKERS}, "OP": "?"}, {"LIKE_NUM": True}],
        ],
        item_anchor_phrases=[
            "Aviso de Projeto de Portaria", "Aviso de Projecto de Portaria", "Portaria n.º",
        ],
        stop_cues=None,
    ),

    "Convencoes": BodySection(
        canonical="Convencoes",
        header_aliases=[
            "Convenções Coletivas de Trabalho", "Convenções Colectivas de Trabalho",
            "Convenções Coletivas de Trabalho:",
        ],
        item_title_patterns=[
            # "Contrato Coletivo/Colectivo ..."
            [{"LOWER": "contrato"}, {"LOWER": {"IN": ["coletivo", "colectivo"]}}],
            # "CCT ..."
            [{"LOWER": "cct"}],
            # "Acordo de adesão/adesao ..."
            [{"LOWER": "acordo"}, {"LOWER": "de"}, {"LOWER": {"IN": ["adesão", "adesao"]}}],
            # "Acordo Coletivo ..."
            [{"LOWER": "acordo"}, {"LOWER": {"IN": ["coletivo", "colectivo"]}}],
        ],
        item_anchor_phrases=[
            "Contrato Coletivo", "CCT", "Acordo de adesão", "Acordo Coletivo",
        ],
        stop_cues=None,
    ),

    "Despachos": BodySection(
        canonical="Despachos",
        header_aliases=["Despachos", "Despachos:"],
        item_title_patterns=[
            [{"LOWER": "despacho"}],
            [{"LOWER": "despacho"}, {"LOWER": "conjunto"}],
            [{"LOWER": "despacho"}, {"LOWER": {"IN": NUM_MARKERS}, "OP": "?"}, {"LIKE_NUM": True}],
        ],
        item_anchor_phrases=["Despacho", "Despacho Conjunto", "Despacho n.º"],
    ),

    "PortariasCondições": BodySection(
        canonical="PortariasCondições",
        header_aliases=["Portarias de Condições de Trabalho", "Portarias de Condições de Trabalho:"],
        item_title_patterns=[
            [{"LOWER": "portaria"}],
            [{"LOWER": "portaria"}, {"LOWER": {"IN": NUM_MARKERS}, "OP": "?"}, {"LIKE_NUM": True}],
        ],
        item_anchor_phrases=["Portaria", "Portaria n.º"],
    ),

    "DespachosConjuntos": BodySection(
        canonical="DespachosConjuntos",
        header_aliases=[
            "Despachos Conjuntos", "Despachos Conjuntos:",
        ],
        item_title_patterns=[
            [{"LOWER": "despacho"}, {"LOWER": "conjunto"}],
            [{"LOWER": "despacho"}, {"LOWER": "conjunto"}, {"LOWER": {"IN": NUM_MARKERS}, "OP": "?"}, {"LIKE_NUM": True}],
        ],
        item_anchor_phrases=["Despacho Conjunto", "Despacho Conjunto n.º"],
    ),

    "RegulamentosCondicoesMinimas": BodySection(
        canonical="RegulamentosCondicoesMinimas",
        header_aliases=[
            "Regulamentos de Condições Mínimas", "Regulamentos de Condicoes Minimas",
            "Regulamentos de Condições Mínimas:", "Regulamentos de Condicoes Minimas:",
            "Regulamento de Condições Mínimas", "Regulamento de Condicoes Minimas",
        ],
        item_title_patterns=[
            [{"LOWER": "aviso"}, {"LOWER": "de"}, {"LOWER": {"IN": ["projeto", "projecto"]}}, {"LOWER": "de"}, {"LOWER": "regulamento"}],
            [{"LOWER": "regulamento"}, {"LOWER": "de"}, {"LOWER": "condições"}, {"LOWER": {"IN": ["mínimas", "minimas"]}}],
            [{"LOWER": "regulamento"}],
        ],
        item_anchor_phrases=[
            "Aviso de Projeto de Regulamento",
            "Regulamento de Condições Mínimas",
            "Regulamento",
        ],
    ),

    "RegulamentosExtensao": BodySection(
        canonical="RegulamentosExtensao",
        header_aliases=[
            "Regulamentos de Extensão", "Regulamentos de Extensao",
            "Regulamentos de Extensão:", "Regulamentos de Extensao:",
            "Regulamento de Extensão", "Regulamento de Extensao",
        ],
        item_title_patterns=[
            [{"LOWER": "aviso"}, {"LOWER": "de"}, {"LOWER": {"IN": ["projeto", "projecto"]}}, {"LOWER": "de"}, {"LOWER": "regulamento"}],
            [{"LOWER": "regulamento"}, {"LOWER": "de"}, {"LOWER": {"IN": ["extensão", "extensao"]}}],
            [{"LOWER": "regulamento"}],
        ],
        item_anchor_phrases=[
            "Aviso de Projeto de Regulamento",
            "Regulamento de Extensão",
            "Regulamento",
        ],
    ),

    "AvisosColetivas": BodySection(
        canonical="AvisosColetivas",
        header_aliases=[
            "Avisos de cessação da vigência de convenções coletivas",
            "Avisos de cessacao da vigencia de convencoes coletivas",
            "Avisos de cessação da vigência de convenções coletivas:",
        ],
        item_title_patterns=[
            [{"LOWER": "aviso"}, {"LOWER": "de"}, {"LOWER": {"IN": ["cessação", "cessacao"]}}],
            [{"LOWER": "aviso"}],
        ],
        item_anchor_phrases=["Aviso de cessação", "Aviso"],
    ),

    "Convocatorias": BodySection(
        canonical="Convocatorias",
        header_aliases=[
            "Convocatórias", "Convocatórias:", "Convocatorias", "Convocatorias:",
        ],
        item_title_patterns=[
            [{"LOWER": {"IN": ["convocatória", "convocatoria"]}}],
        ],
        item_anchor_phrases=["Convocatória", "Convocatoria"],
    ),

    "OrganizacoesTrabalho": BodySection(
        canonical="OrganizacoesTrabalho",
        header_aliases=[
            "Organizações do Trabalho", "Organizacoes do Trabalho",
            "Organizações do Trabalho:", "Organizacoes do Trabalho:",
            "ORGANIZAÇÕES DO TRABALHO:",
        ],
        item_title_patterns=[
            [{"LOWER": {"IN": ["associação", "associacao", "federação", "federacao", "sindicato"]}}],
            [{"LOWER": "estatutos"}],
            [{"LOWER": {"IN": ["alteração", "alteracao", "alterações", "alteracoes"]}}]
        ],
        item_anchor_phrases=["Associação", "Federação", "Sindicato", "Estatutos", "Alteração"],
    ),

    "Estatutos": BodySection(
        canonical="Estatutos",
        header_aliases=["Estatutos", "Estatutos:"],
        item_title_patterns=[
            [{"LOWER": "estatutos"}],
            [{"LOWER": {"IN": ["alteração", "alteracao", "alterações", "alteracoes"]}},
             {"LOWER": "dos", "OP": "?"}, {"LOWER": "estatutos"}],
        ],
        item_anchor_phrases=["Estatutos", "Alteração de Estatutos"],
    ),

    "AcordosColetivosTrabalho": BodySection(
        canonical="AcordosColetivosTrabalho",
        header_aliases=[
            "Acordos Coletivos de Trabalho", "Acordos Colectivos de Trabalho",
            "Acordos Coletivos de Trabalho:", "Acordos Colectivos de Trabalho:",
        ],
        item_title_patterns=[
            [{"LOWER": "acordo"}, {"LOWER": {"IN": ["coletivo", "colectivo"]}}, {"LOWER": "de"}, {"LOWER": "trabalho"}],
            [{"LOWER": "acordo"}, {"LOWER": {"IN": ["coletivo", "colectivo"]}}],
            [{"LOWER": "acordo"}, {"LOWER": {"IN": ["coletivo", "colectivo"]}},
             {"LOWER": {"IN": NUM_MARKERS}, "OP": "?"}, {"LIKE_NUM": True}],
        ],
        item_anchor_phrases=["Acordo Coletivo de Trabalho", "Acordo Coletivo", "Acordo Colectivo"],
    ),

    "EleicaoRepresentantes": BodySection(
        canonical="EleicaoRepresentantes",
        header_aliases=[
            "Eleição de Representantes", "Eleicao de Representantes",
            "Eleição de Representantes:", "Eleicao de Representantes:",
        ],
        item_title_patterns=[
            [{"LOWER": {"IN": ["eleição", "eleicao"]}}, {"LOWER": "de"}, {"LOWER": "representantes"}],
            [{"LOWER": {"IN": ["eleição", "eleicao"]}}],
        ],
        item_anchor_phrases=["Eleição de Representantes", "Eleições"],
    ),

    "Eleicoes": BodySection(
        canonical="Eleicoes",
        header_aliases=["Eleições", "Eleicoes", "Eleições:", "Eleicoes:"],
        item_title_patterns=[[{"LOWER": {"IN": ["eleições", "eleicoes"]}}]],
        item_anchor_phrases=["Eleições", "Eleicoes"],
    ),

    "Direcao": BodySection(
        canonical="Direcao",
        header_aliases=["Direção", "Direcao", "Direção:", "Direcao:"],
        item_title_patterns=[
            [{"LOWER": {"IN": ["direção", "direcao"]}}],
            [{"LOWER": "membros"}, {"LOWER": "da"}, {"LOWER": {"IN": ["direção", "direcao"]}}],
        ],
        item_anchor_phrases=["Direção", "Direcao", "Membros da Direção", "Membros da Direcao"],
    ),

    "MembrosDirecao": BodySection(
        canonical="MembrosDirecao",
        header_aliases=["Membros da Direção", "Membros da Direcao"],
        item_title_patterns=[[{"LOWER": "membros"}, {"LOWER": "da"}, {"LOWER": {"IN": ["direção", "direcao"]}}]],
        item_anchor_phrases=["Membros da Direção", "Membros da Direcao"],
    ),

    "CorposGerentes": BodySection(
        canonical="CorposGerentes",
        header_aliases=["Corpos Gerentes", "Corpos Gerentes:"],
        item_title_patterns=[
            [{"LOWER": "corpos"}, {"LOWER": "gerentes"}],
            [{"LOWER": "membros"}, {"LOWER": "da"}, {"LOWER": {"IN": ["direção", "direcao"]}}],
        ],
        item_anchor_phrases=["Corpos Gerentes", "Membros da Direção", "Membros da Direcao"],
    ),

     "Alteracoes": BodySection(
        canonical="Alteracoes",
        header_aliases=["Alterações", "Alteracoes", "Alterações:", "Alteracoes:"],
        item_title_patterns=[
            [{"LOWER": {"IN": ["alteração", "alteracao", "alterações", "alteracoes"]}}],
        ],
        item_anchor_phrases=["Alteração", "Alterações", "Alteracoes"],
    ),

    "Estatutos/Alteracoes": BodySection(
        canonical="Estatutos/Alteracoes",
        header_aliases=[
            "Estatutos/Alterações", "Estatutos/Alteracoes",
            "Estatutos / Alterações", "Estatutos / Alteracoes",
            "Estatutos/Alterações:", "Estatutos / Alterações:",
        ],
        item_title_patterns=[
            [{"LOWER": "estatutos"}],
            [{"LOWER": {"IN": ["alteração", "alteracao", "alterações", "alteracoes"]}}],
        ],
        item_anchor_phrases=["Estatutos", "Alteração", "Alterações", "Alteracoes"],
    ),
    "CorposGerentes/Alteracoes": BodySection(
        canonical="CorposGerentes/Alteracoes",
        header_aliases=[
            "Corpos Gerentes/Alterações", "Corpos Gerentes/Alteracoes",
            "Corpos Gerentes / Alterações", "Corpos Gerentes / Alteracoes",
            "Corpos Gerentes/Alterações:", "Corpos Gerentes/Alteracoes:",
            "Corpos Gerentes / Alterações:", "Corpos Gerentes / Alteracoes:",
        ],
        item_title_patterns=[
            # "Corpos Gerentes"
            [{"LOWER": "corpos"}, {"LOWER": "gerentes"}],
            # "Membros da Direção/Direcao"
            [{"LOWER": "membros"}, {"LOWER": "da"}, {"LOWER": {"IN": ["direção", "direcao"]}}],
            # "Alteração/Alterações"
            [{"LOWER": {"IN": ["alteração", "alteracao", "alterações", "alteracoes"]}}],
        ],
        item_anchor_phrases=[
            "Corpos Gerentes", 
             "Alterações", "Alteracoes",],
),

}

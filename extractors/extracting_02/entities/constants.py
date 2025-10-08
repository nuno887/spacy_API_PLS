import re

HEADER_STARTERS = {
    "SECRETARIA", "SECRETARIAS", "VICE-PRESIDÊNCIA", "VICE-PRESIDENCIA",
    "PRESIDÊNCIA", "PRESIDENCIA", "DIREÇÃO", "DIRECÇÃO",
    "ASSEMBLEIA", "CÂMARA", "CAMARA", "MUNICIPIO",
    "TRIBUNAL", "CONSERVATÓRIA", "CONSERVATORIA",
    "ADMINISTRAÇÃO"
}

DOT_LEADER_LINE_RE = re.compile(r'^\s*\.{5,}\s*$')
DOT_LEADER_TAIL_RE = re.compile(r'\.{5,}\s*$')
BLANK_RE = re.compile(r'^\s*$')

ITEM_STARTERS = ("portaria", "aviso", "acordo", "contrato", "cct", "cctv", "regulamento", "despacho")

SUMARIO_PAT = re.compile(r'\bS[UÚ]M[ÁA]RIO\b', re.IGNORECASE)

# main.py
import sys
import json
from entities import nlp, parse_sumario_and_body_bundle

def run_pipeline(text_raw: str):
    """
    One-call entry point.
    Returns: (payload_dict, sumario_text, body_text, text_raw)
    """
    payload, sumario_text, body_text, full_text = parse_sumario_and_body_bundle(text_raw, nlp)
    return payload, sumario_text, body_text, full_text

def main():
    # 1) Load input (file path or inline fallback)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            text_raw = f.read()
    else:
        text_raw = (
            "ADMINISTRAÇÃO PÚBLICA REGIONAL - RELAÇÕES COLETIVAS\n"
            "DE TRABALHO\n"
            "Acordo Coletivo n.º 9/2014 - Acordo Coletivo de Entidade Empregadora Pública\n"
            "celebrado entre a Assembleia Legislativa da Madeira, o Sindicato dos Trabalhadores\n"
            "da Função Pública da Região Autónoma da Madeira e o Sindicato dos Trabalhadores\n"
            "Administração Pública e de Entidades com Fins Públicos. .........................................\n"
            "Acordo Coletivo n.º 10/2014 - Acordo Coletivo de Empregador Público celebrado\n"
            "entre a Secretaria dos Assuntos Sociais - SRAS, a Secretaria Regional do Plano e\n"
            "Finanças - SRPF, a Vice-Presidência do Governo da Região Autónoma da Madeira -\n"
            "VP, o Serviço de Saúde da Região Autónoma da Madeira, E.P.E. - SESARAM, a\n"
            "Federação dos Sindicatos da Administração Pública - FESAP, o Sindicato dos\n"
            "Trabalhadores da Função Pública da Região Autónoma da Madeira - STFP, RAM e\n"
            "o Sindicato Nacional dos Técncos Superiores de Saúde das Áreas de Diagnóstico e\n"
            "Terapêutica - SNTSSDT. ..............................................................................................\n"
            "SECRETARIA REGIONAL DA EDUCAÇÃO E RECURSOS HUMANOS\n"
            "Direção Regional do Trabalho\n"
            "Regulamentação do Trabalho\n"
            "Despachos:\n"
            "“Capio - Consultoria e Comércio, Lda” - Autorização para Adoção de Período de\n"
            "Laboração com Amplitude Superior aos Limites Normais. .........................................\n"
            "Portarias de Condições de Trabalho:\n"
            "Portarias de Extensão:\n"
            "Aviso de Projeto de Portaria de Extensão do Acordo de Empresa celebrado entre o\n"
            "Serviço de Saúde da Região Autónoma da Madeira, E.P.E. - SESARAM, a Federação\n"
            "dos Sindicatos da Administração Pública - FESAP, o Sindicato dos Trabalhadores da\n"
            "Função Pública da Região Autónoma da Madeira - STFP, RAM e o Sindicato Nacional\n"
            "dos Técnicos Superiores de Saúde das Áreas de Diagnóstico e Terapêutica - SNTSSDT. ......................\n"
            "Organizações do Trabalho:\n"
            "Associações Sindicais:\n"
            "Estatutos:\n"
            "Sindicato Democrático dos Professores da Madeira - Alteração. ...............................\n"
            "Associações de Empregadores:\n"
            "Alterações:\n"
            "Associação Comercial e Industrial do Funchal - Câmara de Comércio e Indústria da\n"
            "Madeira - Alteração. ..............................................................................................\n"

            "ADMINISTRAÇÃO PÚBLICA REGIONAL - RELAÇÕES COLETIVAS\n"
            "DE TRABALHO\n"
        )
        text_raw_01 = """
SECRETARIAREGIONAL DOS RECURSOS HUMANOS
Direcção Regional do Trabalho
Regulamentação do Trabalho
Despachos:
"Teixeira Duarte - Engenharia e Construções, S.A.", - Autorização para adopção de 
período de laboração com amplitude superior aos limites normais..................................
Regulamentos de Condições Mínimas:
Portaria que Aprova o Regulamento de Condições Mínimas para o sector da Indústria
Hoteleira da Região Autónoma da Madeira.....................................................................
Regulamentos de Extensão:
Portaria n.º 22/RE/2008 - Aprova o Regulamento de Extensão do CCT entre a ANF -
Associação Nacional das Farmácias e o SINPROFARM - Sindicato Nacional dos
Profissionais de Farmácia - Alteração Salarial e Outras.................................................
Portaria n.º 23/RE/2008 - Aprova o Regulamento de Extensão do ACTentre a Empresa
de Navegação Madeirense, Ld.ª, e Outras e a FESMAR - Federação de Sindicatos dos
Trabalhadores do Mar - Alteração Salarial e Outras........................................................
Aviso de Projecto de Portaria que Aprova o Regulamento de Extensão do Contrato
Colectivo de Trabalho Vertical entre a ACIF - CCIM - Associação Comercial e
Industrial do Funchal - Câmara de Comércio e Indústria da Madeira e o SITAM -
Sindicato dos Trabalhadores de Escritório, Comércio e Serviços da R.A.M. - Para o
Sector de Armazenamento, Engarrafamento, Comércio por grosso e Exportação do
Vinho da Madeira na Região Autónoma da Madeira - Revisão Salarial.........................
Aviso de Projecto de Portaria que Aprova o Regulamento de Extensão do Contrato
Colectivo de Trabalho entre a Associação Comercial e Industrial do Funchal - Câmara
de Comércio e Indústria da Madeira e o Sindicato das Indústrias Eléctricas do Sul e
Ilhas - Revisão Salarial..................................................................................................
Aviso de Projecto de Portaria que Aprova o Regulamento de Extensão do CCT entre a
AEEP - Associação dos Estabelecimentos de Ensino Particular e Cooperativo e a
FENPROF - Federação Nacional dos Professores e Outros - Alteração Salarial e
Outras.................................................................................................................................
Aviso de Projecto de Portaria que Aprova o Regulamento de Extensão do Contrato
Colectivo de Trabalho entre a ATMARAM - Associação de Transportes de Mercadorias
em Aluguer da Região Autónoma da Madeira e o Sindicato dos Trabalhadores de
Transportes Rodoviários da Região Autónoma da Madeira - Tabelas Salariais e
Outras.................................................................................................................................
Convenções Colectivas de Trabalho:
Contrato Colectivo de Trabalho Vertical entre a ACIF - CCIM - Associação Comercial
e Industrial do Funchal - Câmara de Comércio e Indústria da Madeira e o SITAM -
Sindicato dos Trabalhadores de Escritório, Comércio e Serviços da R.A.M. - Para o
Sector de Armazenamento, Engarrafamento, Comércio por grosso e Exportação do
Vinho da Madeira na Região Autónoma da Madeira - Revisão
Salarial...............................................................................................................................
Contrato Colectivo de Trabalho entre a Associação Comercial e Industrial do Funchal -
Câmara de Comércio e Indústria da Madeira e o Sindicato das Indústrias Eléctricas do
Sul e Ilhas - Revisão Salarial.............................................................................................. 
C C T entre a A E E P - Associação dos Estabelecimentos de Ensino Particular e
Cooperativo e a FENPROF - Federação Nacional dos Professores e Outros - Alteração
Salarial e Outras.................................................................................................................
CCTentre a APEB - Associação Portuguesa das Empresas de Betão Pronto e a FETESE
- Federação dos Sindicatos dos Trabalhadores de Serviços e outros (revisão global) -
Rectificação.......................................................................................................................
Contrato Colectivo de Trabalho entre a ATMARAM - Associação de Transportes de
Mercadorias em Aluguer da Região Autónoma da Madeira e o Sindicato dos
Trabalhadores de Transportes Rodoviários da Região Autónoma da Madeira -Tabelas
Salariais e Outras. ..............................................................................................................

SECRETARIAREGIONAL DOS RECURSOS HUMANOS

"""
            


        

    # 2) Run the full pipeline
    payload, sumario_text, body_text, full_text = run_pipeline(text_raw)

    # 3) Concise console summary
    sum_span = payload["sumario"]["span"]
    body_span = payload["body"]["span"]
    print("\n=== SPLIT ===")
    print(f"Sumário: {sum_span['start']}..{sum_span['end']} | len={sum_span['end']-sum_span['start']}")
    print(f"Body   : {body_span['start']}..{body_span['end']} | len={body_span['end']-body_span['start']}")
    print(f"Strategy: {payload['diagnostics']['strategy']}")

    # Sections & items
    print("\n=== SUMÁRIO SECTIONS ===")
    for s in payload["sumario"]["sections"]:
        path = " > ".join(s["path"])
        print(f"- {path}  @ {s['span']['start']}..{s['span']['end']}")
        for it in s["items"]:
            print(f"    • {it['text']}  @ {it['span']['start']}..{it['span']['end']}")

    # Section → Item relations
    print("\n=== SUMÁRIO RELATIONS (Section → Item) ===")
    for r in payload["sumario"]["relations_section_item"]:
        print(f"{' > '.join(r['section_path'])}  ::  {r['item_text']}")

    # (Optional) Section ranges
    print("\n=== SUMÁRIO SECTION RANGES ===")
    for sr in payload["sumario"]["section_ranges"]:
        print(f"{' > '.join(sr['section_path'])}  ::  content {sr['content_range']['start']}..{sr['content_range']['end']}")

    # ORG → ORG relations
    print("\n=== ORG → ORG RELATIONS ===")
    for r in payload["relations_org_to_org"]:
        print(f"- {r['key']}")
        print(f"  sumário: '{r['sumario']['surface_raw']}' @{r['sumario']['span']['start']}..{r['sumario']['span']['end']}")
        print(f"  body   : '{r['body']['surface_raw']}' @{r['body']['span']['start']}..{r['body']['span']['end']}")
        print(f"  conf   : {r['confidence']}")

    # Diagnostics
    diag = payload["diagnostics"]
    if diag.get("unmatched_sumario_orgs") or diag.get("unmatched_body_orgs") or diag.get("split_anchor"):
        print("\n=== DIAGNOSTICS ===")
        if diag.get("split_anchor"):
            print(f"Split anchor: {diag['split_anchor']}")
        if diag.get("unmatched_sumario_orgs"):
            print("Unmatched Sumário ORGs:")
            for h in diag["unmatched_sumario_orgs"]:
                print(f"  - '{h['surface_raw']}' @{h['span']['start']}..{h['span']['end']} | key={h['canonical_key']}")
        if diag.get("unmatched_body_orgs"):
            print("Unmatched Body ORGs:")
            for h in diag["unmatched_body_orgs"]:
                print(f"  - '{h['surface_raw']}' @{h['span']['start']}..{h['span']['end']} | key={h['canonical_key']}")

    # 4) Optional: write payload JSON for inspection
    out_path = "sumario_body_payload.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved payload to {out_path}")

    #print(f"Sumario: {sumario_text}")
    
    #print(f"Body: {body_text}")

    
    #print(f"Full Text: {full_text}")




if __name__ == "__main__":
    main()

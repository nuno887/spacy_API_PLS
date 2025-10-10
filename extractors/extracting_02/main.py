import spacy
from entities import parse, parse_sumario_and_body_bundle, print_results, print_payload_summary

from entities import parse_sumario_and_body_bundle
from body_extraction import run_extraction
from body_extraction.debug_print import print_report  # optional

nlp = spacy.load("pt_core_news_lg", disable=["ner", "tagger", "parser", "lemmatizer"])
nlp.add_pipe("sentencizer", first=True, config={"punct_chars": [".", "!", "?", ";", ":"]})
if __name__ == "__main__":
    with open("sample_input_01.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # If you want just the sumário parse:
    doc, sections_tree = parse(text, nlp)
    print_results(doc, sections_tree)

    # Or the full bundle (sumário/body split + linking):
    payload, sumario_text, body_text, _ = parse_sumario_and_body_bundle(text, nlp)
    print_payload_summary(payload, save_path="sumario_body_payload.json")

    sections = payload["sumario"]["sections"]
    rels = payload.get("relations_org_to_org", [])

    print(f"Body_text:<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< {body_text}")

    report = run_extraction(body_text, sections, rels, nlp)
    print_report(report, body_text, show_full=True)

    # print(payload)  # or handle as needed

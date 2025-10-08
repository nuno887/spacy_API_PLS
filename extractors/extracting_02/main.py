import spacy
from entities import parse, parse_sumario_and_body_bundle, print_results

nlp = spacy.load("pt_core_news_lg", disable=["ner", "tagger", "parser", "lemmatizer"])

if __name__ == "__main__":
    with open("sample_input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # If you want just the sumário parse:
    doc, sections_tree = parse(text, nlp)
    print_results(doc, sections_tree)

    # Or the full bundle (sumário/body split + linking):
    payload, sumario_text, body_text, _ = parse_sumario_and_body_bundle(text, nlp)
    # print(payload)  # or handle as needed

import spacy
from entities import parse, parse_sumario_and_body_bundle
from entities.debug_print import print_results, print_payload_summary
from body_extraction import run_extraction
from body_extraction.debug_print import print_report  # optional


def build_nlp():
    # Load a lightweight PT pipeline without heavy components
    nlp = spacy.load("pt_core_news_lg", disable=["ner", "tagger", "parser", "lemmatizer"])
    # Ensure sentence boundaries exist (needed by expand_to_sentence)
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer", first=True, config={"punct_chars": [".", "!", "?", ";", ":"]})
    return nlp


if __name__ == "__main__":
    with open("sample_input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    nlp = build_nlp()
    # nlp.max_length = max(nlp.max_length, len(text) + 1000)  # optional safety for long docs

    # -------- Split first (Sumário / Body) --------
    payload, sumario_text, body_text, _ = parse_sumario_and_body_bundle(text, nlp)

    # -------- Debug: sections only on Sumário slice (print absolute offsets via 'offset') --------
    sum_start = payload["sumario"]["span"]["start"]
    sum_doc, sum_sections = parse(sumario_text, nlp)
    print_results(sum_doc, sum_sections, offset=sum_start)

    # -------- Bundle summary (Sumário + Body + relations) --------
    print_payload_summary(payload, save_path="sumario_body_payload.json")

    # -------- Body extraction / alignment --------
    sections = payload["sumario"]["sections"]
    rels = payload.get("relations_org_to_org", [])

    # Important: create body_doc via nlp(...) so sentencizer runs
    report = run_extraction(body_text, sections, rels, nlp)
    print_report(report, body_text, show_full=True)

    # (Optional) Avoid dumping full body in console:
    # print(f"Body preview: {body_text[:400]!r} ...")

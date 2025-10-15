# pdf_ocr_extractor/main.py
from pathlib import Path

# Support both: `python -m pdf_ocr_extractor.main` and `python pdf_ocr_extractor/main.py`
try:
    # When run as a module
    from . import load_settings, apply_tesseract_env, process_many
except ImportError:
    # When run directly as a script
    import sys
    pkg_dir = Path(__file__).parent
    sys.path.insert(0, str(pkg_dir.parent))  # add project root to sys.path
    from pdf_ocr_extractor import load_settings, apply_tesseract_env, process_many

def main():
    pkg_dir = Path(__file__).parent
    in_dir = (pkg_dir / "input").resolve()
    out_dir = (pkg_dir / "output").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_settings()          # uses pdf_ocr_extractor/appsettings.json
    cfg["InputDir"] = str(in_dir)  # force the input dir we want
    apply_tesseract_env(cfg)

    pdfs = sorted(in_dir.glob("*.pdf"))
    results = process_many(pdfs, cfg)

    # write only .txt files (no logs)
    for v in results:
        (out_dir / v.filename).write_bytes(v.content)

    print(f"Processed {len(results)} file(s) from {in_dir}")
    for v in results:
        print(f" - {v.filename}: {len(v.content or b'')} bytes")

if __name__ == "__main__":
    main()

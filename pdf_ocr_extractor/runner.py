from pathlib import Path
from typing import List, Optional

from .models import VirtualTextFile
from .config import load_settings
from .ocr_engine import apply_tesseract_env
from .processing import process_many

def run(cfg: Optional[dict] = None, config_path: Optional[str] = None) -> List[VirtualTextFile]:
    cfg = cfg or load_settings(config_path)
    apply_tesseract_env(cfg)

    in_dir = Path(cfg["InputDir"]).resolve()
    if cfg.get("PdfFilename"):
        pdfs = [in_dir / cfg["PdfFilename"]]
    else:
        pdfs = sorted(in_dir.glob("*.pdf"))

    return process_many(pdfs, cfg)

def run_input_dir_once(config_path: Optional[str] = None) -> List[VirtualTextFile]:
    cfg = load_settings(config_path)
    pkg_input = (Path(__file__).parent / "input").resolve()
    cfg["InputDir"] = str(pkg_input)
    apply_tesseract_env(cfg)

    pdfs = sorted(pkg_input.glob("*.pdf"))
    if not pdfs:
        from .logutil import log_info
        v = VirtualTextFile(filename="(no-files).txt", content=b"", text="")
        log_info(v.log, "(runner)", f"No PDFs found in: {pkg_input}")
        return [v]

    return process_many(pdfs, cfg)

def _print_raw(v: VirtualTextFile) -> None:
    import sys
    txt = v.text or ""
    if txt and not txt.endswith("\n"):
        txt += "\n"
    sys.stdout.write(txt)

if __name__ == "__main__":
    import sys

    cfg = load_settings()
    apply_tesseract_env(cfg)

    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
        from .processing import process_pdf_file
        v = process_pdf_file(pdf_path, cfg)
        _print_raw(v)
    else:
        pkg_input = (Path(__file__).parent / "input").resolve()
        cfg["InputDir"] = str(pkg_input)
        pdfs = sorted(pkg_input.glob("*.pdf"))
        results = process_many(pdfs, cfg)
        for v in results:  # fixed slice bug from original
            _print_raw(v)

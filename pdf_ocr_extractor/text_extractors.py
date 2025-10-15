import fitz
from .ocr_engine import extract_text_ocr

def extract_text_digital(page: fitz.Page, clip: fitz.Rect) -> str:
    txt = page.get_text("text", clip=clip) or ""
    return txt.strip()

def extract_text_with_strategy(
    page: fitz.Page,
    page_index: int,
    clip: fitz.Rect,
    cfg: dict,
    should_force_ocr: bool
) -> str:
    # prefer digital text, fall back to OCR
    if should_force_ocr:
        return extract_text_ocr(
            page,
            clip,
            int(cfg["Dpi"]),
            str(cfg["OcrLang"]),
            int(cfg["OcrTimeoutSec"]) if cfg.get("OcrTimeoutSec") is not None else None,
        )

    text = extract_text_digital(page, clip)
    if len(text) >= 3:
        return text

    return extract_text_ocr(
        page,
        clip,
        int(cfg["Dpi"]),
        str(cfg["OcrLang"]),
        int(cfg["OcrTimeoutSec"]) if cfg.get("OcrTimeoutSec") is not None else None,
    )

import io
import os
import platform
from typing import Optional
import fitz
from PIL import Image
import pytesseract

def apply_tesseract_env(cfg: dict) -> None:
    tcmd = cfg.get("TesseractCmd") or os.environ.get("TESSERACT_CMD")
    tessdata = cfg.get("TessdataPrefix") or os.environ.get("TESSDATA_PREFIX")

    if platform.system() == "Windows" and tcmd:
        pytesseract.pytesseract.tesseract_cmd = tcmd

    if tessdata:
        os.environ["TESSDATA_PREFIX"] = tessdata

def extract_text_ocr(page: fitz.Page, clip: fitz.Rect, dpi: int, lang: str, timeout_sec: Optional[int]) -> str:
    pix = page.get_pixmap(dpi=dpi, clip=clip, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    if timeout_sec and timeout_sec > 0:
        return pytesseract.image_to_string(img, lang=lang, timeout=timeout_sec).strip()
    return pytesseract.image_to_string(img, lang=lang).strip()

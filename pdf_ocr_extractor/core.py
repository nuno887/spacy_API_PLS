# core.py (virtual-only)
import io, os, json, datetime, shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import platform


# ---------------------------
# Virtual file model
# ---------------------------
@dataclass
class VirtualTextFile:
    filename: str                                # e.g., "mydoc.txt"
    content: bytes                               # UTF-8 encoded text
    text: str                                    # convenience (decoded)
    mimetype: str = "text/plain; charset=utf-8"
    log: List[str] = field(default_factory=list) # in-memory log lines


# ---------------------------
# Config
# ---------------------------
def load_settings(config_path: Optional[str] = None) -> dict:
    """
    Load settings from JSON. If config_path is None, read appsettings.json
    next to this file.
    """
    cfg_path = Path(config_path) if config_path else (Path(__file__).parent / "appsettings.json")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    defaults = {
        "InputDir": "input",
        "PdfFilename": None,
        "IgnoreTopPercent": 0.10,
        "SkipLastPage": True,
        "Dpi": 600,
        "OcrLang": "por",
        "TesseractCmd": None,
        "TessdataPrefix": None,
        # Safety / behavior
        "ContinueOnError": True,
        "FailIfEmptyOutput": False,
        "MaxPdfSizeMB": None,              # None = unlimited
        "CheckPdfSignature": True,
        "OcrRetryLowerDpi": True,
        "OcrLowerDpi": 300,
        "OcrTimeoutSec": 40,
        "MaxPageErrorsBeforeQuarantine": 3,
        "QuarantineOnOpenFail": True,      # ignored in virtual mode (no copies)
        # The following are ignored in virtual mode (no file outputs):
        "ExportPerPagePdfs": False,
        "ExportPerPageTxts": False,
        "OutputDir": "output",             # kept for compat; never used
    }
    return {**defaults, **(data or {})}


# ---------------------------
# In-memory logging
# ---------------------------
def _now() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%fZ")

def log_error(buf: List[str], pdf_name: str, msg: str, page: Optional[int] = None, exc: Optional[Exception] = None):
    pfx = f"[{_now()}] [{pdf_name}]"
    if page is not None:
        pfx += f" [page {page}]"
    if exc:
        buf.append(f"{pfx} ERROR: {msg} :: {repr(exc)}")
    else:
        buf.append(f"{pfx} ERROR: {msg}")

def log_info(buf: List[str], pdf_name: str, msg: str):
    buf.append(f"[{_now()}] [{pdf_name}] INFO: {msg}")


# ---------------------------
# Extraction helpers
# ---------------------------
def page_clip_rect(page: fitz.Page, page_index: int, ignore_top_fraction: float) -> fitz.Rect:
    rect = page.rect
    if page_index == 0 or ignore_top_fraction <= 0:
        return rect
    top_cut = rect.height * ignore_top_fraction
    return fitz.Rect(rect.x0, rect.y0 + top_cut, rect.x1, rect.y1)

def extract_text_digital(page: fitz.Page, clip: fitz.Rect) -> str:
    txt = page.get_text("text", clip=clip) or ""
    return txt.strip()

def extract_text_ocr(page: fitz.Page, clip: fitz.Rect, dpi: int, lang: str, timeout_sec: Optional[int]) -> str:
    # Render region to image, run Tesseract with optional timeout
    pix = page.get_pixmap(dpi=dpi, clip=clip, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    if timeout_sec and timeout_sec > 0:
        return pytesseract.image_to_string(img, lang=lang, timeout=timeout_sec).strip()
    return pytesseract.image_to_string(img, lang=lang).strip()

def should_ocr_page2(page: fitz.Page, clip: fitz.Rect) -> bool:
    blocks = page.get_text("blocks", clip=clip) or []
    if not blocks:
        return False
    blocks.sort(key=lambda b: (b[1], b[0]))
    page_w = (clip.x1 - clip.x0) if clip else page.rect.width
    page_h = (clip.y1 - clip.y0) if clip else page.rect.height
    mid_x = ((clip.x0 + clip.x1) / 2) if clip else ((page.rect.x0 + page.rect.x1) / 2)
    zone_y1 = (clip.y0 if clip else page.rect.y0) + 0.15 * page_h

    def is_narrow(b): return (b[2] - b[0]) <= 0.60 * page_w
    top_blocks = [b for b in blocks if b[1] <= zone_y1]
    body_blocks = [b for b in blocks if b[1] > zone_y1]
    starts_one_col = any(is_narrow(b) for b in top_blocks)

    def cx(b): return (b[0] + b[2]) / 2
    left = [b for b in body_blocks if cx(b) < mid_x]
    right = [b for b in body_blocks if cx(b) >= mid_x]
    has_two_cols_later = (len(left) >= 1 and len(right) >= 1)
    return starts_one_col and has_two_cols_later


# ---------------------------
# Core processing (virtual only)
# ---------------------------
def process_pdf_file(pdf_path: Path, cfg: dict) -> VirtualTextFile:
    """
    Process ONE PDF and return a VirtualTextFile.
    No filesystem writes (outputs, per-page artifacts, or quarantine).
    """
    pdf_name = pdf_path.name
    vlog: List[str] = []

    # Basic file validations
    try:
        size = pdf_path.stat().st_size
        max_mb = cfg.get("MaxPdfSizeMB")
        if max_mb is not None:
            max_bytes = int(max_mb) * 1024 * 1024
            if size > max_bytes:
                log_error(vlog, pdf_name, f"File too large ({size} bytes) > MaxPdfSizeMB={max_mb}")
                # return empty virtual file to keep contract (or raise)
                return VirtualTextFile(filename=f"{pdf_path.stem}.txt", content=b"", text="", log=vlog)

        if cfg.get("CheckPdfSignature", True):
            with pdf_path.open("rb") as f:
                sig = f.read(5)
            if sig != b"%PDF-":
                log_error(vlog, pdf_name, "Missing %PDF- signature; likely not a PDF or corrupted")
                return VirtualTextFile(filename=f"{pdf_path.stem}.txt", content=b"", text="", log=vlog)
    except Exception as e:
        log_error(vlog, pdf_name, "Failed basic file validation", exc=e)
        return VirtualTextFile(filename=f"{pdf_path.stem}.txt", content=b"", text="", log=vlog)

    # Open the PDF defensively
    try:
        doc = fitz.open(pdf_path.as_posix())
    except Exception as e:
        log_error(vlog, pdf_name, "Failed to open PDF", exc=e)
        # Quarantine ignored in virtual mode; just return empty
        return VirtualTextFile(filename=f"{pdf_path.stem}.txt", content=b"", text="", log=vlog)

    # Encrypted?
    try:
        if getattr(doc, "needs_pass", False):
            log_error(vlog, pdf_name, "Encrypted PDF (password required). Skipping.")
            doc.close()
            return VirtualTextFile(filename=f"{pdf_path.stem}.txt", content=b"", text="", log=vlog)
    except Exception:
        pass

    page_errors = 0
    all_text: List[str] = []

    try:
        page_count = doc.page_count
        last_index = page_count - 1
        effective_last = last_index - 1 if (cfg["SkipLastPage"] and page_count >= 1) else last_index
        if effective_last < 0:
            doc.close()
            text = ""
            if not text and bool(cfg.get("FailIfEmptyOutput", False)):
                raise RuntimeError(f"No text extracted from {pdf_name} (empty output).")
            return VirtualTextFile(filename=f"{pdf_path.stem}.txt", content=text.encode("utf-8"), text=text, log=vlog)

        for i in range(0, effective_last + 1):
            try:
                page = doc[i]
                clip = page_clip_rect(page, i, float(cfg["IgnoreTopPercent"]))

                # Extraction strategy
                text = ""
                try:
                    if i == 1 and should_ocr_page2(page, clip):
                        text = extract_text_ocr(page, clip, int(cfg["Dpi"]), str(cfg["OcrLang"]), int(cfg["OcrTimeoutSec"]))
                    else:
                        text = extract_text_digital(page, clip)
                        if len(text) < 3:
                            text = extract_text_ocr(page, clip, int(cfg["Dpi"]), str(cfg["OcrLang"]), int(cfg["OcrTimeoutSec"]))
                except Exception as e_ocr:
                    # Retry OCR once at lower DPI if configured
                    if cfg.get("OcrRetryLowerDpi", True):
                        try:
                            text = extract_text_ocr(page, clip, int(cfg["OcrLowerDpi"]), str(cfg["OcrLang"]), int(cfg["OcrTimeoutSec"]))
                            log_info(vlog, pdf_name, f"OCR retried at lower DPI on page {i+1}")
                        except Exception as e_retry:
                            log_error(vlog, pdf_name, "OCR failed at both DPIs", page=i+1, exc=e_retry)
                            page_errors += 1
                            text = f"[ERROR: OCR failed on page {i+1}]"
                    else:
                        log_error(vlog, pdf_name, "OCR failed", page=i+1, exc=e_ocr)
                        page_errors += 1
                        text = f"[ERROR: OCR failed on page {i+1}]"

                all_text.append(text)

                # Stop early if too many page errors
                if page_errors >= int(cfg["MaxPageErrorsBeforeQuarantine"]):
                    log_error(vlog, pdf_name, f"Exceeded MaxPageErrorsBeforeQuarantine={cfg['MaxPageErrorsBeforeQuarantine']}; stopping file.")
                    break

            except Exception as e_page:
                page_errors += 1
                log_error(vlog, pdf_name, "Unhandled page-level error", page=i+1, exc=e_page)
                all_text.append(f"[ERROR: Exception on page {i+1}]")
                if page_errors >= int(cfg["MaxPageErrorsBeforeQuarantine"]):
                    log_error(vlog, pdf_name, "Too many page errors; aborting this file.")
                    break

    finally:
        try:
            doc.close()
        except Exception:
            pass

    # Build virtual file (no disk writes)
    text = "\n\n".join(all_text).strip()
    if not text and bool(cfg.get("FailIfEmptyOutput", False)):
        raise RuntimeError(f"No text extracted from {pdf_name} (empty output).")

    return VirtualTextFile(
        filename=f"{pdf_path.stem}.txt",
        content=text.encode("utf-8"),
        text=text,
        log=vlog,
    )


def _apply_tesseract_env(cfg: dict) -> None:
    # Optional overrides from JSON or env
    tcmd = cfg.get("TesseractCmd") or os.environ.get("TESSERACT_CMD")
    tessdata = cfg.get("TessdataPrefix") or os.environ.get("TESSDATA_PREFIX")

    # Only force an explicit path on Windows. On Linux/mac, rely on system path.
    if platform.system() == "Windows" and tcmd:
        pytesseract.pytesseract.tesseract_cmd = tcmd

    if tessdata:
        os.environ["TESSDATA_PREFIX"] = tessdata


# ---------------------------
# Library-friendly runners (virtual only)
# ---------------------------
def process_many(pdfs: List[Path], cfg: dict) -> List[VirtualTextFile]:
    """
    Process a list of PDFs and return VirtualTextFile objects for each.
    """
    _apply_tesseract_env(cfg)
    results: List[VirtualTextFile] = []
    for p in pdfs:
        if not (p.exists() and p.suffix.lower() == ".pdf"):
            # make a minimal "virtual" with error log
            v = VirtualTextFile(filename=f"{p.stem or 'unknown'}.txt", content=b"", text="")
            log_error(v.log, p.name or "unknown.pdf", "Skipping: not a PDF or does not exist")
            results.append(v)
            continue
        try:
            results.append(process_pdf_file(p, cfg))
        except Exception as e:
            v = VirtualTextFile(filename=f"{p.stem}.txt", content=b"", text="")
            log_error(v.log, p.name, "Top-level error while processing file", exc=e)
            results.append(v)
            if not bool(cfg.get("ContinueOnError", True)):
                # stop early and bubble up
                raise
    return results


def run(cfg: Optional[dict] = None, config_path: Optional[str] = None) -> List[VirtualTextFile]:
    """
    Convenience entry for library use (virtual-only).
    - If cfg is None, load from config_path (or module's appsettings.json).
    - Returns a list of VirtualTextFile.
    """
    cfg = cfg or load_settings(config_path)
    _apply_tesseract_env(cfg)

    in_dir = Path(cfg["InputDir"]).resolve()

    if cfg.get("PdfFilename"):
        pdfs = [in_dir / cfg["PdfFilename"]]
    else:
        pdfs = sorted(in_dir.glob("*.pdf"))

    return process_many(pdfs, cfg)


# ---------------------------
# (Optional) CLI – prints filenames + sizes; no disk writes
# ---------------------------
if __name__ == "__main__":
    cfg = load_settings()
    results = run(cfg)
    for v in results:
        print(f"{v.filename}  ({len(v.content)} bytes)  logs={len(v.log)}")

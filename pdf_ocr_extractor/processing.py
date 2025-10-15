from pathlib import Path
from typing import List, Optional

import fitz

from .models import VirtualTextFile
from .logutil import log_error, log_info
from .text_regions import page_clip_rect, should_ocr_page2
from .text_extractors import extract_text_with_strategy
from .ocr_engine import extract_text_ocr

def _validate_pdf_header_and_size(pdf_path: Path, cfg: dict, vlog: List[str]) -> bool:
    pdf_name = pdf_path.name
    try:
        size = pdf_path.stat().st_size
        max_mb = cfg.get("MaxPdfSizeMB")
        if max_mb is not None:
            max_bytes = int(max_mb) * 1024 * 1024
            if size > max_bytes:
                log_error(vlog, pdf_name, f"File too large ({size} bytes) > MaxPdfSizeMB={max_mb}")
                return False

        if cfg.get("CheckPdfSignature", True):
            with pdf_path.open("rb") as f:
                sig = f.read(5)
            if sig != b"%PDF-":
                log_error(vlog, pdf_name, "Missing %PDF- signature; likely not a PDF or corrupted")
                return False
    except Exception as e:
        log_error(vlog, pdf_name, "Failed basic file validation", exc=e)
        return False

    return True

def process_pdf_file(pdf_path: Path, cfg: dict) -> VirtualTextFile:
    pdf_name = pdf_path.name
    vlog: List[str] = []

    if not _validate_pdf_header_and_size(pdf_path, cfg, vlog):
        return VirtualTextFile(filename=f"{pdf_path.stem}.txt", content=b"", text="", log=vlog)

    try:
        doc = fitz.open(pdf_path.as_posix())
    except Exception as e:
        log_error(vlog, pdf_name, "Failed to open PDF", exc=e)
        return VirtualTextFile(filename=f"{pdf_path.stem}.txt", content=b"", text="", log=vlog)

    try:
        try:
            if getattr(doc, "needs_pass", False):
                log_error(vlog, pdf_name, "Encrypted PDF (password required). Skipping.")
                return VirtualTextFile(filename=f"{pdf_path.stem}.txt", content=b"", text="", log=vlog)
        except Exception:
            pass

        page_errors = 0
        all_text: List[str] = []

        page_count = doc.page_count
        last_index = page_count - 1
        effective_last = last_index - 1 if (cfg["SkipLastPage"] and page_count >= 1) else last_index

        if effective_last < 0:
            text = ""
            if not text and bool(cfg.get("FailIfEmptyOutput", False)):
                raise RuntimeError(f"No text extracted from {pdf_name} (empty output).")
            return VirtualTextFile(filename=f"{pdf_path.stem}.txt", content=text.encode("utf-8"), text=text, log=vlog)

        for i in range(0, effective_last + 1):
            try:
                page = doc[i]
                clip = page_clip_rect(page, i, float(cfg["IgnoreTopPercent"]))

                # Strategy: page 2 heuristic or digital-first
                text = ""
                try:
                    force_ocr = (i == 1 and should_ocr_page2(page, clip))
                    text = extract_text_with_strategy(page, i, clip, cfg, should_force_ocr=force_ocr)
                except Exception as e_ocr:
                    if cfg.get("OcrRetryLowerDpi", True):
                        try:
                            text = extract_text_ocr(
                                page,
                                clip,
                                int(cfg["OcrLowerDpi"]),
                                str(cfg["OcrLang"]),
                                int(cfg["OcrTimeoutSec"]) if cfg.get("OcrTimeoutSec") is not None else None,
                            )
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

        text = "\n\n".join(all_text).strip()
        if not text and bool(cfg.get("FailIfEmptyOutput", False)):
            raise RuntimeError(f"No text extracted from {pdf_name} (empty output).")

        return VirtualTextFile(
            filename=f"{pdf_path.stem}.txt",
            content=text.encode("utf-8"),
            text=text,
            log=vlog,
        )

    finally:
        try:
            doc.close()
        except Exception:
            pass

def process_many(pdfs: List[Path], cfg: dict) -> List[VirtualTextFile]:
    results: List[VirtualTextFile] = []
    for p in pdfs:
        if not (p.exists() and p.suffix.lower() == ".pdf"):
            v = VirtualTextFile(filename=f"{p.stem or 'unknown'}.txt", content=b"", text="")
            from .logutil import log_error
            log_error(v.log, p.name or "unknown.pdf", "Skipping: not a PDF or does not exist")
            results.append(v)
            continue
        try:
            results.append(process_pdf_file(p, cfg))
        except Exception as e:
            v = VirtualTextFile(filename=f"{p.stem}.txt", content=b"", text="")
            from .logutil import log_error
            log_error(v.log, p.name, "Top-level error while processing file", exc=e)
            results.append(v)
            if not bool(cfg.get("ContinueOnError", True)):
                raise
    return results

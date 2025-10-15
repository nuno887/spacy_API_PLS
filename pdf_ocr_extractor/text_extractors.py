# pdf_ocr_extractor/text_extractors.py
from typing import List, Tuple, Optional
import fitz
from .ocr_engine import extract_text_ocr

Word = Tuple[float, float, float, float, str, int, int, int]  # (x0,y0,x1,y1,text,block,line,word)

def _words_in_clip(page: fitz.Page, clip: fitz.Rect) -> List[Word]:
    words: List[Word] = page.get_text("words", clip=clip) or []
    words.sort(key=lambda w: (w[1], w[0]))  # top → left
    return words

def _group_into_bands(words: List[Word], y_merge_tol: float) -> List[List[Word]]:
    bands: List[List[Word]] = []
    current: List[Word] = []
    last_y: Optional[float] = None

    for w in words:
        y_top = w[1]
        if last_y is None or abs(y_top - last_y) <= y_merge_tol:
            current.append(w)
        else:
            bands.append(current)
            current = [w]
        last_y = y_top

    if current:
        bands.append(current)
    return bands

def _find_gutter_x(band: List[Word], x0: float, x1: float) -> Optional[float]:
    if len(band) < 8:
        return None

    mids = [(w[0] + w[2]) / 2.0 for w in band]
    width = x1 - x0
    bins = max(16, min(64, int(len(band) / 4)))
    step = width / bins if bins > 0 else 0
    if step <= 0:
        return None

    hist = [0] * bins
    for m in mids:
        idx = int((m - x0) / step)
        if 0 <= idx < bins:
            hist[idx] += 1

    # 3-bin smoothing
    sm = hist[:]
    for i in range(1, bins - 1):
        sm[i] = (hist[i - 1] + hist[i] + hist[i + 1]) / 3.0

    third = bins // 3
    left_peak = max(range(1, max(2, third)), key=lambda i: sm[i], default=None)
    right_peak = max(range(max(2, 2 * third), bins - 1), key=lambda i: sm[i], default=None)
    if left_peak is None or right_peak is None or right_peak - left_peak < 3:
        return None

    valley = min(range(left_peak + 1, right_peak), key=lambda i: sm[i], default=None)
    if valley is None:
        return None

    peak_level = max(sm[left_peak], sm[right_peak])
    if peak_level <= 0:
        return None

    # require a deep valley to trust the split
    if sm[valley] / peak_level > 0.45:
        return None

    split_x = x0 + valley * step
    # keep away from edges (10% margins)
    margin = (x1 - x0) * 0.10
    if split_x < x0 + margin or split_x > x1 - margin:
        return None

    return split_x

def _text_from_words(words: List[Word]) -> str:
    if not words:
        return ""
    # group by (block,line)
    lines_map = {}
    for w in words:
        key = (w[5], w[6])
        lines_map.setdefault(key, []).append(w)

    lines = list(lines_map.values())
    lines.sort(key=lambda ln: min(w[1] for w in ln))  # by top y

    out: List[str] = []
    last_bottom = None
    for ln in lines:
        ln.sort(key=lambda w: w[0])  # left→right
        y_top = min(w[1] for w in ln)
        y_bot = max(w[3] for w in ln)
        if last_bottom is not None and (y_top - last_bottom) > 10:
            out.append("")  # paragraph gap
        out.append(" ".join(w[4] for w in ln))
        last_bottom = y_bot

    return "\n".join(out).strip()

def extract_text_layout_aware(page: fitz.Page, clip: fitz.Rect) -> str:
    rect = clip if clip else page.rect
    words = _words_in_clip(page, rect)
    if not words:
        return ""

    heights = [w[3] - w[1] for w in words]
    med_h = sorted(heights)[len(heights) // 2] if heights else 12.0
    y_tol = max(6.0, min(16.0, 0.35 * med_h))  # adaptive band tolerance

    bands = _group_into_bands(words, y_merge_tol=y_tol)

    chunks: List[str] = []
    for band in bands:
        split_x = _find_gutter_x(band, rect.x0, rect.x1)
        if split_x is None:
            chunks.append(_text_from_words(band))
        else:
            left = [w for w in band if (w[0] + w[2]) / 2.0 < split_x]
            right = [w for w in band if (w[0] + w[2]) / 2.0 >= split_x]
            lt = _text_from_words(left)
            rt = _text_from_words(right)
            band_txt = "\n".join(t for t in (lt, rt) if t).strip()
            chunks.append(band_txt)

    return "\n\n".join([c for c in chunks if c]).strip()

def extract_text_digital(page: fitz.Page, clip: fitz.Rect) -> str:
    txt = (page.get_text("text", clip=clip) or "").strip()
    if len(txt) >= 10:
        return txt
    la = extract_text_layout_aware(page, clip)
    return la if len(la) > len(txt) else txt

def extract_text_with_strategy(
    page: fitz.Page,
    page_index: int,
    clip: fitz.Rect,
    cfg: dict,
    should_force_ocr: bool
) -> str:
    # Prefer digital. Only OCR if digital is empty.
    digital = extract_text_digital(page, clip)
    if digital and len(digital) >= 3:
        return digital

    if should_force_ocr or not digital:
        timeout = cfg.get("OcrTimeoutSec")
        return extract_text_ocr(
            page, clip, int(cfg["Dpi"]), str(cfg["OcrLang"]),
            int(timeout) if timeout is not None else None
        ).strip()

    return digital

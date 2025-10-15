import fitz  # PyMuPDF

def page_clip_rect(page: fitz.Page, page_index: int, ignore_top_fraction: float) -> fitz.Rect:
    rect = page.rect
    if page_index == 0 or ignore_top_fraction <= 0:
        return rect
    top_cut = rect.height * ignore_top_fraction
    return fitz.Rect(rect.x0, rect.y0 + top_cut, rect.x1, rect.y1)

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

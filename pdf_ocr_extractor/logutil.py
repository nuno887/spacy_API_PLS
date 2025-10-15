import datetime
from typing import List, Optional

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

import json
from pathlib import Path
from typing import Optional, Dict, Any

_DEFAULTS: Dict[str, Any] = {
    "InputDir": "input",
    "PdfFilename": None,
    "IgnoreTopPercent": 0.10,
    "SkipLastPage": True,
    "Dpi": 600,
    "OcrLang": "por",
    "TesseractCmd": None,
    "TessdataPrefix": None,
    "ContinueOnError": True,
    "FailIfEmptyOutput": False,
    "MaxPdfSizeMB": None,
    "CheckPdfSignature": True,
    "OcrRetryLowerDpi": True,
    "OcrLowerDpi": 300,
    "OcrTimeoutSec": 40,
    "MaxPageErrorsBeforeQuarantine": 3,
    "QuarantineOnOpenFail": True,
    "ExportPerPagePdfs": False,
    "ExportPerPageTxts": False,
    "OutputDir": "output",
}

def load_settings(config_path: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = Path(config_path) if config_path else (Path(__file__).parent / "appsettings.json")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f) or {}
    return {**_DEFAULTS, **data}

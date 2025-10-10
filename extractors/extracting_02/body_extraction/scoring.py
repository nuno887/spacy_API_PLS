# body_extraction/scoring.py
from typing import List, Dict, Any

def pick_best_candidate(cands: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Choose the best candidate by highest 'confidence'.
    Expected candidate keys: 'start', 'end', 'confidence', 'method' (optional), 'anchor_used' (optional).
    Returns {'method': 'none'} if there are no candidates.
    """
    if not cands:
        return {"method": "none", "confidence": 0.0}

    # normalize and guard
    for c in cands:
        if "confidence" not in c or c["confidence"] is None:
            c["confidence"] = 0.0
        if "method" not in c or not c["method"]:
            c["method"] = "anchor"

    best = max(cands, key=lambda x: float(x.get("confidence", 0.0)))

    # Ensure required fields exist
    return {
        "start": int(best.get("start", 0)),
        "end": int(best.get("end", 0)),
        "confidence": float(best.get("confidence", 0.0)),
        "method": best.get("method", "anchor"),
        "anchor_used": best.get("anchor_used"),
    }

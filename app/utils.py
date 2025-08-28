# app/utils.py
import re
import hashlib
from typing import Any

def md5(s: str) -> str:
    """Stable hash helper (used in chunk IDs)."""
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def truncate_to_words(text: str, max_words: int) -> str:
    """Truncate a string to a word limit with ellipsis like original behavior."""
    words = re.split(r"\s+", (text or "").strip())
    if len(words) <= max_words:
        return (text or "").strip()
    return " ".join(words[:max_words]).rstrip() + "..."

def decide_word_limit_from_query(query: str, length_choice: str) -> int:
    """
    Reproduces your original heuristic for answer-length selection.
    """
    q = (query or "").lower()
    if any(k in q for k in ("summary", "brief overview", "overview", "summarize", "short summary")):
        default = 120
    elif len(q.split()) > 30 or "explain" in q or "describe" in q:
        default = 100
    else:
        default = 60
    length_map = {"Short": 40, "Medium": 80, "Long": 200}
    return length_map.get(length_choice, min(default, 120))

def extract_video_id(url: str):
    """
    Extract a YouTube video ID from typical URL forms (incl. youtu.be/shorts).
    """
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:\W|$)",
        r"youtu\.be\/([0-9A-Za-z_-]{11})(?:\W|$)",
        r"shorts\/([0-9A-Za-z_-]{11})(?:\W|$)",
    ]
    for p in patterns:
        m = re.search(p, url or "")
        if m:
            return m.group(1)
    return None

def chunk_id_for(url: str, idx: int, content_sample: str) -> str:
    """
    Stable chunk identifier: url + chunk index + sample hash.
    Mirrors your original implementation.
    """
    sample_hash = md5((content_sample or "")[:200])
    return md5(f"{url}::chunk::{idx}::{sample_hash}")

def extract_text_from_invoke_result(res: Any) -> str:
    """
    Normalize LangChain chain.invoke() outputs into a plain string.
    Kept identical in spirit to your original for compatibility.
    """
    if res is None:
        return ""
    try:
        if isinstance(res, str):
            return res.strip()
        if isinstance(res, dict):
            for k in ("text", "output_text", "content", "answer", "response"):
                v = res.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            for v in res.values():
                if isinstance(v, str) and v.strip():
                    return v.strip()
        for attr in ("output_text", "text", "content"):
            if hasattr(res, attr):
                val = getattr(res, attr)
                if isinstance(val, str) and val.strip():
                    return val.strip()
    except Exception:
        pass
    try:
        return str(res)
    except Exception:
        return ""

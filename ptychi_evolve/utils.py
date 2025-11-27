"""Utility helpers used across ptychi_evolve."""

import json
import re
from typing import Any, Dict, Union


def response_text(source: Union[str, Any]) -> str:
    """Return text content from a Response-like object or plain string."""
    if hasattr(source, "output_text"):
        try:
            return source.output_text
        except Exception:
            pass
    return str(source)


def extract_json_from_text(source: Union[str, Any]) -> Dict[str, Any]:
    """
    Extract the first JSON object from either a Response-like object or plain text.

    Handles ```json``` fences, plain JSON strings, and falls back to raw text.
    """
    text = response_text(source)

    # Try fenced ```json ... ```
    fence = re.search(r"```json\s*([\s\S]+?)```", text, re.IGNORECASE)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    # Try parsing the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Return raw text as fallback
    return {"raw_text": text}

"""Utility helpers used across ptychi_evolve."""

import json
import re
from typing import Any, Dict, Union


def extract_json_from_text(source: Union[str, Any]) -> Dict[str, Any]:
    """
    Extract the first JSON object from either a Response-like object or plain text.

    Handles ```json``` fences, plain JSON strings, and falls back to raw text.
    """
    text = getattr(source, "output_text", None) or str(source)

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

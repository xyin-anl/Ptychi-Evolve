"""Utility helpers used across ptychi_evolve."""

import json
import re
from typing import Any, Dict, Union


def response_text(source: Union[str, Any]) -> str:
    """Return text content from a Response-like object or plain string."""
    if isinstance(source, str):
        return source

    # Prefer output_text when available (Responses API convenience)
    txt = getattr(source, "output_text", None)
    if txt:
        return txt

    # Fallback for SDKs without output_text: walk output[].content[].text
    try:
        output = getattr(source, "output", None) or []
        chunks = []
        for msg in output:
            for content in getattr(msg, "content", []):
                text_obj = getattr(content, "text", None)
                if text_obj is None:
                    continue
                # Pydantic models expose .value
                value = getattr(text_obj, "value", None)
                if value is not None:
                    chunks.append(str(value))
                else:
                    chunks.append(str(text_obj))
        if chunks:
            return "\n".join(chunks)
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

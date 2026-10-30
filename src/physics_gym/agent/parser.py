"""Pull {"action": ..., "params": {...}} out of a model response."""
import json
import re
from typing import Tuple


_VALID_ACTIONS = {"validate_code", "submit"}


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_fence(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json(text: str) -> dict:
    text = _strip_fence(_strip_think(text))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # outermost-braces fallback
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def parse_action(response: str) -> Tuple[str, dict]:
    payload = _extract_json(response)
    if not isinstance(payload, dict) or "action" not in payload:
        raise ValueError(f"missing 'action': {payload}")
    action = payload["action"]
    if action not in _VALID_ACTIONS:
        raise ValueError(f"invalid action {action!r}; must be one of {_VALID_ACTIONS}")
    params = payload.get("params", {}) or {}
    if not isinstance(params, dict):
        raise ValueError(f"'params' must be a dict, got {type(params).__name__}")
    if action == "validate_code":
        if "code" not in params or not isinstance(params["code"], str):
            raise ValueError("'validate_code' requires params.code: str")
    return action, params

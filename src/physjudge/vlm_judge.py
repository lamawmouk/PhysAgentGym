"""Claude-based VLM judge that mirrors PhyWorldBench's evaluate_videos.py protocol.

Same one_step prompt, same 8 evenly-sampled frames, same Yes/No JSON output —
the only swap is GPT-4o → Claude. Output is structurally compatible with
PhysJudge's JudgeResult so the comparison is apples-to-apples.
"""
import base64
import json
import os
import re
from pathlib import Path
from typing import Any
import numpy as np

from .types import Scenario, Verdict


_VLM_PROMPT_TEMPLATE = (
    "Suppose you are an expert in judging and evaluating the quality of AI-generated videos. "
    "These are frames evenly sampled from a generated video from the begining to the end. "
    "This is a generated video from a video model rather than captured from real world, so "
    "the video could be low quality, such as fuzzy, inconsistency, especially not following "
    'real world physics. Compare the objects and quantities visually present in the video with '
    'the specified object(s): "{objects}". Answer "Yes" if the object(s) could be found in the '
    'video, otherwise answer "No". Also, pleaes check if "{event}" is visually depicted in the '
    'video, and answer "Yes" or "No". Lastly, please check if video satisfies the standards in '
    'list: "{standards}", and answer "Yes" or "No" for each standard in the list.\n'
    "Return your evaluation in the following JSON format:\n"
    '"Objects": "Yes/No",\n'
    '"Event": "Yes/No",\n'
    '"Standard_1": "Yes/No",\n'
    '"Standard_2": "Yes/No",\n'
    '"...": "Yes/No"'
)


def sample_frames_jpeg(video_path: str | Path, n: int = 8, max_dim: int = 768) -> list[bytes]:
    """Read N evenly-spaced frames as JPEG bytes (lower bandwidth than PNG)."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total < n:
        raise RuntimeError(f"video {video_path} has < {n} frames ({total})")
    indices = np.linspace(0, total - 1, n).astype(int)
    out: list[bytes] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"could not read frame {idx} from {video_path}")
        h, w = frame.shape[:2]
        scale = min(1.0, max_dim / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            cap.release()
            raise RuntimeError(f"jpeg encode failed for frame {idx}")
        out.append(buf.tobytes())
    cap.release()
    return out


class ClaudeVLMJudge:
    """Mirrors PhyWorldBench's GPT-4o evaluate_videos.py one_step protocol on Claude."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None,
                 max_tokens: int = 512, n_frames: int = 8):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.n_frames = n_frames

    def _build_prompt(self, sc: Scenario) -> str:
        objs = sc.basic_standards.get("Objects", [])
        if isinstance(objs, list):
            objs = ", ".join(objs)
        event = sc.basic_standards.get("Event", "")
        return _VLM_PROMPT_TEMPLATE.format(
            objects=objs, event=event,
            standards=json.dumps(list(sc.key_standards)),
        )

    def judge(self, video_path: str | Path, sc: Scenario) -> dict[str, Any]:
        frames = sample_frames_jpeg(video_path, n=self.n_frames)
        content: list[dict] = [{"type": "text", "text": self._build_prompt(sc)}]
        for fb in frames:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg",
                            "data": base64.b64encode(fb).decode("ascii")},
            })
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        text = msg.content[0].text
        return _parse_yesno_response(text, n_standards=len(sc.key_standards))


class GeminiVLMJudge:
    """Mirrors ClaudeVLMJudge protocol on Google Gemini — same prompt, same 8 frames, same Yes/No JSON."""

    def __init__(self, model: str = "gemini-2.5-pro", api_key: str | None = None,
                 max_tokens: int = 2048, n_frames: int = 8):
        from google import genai
        self._genai = genai
        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.n_frames = n_frames

    def _build_prompt(self, sc: Scenario) -> str:
        objs = sc.basic_standards.get("Objects", [])
        if isinstance(objs, list):
            objs = ", ".join(objs)
        event = sc.basic_standards.get("Event", "")
        return _VLM_PROMPT_TEMPLATE.format(
            objects=objs, event=event,
            standards=json.dumps(list(sc.key_standards)),
        )

    def judge(self, video_path: str | Path, sc: Scenario) -> dict[str, Any]:
        from google.genai import types as gtypes
        frames = sample_frames_jpeg(video_path, n=self.n_frames)
        parts: list = [self._build_prompt(sc)]
        for fb in frames:
            parts.append(gtypes.Part.from_bytes(data=fb, mime_type="image/jpeg"))
        # retry transient errors
        import time
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=parts,
                    config=gtypes.GenerateContentConfig(max_output_tokens=self.max_tokens),
                )
                text = resp.text or ""
                return _parse_yesno_response(text, n_standards=len(sc.key_standards))
            except Exception as e:                  # noqa: BLE001
                last_err = e
                if attempt == 4:
                    raise
                time.sleep(min(60.0, 2.0 ** attempt))
        if last_err:
            raise last_err
        raise RuntimeError("unreachable")


class OpenAIVLMJudge:
    """Mirrors ClaudeVLMJudge protocol on OpenAI GPT-4o (PhyWorldBench's native judge)."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None,
                 max_tokens: int = 512, n_frames: int = 8):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.n_frames = n_frames

    def _build_prompt(self, sc: Scenario) -> str:
        objs = sc.basic_standards.get("Objects", [])
        if isinstance(objs, list):
            objs = ", ".join(objs)
        event = sc.basic_standards.get("Event", "")
        return _VLM_PROMPT_TEMPLATE.format(
            objects=objs, event=event,
            standards=json.dumps(list(sc.key_standards)),
        )

    def judge(self, video_path: str | Path, sc: Scenario) -> dict[str, Any]:
        frames = sample_frames_jpeg(video_path, n=self.n_frames)
        content: list[dict] = [{"type": "text", "text": self._build_prompt(sc)}]
        for fb in frames:
            data_url = "data:image/jpeg;base64," + base64.b64encode(fb).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        # GPT-5 family uses max_completion_tokens; GPT-4o uses max_tokens
        kwargs: dict = {"model": self.model,
                         "messages": [{"role": "user", "content": content}]}
        if self.model.startswith("gpt-5"):
            kwargs["max_completion_tokens"] = max(self.max_tokens, 2048)
        else:
            kwargs["max_tokens"] = self.max_tokens
        import time
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                return _parse_yesno_response(text, n_standards=len(sc.key_standards))
            except Exception as e:                  # noqa: BLE001
                last_err = e
                if attempt == 4:
                    raise
                time.sleep(min(60.0, 2.0 ** attempt))
        if last_err:
            raise last_err
        raise RuntimeError("unreachable")


def _parse_yesno_response(text: str, n_standards: int) -> dict[str, Any]:
    """Robustly parse PhyWorldBench-style Yes/No JSON. Handles code-fenced output and prose preamble."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    obj = None
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    if not obj:
        # last-resort line-by-line scrape
        obj = {}
        for k_match in re.finditer(r'"([^"]+)"\s*:\s*"(Yes|No)"', cleaned, re.IGNORECASE):
            obj[k_match.group(1)] = k_match.group(2).capitalize()
    standards: list[Verdict] = []
    for i in range(1, n_standards + 1):
        key = f"Standard_{i}"
        val = obj.get(key, "?")
        passed = isinstance(val, str) and val.strip().lower() == "yes"
        standards.append(Verdict(passed=passed, score=1.0 if passed else 0.0,
                                  diagnostic={"key": key, "raw": val}))
    return {
        "objects_passed": (obj.get("Objects", "").lower() == "yes"),
        "event_passed": (obj.get("Event", "").lower() == "yes"),
        "standards": standards,
        "raw_response": text,
        "raw_json": obj,
    }

import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from physjudge.types import Trajectory


CACHE_DIR = Path(os.environ.get("PHYSAGENTGYM_CACHE", "./cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TRAJECTORY_INSTRUCTIONS = (
    "Your code MUST print a single JSON object on the final line of stdout with keys:\n"
    '  {"t": [...], "pos": [[x, y], ...], "object_id": "...", "fps": <float>}\n'
    "  - t: list of N timestamps in seconds (monotonic).\n"
    "  - pos: list of N [x, y] pairs in METERS, y-up (physical convention, not pixel-down).\n"
    "  - confidence: optional list of N values in [0,1]; defaults to all-1.0.\n"
)


def _run_python(code: str, timeout: int = 60) -> tuple[int, str, str, float]:
    code_path = CACHE_DIR / f"agent_{uuid.uuid4().hex}.py"
    code_path.write_text(code)
    start = time.time()
    try:
        proc = subprocess.run(
            ["python", str(code_path)],
            capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr, time.time() - start
    except subprocess.TimeoutExpired as e:
        return -1, e.stdout or "", f"TimeoutError: {timeout}s", time.time() - start
    finally:
        try:
            code_path.unlink()
        except OSError:
            pass


def _parse_trajectory_payload(stdout: str) -> dict:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise ValueError("no trajectory JSON found on the final line of stdout")


def _payload_to_trajectory(payload: dict) -> Trajectory:
    t = np.asarray(payload["t"], dtype=float)
    pos = np.asarray(payload["pos"], dtype=float)
    if pos.ndim != 2 or pos.shape[0] != t.shape[0]:
        raise ValueError(f"bad shapes: t={t.shape}, pos={pos.shape}")
    if t.shape[0] < 3:
        raise ValueError(f"trajectory must have >=3 samples; got N={t.shape[0]}")
    conf = np.asarray(payload.get("confidence", [1.0] * len(t)), dtype=float)
    return Trajectory(
        t=t, pos=pos, confidence=conf,
        object_id=str(payload.get("object_id", "obj")),
        fps=float(payload.get("fps", 30.0)),
    )


def validate_code(code: str, timeout: int = 60) -> dict[str, Any]:
    try:
        compile(code, "<agent>", "exec")
    except SyntaxError as e:
        return {"type": "code_execution", "status": "FAILED",
                "env_message": f"SyntaxError: {e}", "trajectory": None}

    rc, stdout, stderr, dt = _run_python(code, timeout=timeout)
    if rc != 0:
        return {"type": "code_execution", "status": "FAILED",
                "env_message": (stderr or stdout)[-2000:],
                "execution_time": f"{dt:.2f}s", "trajectory": None}

    try:
        payload = _parse_trajectory_payload(stdout)
        traj = _payload_to_trajectory(payload)
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        return {"type": "code_execution", "status": "FAILED",
                "env_message": f"trajectory parse error: {e}\n--- stdout tail ---\n{stdout[-1000:]}",
                "execution_time": f"{dt:.2f}s", "trajectory": None}

    return {"type": "code_execution", "status": "SUCCESS",
            "env_message": f"executed; trajectory N={traj.N} D={traj.D} fps={traj.fps}",
            "execution_time": f"{dt:.2f}s", "trajectory": traj}


def submit(answer: str = "") -> dict[str, Any]:
    return {"type": "submit", "status": "SUCCESS", "env_message": "submitted"}

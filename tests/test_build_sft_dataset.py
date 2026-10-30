"""Synthetic rollouts -> filter -> SFT records, end-to-end."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def synth_rollout(task_id: int, best_score: float, pass_rate: float, n_steps: int = 2,
                  has_error: bool = False) -> dict:
    if has_error:
        return {"task_id": task_id, "error": "agent failed", "best_score": 0.0,
                "n_steps": 0, "transcript": []}
    transcript = [
        {"step": 0, "action": "validate_code",
         "raw_response": json.dumps({"action": "validate_code",
                                     "params": {"code": "print('{}')"}}),
         "params": {"code": "print('{}')"},
         "verdicts": [{"standard": "x", "passed": True, "score": best_score, "diagnostic": {}}],
         "mean_score": best_score},
        {"step": 1, "action": "submit",
         "raw_response": json.dumps({"action": "submit", "params": {}}),
         "params": {}},
    ]
    return {
        "task_id": task_id,
        "best_score": best_score,
        "pass_rate": pass_rate,
        "n_steps": n_steps,
        "regime": "synth",
        "category": "Test",
        "subcategory": "Synth",
        "scenario": f"synth scenario {task_id}",
        "initial_conditions": {"x0": 0.0},
        "usage": [],
        "transcript": transcript,
    }


def main():
    with tempfile.TemporaryDirectory() as td:
        rollout_path = Path(td) / "rollouts.jsonl"
        out_path = Path(td) / "sft.jsonl"
        # 3 keep (>=0.85, pass_rate=1.0), 2 drop (low score), 1 errored, 1 partial pass_rate
        rollouts = [
            synth_rollout(0, 0.95, 1.0),
            synth_rollout(1, 0.90, 1.0),
            synth_rollout(2, 0.86, 1.0),
            synth_rollout(3, 0.40, 0.5),
            synth_rollout(4, 0.70, 0.75),
            synth_rollout(5, 0.0, 0.0, has_error=True),
            synth_rollout(6, 0.99, 0.5),  # high score but low pass_rate -> dropped
        ]
        with rollout_path.open("w") as f:
            for r in rollouts:
                f.write(json.dumps(r) + "\n")

        result = subprocess.run(
            [sys.executable, "scripts/build_sft_dataset.py",
             "--rollouts", str(rollout_path), "--out", str(out_path),
             "--threshold", "0.85", "--require_pass_rate", "1.0"],
            cwd=REPO, capture_output=True, text=True, env={
                **__import__("os").environ,
                "PYTHONPATH": f"{REPO}",
            },
        )
        print(result.stdout)
        if result.returncode != 0:
            print("STDERR:", result.stderr)
            raise SystemExit(1)

        kept = [json.loads(line) for line in out_path.read_text().strip().splitlines()]
        assert len(kept) == 3, f"expected 3 SFT records, got {len(kept)}"
        for rec in kept:
            assert rec["meta"]["best_score"] >= 0.85
            assert rec["meta"]["pass_rate"] == 1.0
            roles = [m["role"] for m in rec["messages"]]
            assert roles[0] == "system"
            assert roles[1] == "user"
            assert "assistant" in roles
            assert "submit" in rec["messages"][-1]["content"]
        print("OK")


if __name__ == "__main__":
    main()


def test_run():
    main()

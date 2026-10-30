"""Smoke test: env loads a PE-KE task, runs a hand-crafted correct script through
validate_code, and physjudge mean_score is high. No LLM in this loop."""

from physics_gym.env.base import PhysicsEnv
from physics_gym.env.task.pe_ke import PEKETask
from physics_gym.utils.compiler import PhysicsGymCompiler


GOLDEN_AGENT_CODE = """
import json
import numpy as np

g = 9.81
ic = {ic}
T = ic["duration_s"]; fps = ic["fps"]
n = int(round(T * fps))
t = np.linspace(0.0, T, n)
x = ic["x0"] + ic["vx0"] * t
y = ic["y0"] + ic["vy0"] * t - 0.5 * g * t * t
pos = np.stack([x, y], axis=1)
print(json.dumps({{"t": t.tolist(), "pos": pos.tolist(), "object_id": "ball", "fps": fps}}))
"""


def main():
    env = PhysicsEnv(
        task_entrypoint=PEKETask,
        task_kwargs={"data_path": "data/pe_ke", "mode": "test"},
        max_steps=4,
        compiler=PhysicsGymCompiler(),
    )
    scores = []
    for tid in range(5):
        obs, info = env.reset(task_id=tid)
        ic = obs["info"]["initial_conditions"]
        code = GOLDEN_AGENT_CODE.format(ic=ic)
        result, _, _, _, _ = env.step("validate_code", code=code)
        result, reward, done, _, _ = env.step("submit")
        scores.append(env.best_score)
        regime = env.task.task_list[tid]["regime"]
        print(f"task {tid:>2}: regime={regime:>8} best_score={env.best_score:.3f} "
              f"pass_rate={env.best_judge_result.pass_rate:.3f}")
        for a, v in zip(env.best_judge_result.assertions, env.best_judge_result.verdicts):
            print(f"   {'PASS' if v.passed else 'FAIL'} score={v.score:.3f} "
                  f"[{a.description}] {a.standard_text[:70]}")

    mean = sum(scores) / len(scores)
    print(f"\nmean best_score over {len(scores)} tasks: {mean:.3f}")
    assert mean > 0.85, f"golden code should score high; got {mean:.3f}"
    print("OK")


if __name__ == "__main__":
    main()


def test_run():
    main()

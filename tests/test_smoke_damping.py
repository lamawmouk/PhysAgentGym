"""Smoke test: golden agent against Damping regimes."""

from physics_gym.env.base import PhysicsEnv
from physics_gym.env.task.damping import DampingTask
from physics_gym.utils.compiler import PhysicsGymCompiler


GOLDEN_CODE = """
import json
import numpy as np
ic = {ic}
T = ic['duration_s']; fps = ic['fps']
n = int(round(T * fps))
t = np.linspace(0.0, T, n)
if 'A' in ic:
    A = ic['A']; gamma = ic['gamma']; omega = ic['omega']; phi = ic.get('phi', 0.0)
    y = A * np.exp(-gamma * t) * np.cos(omega * t + phi)
else:
    y0 = ic['y0']; gamma = ic['gamma']
    y = y0 * np.exp(-gamma * t)
x = np.zeros_like(t)
pos = np.stack([x, y], axis=1)
print(json.dumps({{'t': t.tolist(), 'pos': pos.tolist(), 'object_id': 'mass', 'fps': fps}}))
"""


def main():
    env = PhysicsEnv(
        task_entrypoint=DampingTask,
        task_kwargs={"data_path": "data/damping", "mode": "test"},
        max_steps=4,
        compiler=PhysicsGymCompiler(),
    )
    scores = []
    for tid in range(10):
        obs, _ = env.reset(task_id=tid)
        ic = obs["info"]["initial_conditions"]
        env.step("validate_code", code=GOLDEN_CODE.format(ic=ic))
        env.step("submit")
        scores.append(env.best_score)
        regime = env.task.task_list[tid]["regime"]
        print(f"task {tid:>2}: regime={regime:>22} best_score={env.best_score:.3f} "
              f"pass_rate={env.best_judge_result.pass_rate:.3f}")
        for a, v in zip(env.best_judge_result.assertions, env.best_judge_result.verdicts):
            mark = "PASS" if v.passed else "FAIL"
            print(f"   {mark} score={v.score:.3f} [{a.description}] {a.standard_text[:60]}")

    mean = sum(scores) / len(scores)
    print(f"\nmean = {mean:.3f}")
    assert mean > 0.80, f"golden code should score high; got {mean:.3f}"
    print("OK")


if __name__ == "__main__":
    main()


def test_run():
    main()

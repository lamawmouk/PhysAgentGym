"""Smoke test: golden agent against Linear Motion regimes."""

from physics_gym.env.base import PhysicsEnv
from physics_gym.env.task.linear_motion import LinearMotionTask
from physics_gym.utils.compiler import PhysicsGymCompiler


GOLDEN_CODE = """
import json
import numpy as np
ic = {ic}
T = ic['duration_s']; fps = ic['fps']
n = int(round(T * fps))
t = np.linspace(0.0, T, n)

if 'T_stop' in ic:
    T_stop = ic['T_stop']
    vx = ic['vx0']; decel = -ic['ax']
    x_at_stop = ic['x0'] + vx * T_stop - 0.5 * decel * T_stop * T_stop
    moving = t <= T_stop
    x = np.where(moving, ic['x0'] + vx * t - 0.5 * decel * t * t, x_at_stop)
    y = np.full_like(t, ic['y0'])
else:
    x = ic['x0'] + ic['vx0'] * t + 0.5 * ic['ax'] * t * t
    y = ic['y0'] + ic['vy0'] * t + 0.5 * ic['ay'] * t * t

pos = np.stack([x, y], axis=1)
print(json.dumps({{'t': t.tolist(), 'pos': pos.tolist(), 'object_id': 'obj', 'fps': fps}}))
"""


def main():
    env = PhysicsEnv(
        task_entrypoint=LinearMotionTask,
        task_kwargs={"data_path": "data/linear_motion", "mode": "test"},
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
        print(f"task {tid:>2}: regime={regime:>18} best_score={env.best_score:.3f} "
              f"pass_rate={env.best_judge_result.pass_rate:.3f}")
        for a, v in zip(env.best_judge_result.assertions, env.best_judge_result.verdicts):
            mark = "PASS" if v.passed else "FAIL"
            print(f"   {mark} score={v.score:.3f} [{a.description}] {a.standard_text[:60]}")

    mean = sum(scores) / len(scores)
    print(f"\nmean = {mean:.3f}")
    assert mean > 0.85, f"golden code should score high; got {mean:.3f}"
    print("OK")


if __name__ == "__main__":
    main()


def test_run():
    main()

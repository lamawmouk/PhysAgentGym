"""Run the GOLDEN agent through the env on a chosen task subcategory and dump
the rollouts to JSONL in the same format rollout.py emits. Useful for testing
the SFT pipeline before real LLM rollouts are available.
"""
import argparse
import json
from pathlib import Path


from physics_gym.env.base import PhysicsEnv
from physics_gym.env.task.linear_motion import LinearMotionTask
from physics_gym.env.task.pe_ke import PEKETask
from physics_gym.utils.compiler import PhysicsGymCompiler


GOLDEN_PE_KE = """
import json
import numpy as np
g = 9.81
ic = {ic}
T = ic['duration_s']; fps = ic['fps']
n = int(round(T * fps))
t = np.linspace(0.0, T, n)
x = ic['x0'] + ic['vx0'] * t
y = ic['y0'] + ic['vy0'] * t - 0.5 * g * t * t
pos = np.stack([x, y], axis=1)
print(json.dumps({{'t': t.tolist(), 'pos': pos.tolist(), 'object_id': 'ball', 'fps': fps}}))
"""

GOLDEN_LINEAR = """
import json
import numpy as np
ic = {ic}
T = ic['duration_s']; fps = ic['fps']
n = int(round(T * fps))
t = np.linspace(0.0, T, n)
if 'T_stop' in ic:
    T_stop = ic['T_stop']; vx = ic['vx0']; decel = -ic['ax']
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

REGISTRY = {
    "pe_ke": (PEKETask, GOLDEN_PE_KE),
    "linear_motion": (LinearMotionTask, GOLDEN_LINEAR),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(REGISTRY))
    ap.add_argument("--mode", default="train", choices=["train", "test"])
    ap.add_argument("--n_tasks", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    task_cls, golden_template = REGISTRY[args.task]
    out_path = Path(args.out or f"results/{args.task}_{args.mode}_synthetic.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = PhysicsEnv(
        task_entrypoint=task_cls,
        task_kwargs={"data_path": f"data/{args.task}", "mode": args.mode},
        max_steps=4,
        compiler=PhysicsGymCompiler(),
    )

    with out_path.open("w") as f:
        for tid in range(args.n_tasks):
            obs, _ = env.reset(task_id=tid)
            ic = obs["info"]["initial_conditions"]
            code = golden_template.format(ic=ic)
            transcript = []

            r1, _, _, _, _ = env.step("validate_code", code=code)
            transcript.append({
                "step": 0, "action": "validate_code",
                "raw_response": json.dumps({"action": "validate_code", "params": {"code": code}}),
                "params": {"code": code},
                "verdicts": r1.get("verdicts"),
                "mean_score": r1.get("mean_score"),
            })
            r2, reward, terminated, _, _ = env.step("submit")
            transcript.append({
                "step": 1, "action": "submit",
                "raw_response": json.dumps({"action": "submit", "params": {}}),
                "params": {},
            })

            td = env.task.task_list[tid]
            rollout = {
                "task_id": tid,
                "best_score": env.best_score,
                "pass_rate": env.best_judge_result.pass_rate,
                "n_steps": 2,
                "regime": td["regime"],
                "category": td["category"],
                "subcategory": td["subcategory"],
                "scenario": td["scenario"],
                "initial_conditions": td["initial_conditions"],
                "usage": [],
                "transcript": transcript,
            }
            f.write(json.dumps(rollout) + "\n")

    print(f"wrote {args.n_tasks} synthetic rollouts -> {out_path}")


if __name__ == "__main__":
    main()

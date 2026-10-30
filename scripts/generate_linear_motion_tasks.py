"""Linear-motion scenarios. Three regimes that cover different physjudge checks:
  - horizontal_cruise: constant horizontal velocity (no accel)
  - rocket_launch:     constant upward acceleration (engines, no gravity)
  - slowing_skid:      horizontal motion decelerating to rest then staying at rest

Each regime hits a distinct subset of physjudge rules so the verifier exercises
   straight_line + constant_speed + elevation_unchanged + no_direction_changes
   straight_line(vertical) + monotonic_up + smooth_velocity
   straight_line(horizontal) + damping(energy_non_increasing) + elevation_unchanged
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np


FPS = 60.0


def simulate_uniform_motion(x0, y0, vx, vy, ax, ay, T, fps=FPS):
    n = int(round(T * fps))
    t = np.linspace(0.0, T, n)
    x = x0 + vx * t + 0.5 * ax * t * t
    y = y0 + vy * t + 0.5 * ay * t * t
    return t, np.stack([x, y], axis=1)


def simulate_skid(x0, y0, vx, decel, T_stop, T, fps=FPS):
    n = int(round(T * fps))
    t = np.linspace(0.0, T, n)
    x_at_stop = x0 + vx * T_stop - 0.5 * decel * T_stop * T_stop
    moving = t <= T_stop
    x = np.where(moving, x0 + vx * t - 0.5 * decel * t * t, x_at_stop)
    y = np.full_like(t, y0)
    return t, np.stack([x, y], axis=1)


def regime_horizontal_cruise(rng) -> dict:
    vx = float(rng.uniform(5.0, 30.0))
    T = float(rng.uniform(2.0, 8.0))
    h = float(rng.uniform(0.0, 5.0))
    t, pos = simulate_uniform_motion(0.0, h, vx, 0.0, 0.0, 0.0, T)
    return {
        "regime": "horizontal_cruise",
        "ic": {"x0": 0.0, "y0": h, "vx0": vx, "vy0": 0.0, "ax": 0.0, "ay": 0.0,
               "duration_s": T, "fps": FPS, "mass_kg": 1.0},
        "scenario": (
            f"A vehicle moves to the right along a flat horizontal road at constant "
            f"velocity {vx:.2f} m/s for {T:.1f} seconds. There is no acceleration. "
            f"Simulate its motion and emit the trajectory."
        ),
        "key_standards": [
            "The vehicle moves horizontally in a straight line.",
            "The vehicle moves at constant speed (no acceleration).",
            "The vehicle's elevation remains unchanged throughout.",
            "There are no changes in direction.",
        ],
        "t": t.tolist(), "pos": pos.tolist(),
    }


def regime_rocket_launch(rng) -> dict:
    a = float(rng.uniform(5.0, 30.0))
    T = float(rng.uniform(2.0, 6.0))
    t, pos = simulate_uniform_motion(0.0, 0.0, 0.0, 0.0, 0.0, a, T)
    return {
        "regime": "rocket_launch",
        "ic": {"x0": 0.0, "y0": 0.0, "vx0": 0.0, "vy0": 0.0, "ax": 0.0, "ay": a,
               "duration_s": T, "fps": FPS, "mass_kg": 1.0},
        "scenario": (
            f"A rocket launches vertically upward from the ground with constant upward "
            f"acceleration {a:.2f} m/s^2 (treat the engine as constant thrust and "
            f"ignore gravity for this idealized scenario). Simulate its motion for "
            f"{T:.1f} seconds and emit the trajectory."
        ),
        "key_standards": [
            "The rocket moves vertically in a straight line.",
            "The rocket ascends (moves upward) throughout the simulation.",
            "The motion is smooth without abrupt changes in velocity.",
        ],
        "t": t.tolist(), "pos": pos.tolist(),
    }


def regime_slowing_skid(rng) -> dict:
    vx = float(rng.uniform(10.0, 30.0))
    decel = float(rng.uniform(2.0, 8.0))
    T_stop = vx / decel
    T = T_stop + float(rng.uniform(0.5, 1.5))
    h = float(rng.uniform(0.0, 3.0))
    t, pos = simulate_skid(0.0, h, vx, decel, T_stop, T)
    return {
        "regime": "slowing_skid",
        "ic": {"x0": 0.0, "y0": h, "vx0": vx, "vy0": 0.0, "ax": -decel, "ay": 0.0,
               "duration_s": T, "T_stop": T_stop, "fps": FPS, "mass_kg": 1.0},
        "scenario": (
            f"An object slides along a horizontal surface, starting at velocity "
            f"{vx:.2f} m/s. It decelerates uniformly at {decel:.2f} m/s^2 due to "
            f"friction and comes to rest at t={T_stop:.2f}s. After that it remains "
            f"stationary. Simulate the motion for {T:.1f} seconds total and emit the "
            f"trajectory."
        ),
        "key_standards": [
            "The object moves horizontally in a straight line.",
            "The object slows down (loses energy) and comes to rest.",
            "The object's elevation remains unchanged throughout.",
        ],
        "t": t.tolist(), "pos": pos.tolist(),
    }


REGIMES = [regime_horizontal_cruise, regime_rocket_launch, regime_slowing_skid]


def make_instance(task_id: int, rng) -> dict:
    spec = rng.choice(REGIMES)(rng)
    physjudge_spec = {
        "prompt_index": f"linear-{task_id:05d}",
        "prompt": spec["scenario"],
        "physics": spec["scenario"],
        "basic_standards": {"Objects": ["object"], "Event": spec["regime"]},
        "key_standards": spec["key_standards"],
    }
    return {
        "task_id": task_id,
        "category": "Object Motion and Kinematics",
        "subcategory": "Linear Motion",
        "regime": spec["regime"],
        "scenario": spec["scenario"],
        "initial_conditions": spec["ic"],
        "physjudge_spec": physjudge_spec,
        "golden": {"t": spec["t"], "pos": spec["pos"], "fps": FPS},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/linear_motion")
    ap.add_argument("--n_train", type=int, default=50)
    ap.add_argument("--n_test", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    class CombinedRNG:
        def uniform(self, a, b): return np_rng.uniform(a, b)
        def choice(self, seq): return py_rng.choice(seq)

    rng = CombinedRNG()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split, n in [("train", args.n_train), ("test", args.n_test)]:
        items = [make_instance(i, rng) for i in range(n)]
        path = out / f"{split}_tasks.jsonl"
        with path.open("w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        print(f"wrote {n} -> {path}")


if __name__ == "__main__":
    main()

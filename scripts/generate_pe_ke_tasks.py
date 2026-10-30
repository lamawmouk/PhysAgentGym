"""Parameter-sweep PE<->KE scenarios + golden simulator -> train/test JSONL.

Three regimes:
  - drop:    v0 = (0, 0)         -> straight vertical free-fall
  - throw_v: v0 = (0, vy>0)      -> rise to apex, then fall
  - throw_a: v0 = (vx>0, vy>0)   -> parabolic arc

Each instance is a JSON line consumed by physics_gym.env.task.pe_ke.PEKETask.
Golden trajectories use closed-form ballistics; the verifier (physjudge)
scores against the agent's reproduction at evaluation time.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

G = 9.81


def simulate(x0: float, y0: float, vx: float, vy: float, T: float, fps: float = 60.0):
    n = int(round(T * fps))
    t = np.linspace(0.0, T, n)
    x = x0 + vx * t
    y = y0 + vy * t - 0.5 * G * t * t
    pos = np.stack([x, y], axis=1)
    return t, pos


def regime_drop(rng) -> dict:
    h = float(rng.uniform(2.0, 20.0))
    T = float(np.sqrt(2 * h / G))
    return {
        "regime": "drop",
        "x0": 0.0, "y0": h, "vx": 0.0, "vy": 0.0, "T": T,
        "scenario": (
            f"A point mass is released from rest at height {h:.2f} m above the ground "
            f"and falls under gravity (g = {G} m/s^2, no air drag). Simulate the motion "
            f"from t=0 until it reaches y=0 and emit the trajectory."
        ),
        "key_standards": [
            "The object falls due to gravity along a straight vertical line.",
            "The object descends monotonically downward with smooth, low-jerk motion.",
            "Total mechanical energy is conserved (energy non-increasing within tolerance).",
        ],
    }


def regime_throw_vertical(rng) -> dict:
    vy = float(rng.uniform(5.0, 25.0))
    h = float(rng.uniform(0.0, 5.0))
    # time to apex + fall back to ground
    T_apex = vy / G
    H_top = h + 0.5 * vy * vy / G
    T = T_apex + float(np.sqrt(2 * H_top / G))
    return {
        "regime": "throw_v",
        "x0": 0.0, "y0": h, "vx": 0.0, "vy": vy, "T": T,
        "scenario": (
            f"A point mass is launched vertically upward from height {h:.2f} m with "
            f"initial speed {vy:.2f} m/s. Under gravity (g = {G}, no drag) it rises to "
            f"a clear apex and then falls. Simulate the motion until it returns to the "
            f"ground and emit the trajectory."
        ),
        "key_standards": [
            "The object rises to a clear apex and then descends.",
            "Acceleration is approximately constant downward (gravity-like).",
            "Total mechanical energy is conserved.",
            "The motion follows a straight vertical line.",
        ],
    }


def regime_throw_angle(rng) -> dict:
    speed = float(rng.uniform(8.0, 25.0))
    angle = float(rng.uniform(20.0, 70.0))  # degrees
    rad = np.deg2rad(angle)
    vx = float(speed * np.cos(rad))
    vy = float(speed * np.sin(rad))
    h = float(rng.uniform(0.0, 5.0))
    H_top = h + 0.5 * vy * vy / G
    T = vy / G + float(np.sqrt(2 * H_top / G))
    return {
        "regime": "throw_a",
        "x0": 0.0, "y0": h, "vx": vx, "vy": vy, "T": T,
        "scenario": (
            f"A point mass is launched from height {h:.2f} m with initial velocity "
            f"({vx:.2f}, {vy:.2f}) m/s under gravity (g = {G}, no drag). It traces a "
            f"parabolic arc to a clear peak and then descends. Simulate until it "
            f"returns to ground level and emit the trajectory."
        ),
        "key_standards": [
            "The object follows a parabolic arc through the air.",
            "Acceleration is approximately constant downward (gravity-like).",
            "The object reaches a clear apex and then descends smoothly.",
            "Total mechanical energy is conserved.",
        ],
    }


REGIMES = [regime_drop, regime_throw_vertical, regime_throw_angle]


def make_instance(task_id: int, rng) -> dict:
    spec = rng.choice(REGIMES)(rng)
    t, pos = simulate(spec["x0"], spec["y0"], spec["vx"], spec["vy"], spec["T"])
    physjudge_spec = {
        "prompt_index": f"pe_ke-{task_id:05d}",
        "prompt": spec["scenario"],
        "physics": spec["scenario"],
        "basic_standards": {"Objects": ["point mass"], "Event": spec["regime"]},
        "key_standards": spec["key_standards"],
    }
    return {
        "task_id": task_id,
        "category": "Energy Conservation",
        "subcategory": "Potential to Kinetic Energy",
        "regime": spec["regime"],
        "scenario": spec["scenario"],
        "initial_conditions": {
            "x0": spec["x0"], "y0": spec["y0"],
            "vx0": spec["vx"], "vy0": spec["vy"],
            "g": G, "duration_s": spec["T"], "fps": 60.0, "mass_kg": 1.0,
        },
        "physjudge_spec": physjudge_spec,
        "golden": {
            "t": t.tolist(),
            "pos": pos.tolist(),
            "fps": 60.0,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/pe_ke")
    ap.add_argument("--n_train", type=int, default=180)
    ap.add_argument("--n_test", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    class CombinedRNG:
        def __init__(self, py_rng, np_rng):
            self.py = py_rng; self.np = np_rng
        def uniform(self, a, b): return self.np.uniform(a, b)
        def choice(self, seq): return self.py.choice(seq)

    crng = CombinedRNG(rng, np_rng)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = [make_instance(i, crng) for i in range(args.n_train)]
    test = [make_instance(i, crng) for i in range(args.n_test)]

    for split, items in [("train", train), ("test", test)]:
        path = out / f"{split}_tasks.jsonl"
        with path.open("w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        print(f"wrote {len(items)} -> {path}")


if __name__ == "__main__":
    main()

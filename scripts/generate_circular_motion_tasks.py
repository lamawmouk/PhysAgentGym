"""Uniform circular motion scenarios. Closed-form: x = cx + r*cos(omega*t + phi),
y = cy + r*sin(omega*t + phi).

Two regimes:
  - ferris_wheel: human-scale (r in [2, 15] m), several rotations
  - orbit:        astronomical-scale (r in [100, 1000] m), single revolution

Both reduce to the same simulator; differ only in scale and framing of the scenario
text. Standards trigger physjudge's circular_motion (constant radius) + constant_speed.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np


FPS = 60.0


def simulate_circle(cx, cy, r, omega, phi, T, fps=FPS):
    n = int(round(T * fps))
    t = np.linspace(0.0, T, n)
    x = cx + r * np.cos(omega * t + phi)
    y = cy + r * np.sin(omega * t + phi)
    pos = np.stack([x, y], axis=1)
    return t, pos


def regime_ferris_wheel(rng) -> dict:
    r = float(rng.uniform(2.0, 15.0))
    omega = float(rng.uniform(0.3, 1.0))           # 0.3-1.0 rad/s -> period 6-21 s
    # Integer rotations so the trajectory's centroid coincides with the geometric
    # center; otherwise circular_motion check sees radius variation from offset.
    n_rotations = int(rng.choice([1, 2, 3]))
    T = 2 * np.pi * n_rotations / omega
    phi = 0.0
    cx, cy = 0.0, r                                 # bottom of wheel at origin
    t, pos = simulate_circle(cx, cy, r, omega, phi, T)
    return {
        "regime": "ferris_wheel",
        "ic": {"cx": cx, "cy": cy, "r": r, "omega": omega, "phi": phi,
               "duration_s": T, "fps": FPS, "mass_kg": 1.0},
        "scenario": (
            f"A passenger car on a ferris wheel of radius {r:.2f} m moves in a circle "
            f"around the wheel's hub at constant angular velocity {omega:.3f} rad/s. "
            f"The hub is at (0, {r:.2f}); the car starts at the rightmost point of the "
            f"wheel and moves counterclockwise. Simulate the motion for {T:.1f} seconds "
            f"({n_rotations:.1f} full rotations) and emit the trajectory."
        ),
        "key_standards": [
            "The car moves in a circular path at constant radius from the hub.",
            "The car moves at constant speed throughout the rotation.",
            "The motion is smooth without abrupt jumps in velocity.",
        ],
        "t": t.tolist(), "pos": pos.tolist(),
    }


def regime_orbit(rng) -> dict:
    r = float(rng.uniform(100.0, 1000.0))           # radius in meters
    period = float(rng.uniform(60.0, 300.0))        # orbital period in seconds
    omega = 2 * np.pi / period
    n_revolutions = int(rng.choice([1, 2]))
    T = period * n_revolutions
    phi = 0.0
    cx, cy = 0.0, 0.0
    t, pos = simulate_circle(cx, cy, r, omega, phi, T)
    return {
        "regime": "orbit",
        "ic": {"cx": cx, "cy": cy, "r": r, "omega": omega, "phi": phi,
               "duration_s": T, "fps": FPS, "mass_kg": 1.0},
        "scenario": (
            f"A satellite orbits a central body at radius {r:.1f} m with orbital "
            f"period {period:.1f} s (angular velocity {omega:.4f} rad/s). The orbit "
            f"is circular and centered at the origin. Simulate the motion for "
            f"{T:.1f} seconds and emit the trajectory."
        ),
        "key_standards": [
            "The satellite moves in a circular orbit at constant radius from the center.",
            "The satellite moves at constant speed throughout the orbit.",
            "The motion is smooth without abrupt jumps in velocity.",
        ],
        "t": t.tolist(), "pos": pos.tolist(),
    }


REGIMES = [regime_ferris_wheel, regime_orbit]


def make_instance(task_id: int, rng) -> dict:
    spec = rng.choice(REGIMES)(rng)
    physjudge_spec = {
        "prompt_index": f"circular-{task_id:05d}",
        "prompt": spec["scenario"],
        "physics": spec["scenario"],
        "basic_standards": {"Objects": ["object"], "Event": spec["regime"]},
        "key_standards": spec["key_standards"],
    }
    return {
        "task_id": task_id,
        "category": "Object Motion and Kinematics",
        "subcategory": "Circular Motion",
        "regime": spec["regime"],
        "scenario": spec["scenario"],
        "initial_conditions": spec["ic"],
        "physjudge_spec": physjudge_spec,
        "golden": {"t": spec["t"], "pos": spec["pos"], "fps": FPS},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/circular_motion")
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--n_test", type=int, default=30)
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

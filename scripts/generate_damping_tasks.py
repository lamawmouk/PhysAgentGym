"""Damped motion scenarios. Two regimes that exercise different physjudge checks:

  - underdamped_oscillator: x(t) = A*exp(-gamma*t)*cos(omega*t + phi)
                            -> oscillates while losing energy; bounces + damping checks
  - overdamped_decay:       x(t) = x0 * exp(-gamma*t)
                            -> monotonic decay to rest; monotonic + damping checks

Single-axis motion (along x), y stays fixed (height = 0). Both reduce to closed-form;
no ODE integration required.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np


FPS = 60.0


def simulate_underdamped(A, gamma, omega, phi, T, fps=FPS):
    """Vertical mass-spring oscillator: y(t) = A*exp(-gamma*t)*cos(omega*t+phi)
    around equilibrium y=0. Motion along axis=1 (y) so physjudge bounces /
    monotonic / amplitude_decay primitives fire correctly."""
    n = int(round(T * fps))
    t = np.linspace(0.0, T, n)
    y = A * np.exp(-gamma * t) * np.cos(omega * t + phi)
    pos = np.stack([np.zeros_like(t), y], axis=1)
    return t, pos


def simulate_overdamped(y0, gamma, T, fps=FPS):
    """Vertical exponential settling toward y=0. Motion along axis=1."""
    n = int(round(T * fps))
    t = np.linspace(0.0, T, n)
    y = y0 * np.exp(-gamma * t)
    pos = np.stack([np.zeros_like(t), y], axis=1)
    return t, pos


def regime_underdamped(rng) -> dict:
    A = float(rng.uniform(0.5, 3.0))
    gamma = float(rng.uniform(0.2, 0.8))
    omega = float(rng.uniform(2.0, 6.0))
    phi = 0.0
    T = float(rng.uniform(8.0, 14.0))
    t, pos = simulate_underdamped(A, gamma, omega, phi, T)
    return {
        "regime": "underdamped_oscillator",
        "ic": {"A": A, "gamma": gamma, "omega": omega, "phi": phi,
               "duration_s": T, "fps": FPS, "mass_kg": 1.0, "axis": "y"},
        "scenario": (
            f"A mass on a vertical spring oscillates with friction. The vertical "
            f"position follows y(t) = A * exp(-gamma * t) * cos(omega * t + phi) "
            f"with A = {A:.2f} m, gamma = {gamma:.3f} 1/s, omega = {omega:.3f} rad/s, "
            f"phi = 0 rad. Equilibrium is at y=0. Simulate for {T:.1f} seconds and "
            f"emit the trajectory (x stays at 0; motion is along y)."
        ),
        "key_standards": [
            "The mass bounces around its equilibrium (sign changes in vertical velocity).",
            "The oscillation amplitude decreases over time as the system loses energy.",
            "The mass eventually comes to rest near equilibrium.",
        ],
        "t": t.tolist(), "pos": pos.tolist(),
    }


def regime_overdamped(rng) -> dict:
    y0 = float(rng.uniform(1.0, 5.0))
    gamma = float(rng.uniform(0.5, 2.0))
    T = float(rng.uniform(4.0, 8.0))
    t, pos = simulate_overdamped(y0, gamma, T)
    return {
        "regime": "overdamped_decay",
        "ic": {"y0": y0, "gamma": gamma,
               "duration_s": T, "fps": FPS, "mass_kg": 1.0, "axis": "y"},
        "scenario": (
            f"An object in a heavily-damped medium decays exponentially toward "
            f"equilibrium. Vertical position follows y(t) = y0 * exp(-gamma * t) "
            f"with y0 = {y0:.2f} m and gamma = {gamma:.3f} 1/s. The object descends "
            f"monotonically and eventually settles at rest. Simulate for {T:.1f} "
            f"seconds and emit the trajectory."
        ),
        "key_standards": [
            "The object descends monotonically toward equilibrium.",
            "The amplitude decreases over time (the system loses energy).",
            "The object eventually comes to rest.",
        ],
        "t": t.tolist(), "pos": pos.tolist(),
    }


REGIMES = [regime_underdamped, regime_overdamped]


def make_instance(task_id: int, rng) -> dict:
    spec = rng.choice(REGIMES)(rng)
    physjudge_spec = {
        "prompt_index": f"damping-{task_id:05d}",
        "prompt": spec["scenario"],
        "physics": spec["scenario"],
        "basic_standards": {"Objects": ["mass"], "Event": spec["regime"]},
        "key_standards": spec["key_standards"],
    }
    return {
        "task_id": task_id,
        "category": "Energy Conservation",
        "subcategory": "Energy Loss (Damping)",
        "regime": spec["regime"],
        "scenario": spec["scenario"],
        "initial_conditions": spec["ic"],
        "physjudge_spec": physjudge_spec,
        "golden": {"t": spec["t"], "pos": spec["pos"], "fps": FPS},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/damping")
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

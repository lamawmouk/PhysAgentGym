"""Composite (harder) subcategories. Each regime composes several laws and has no
one-line closed form for the agent to copy:

  drag     : drag_launch, drag_drop        quadratic air drag (golden: RK45, rtol 1e-10)
  kepler   : orbit_moderate, orbit_eccentric   two-body orbit from perihelion (golden: Kepler's equation)
  driven   : near_resonance, off_resonance     driven damped oscillator from rest (golden: closed form)
  coupled  : single_excited, mixed_modes       two masses, three springs (golden: normal modes)

Unlike the damping subcategory, scenarios give the physical constants and initial
state, never the solution formula. Standards carry the constants ("name = value")
that the verifier's law-of-motion checks need; no golden trajectory is consulted.

  python scripts/generate_harder_tasks.py            # writes data/{drag,kepler,driven,coupled}/
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

FPS = 60.0


def ic_standard(x0, y0, vx0, vy0):
    return (f"The motion starts from the stated initial state (x0 = {x0:.5f}, y0 = {y0:.5f}, "
            f"vx0 = {vx0:.5f}, vy0 = {vy0:.5f}).")


def grid(T, fps=FPS):
    return np.linspace(0.0, T, int(round(T * fps)) + 1)


# ---------------------------------------------------------------- drag
def _drag_rhs(c_over_m, g):
    def f(_, s):
        vx, vy = s[2], s[3]
        sp = np.hypot(vx, vy)
        return [vx, vy, -c_over_m * sp * vx, -g - c_over_m * sp * vy]
    return f


def _drag_solve(y0, vx0, vy0, c_over_m, g, T=None):
    f = _drag_rhs(c_over_m, g)
    if T is None:  # integrate to ground contact
        hit = lambda _, s: s[1]  # noqa: E731
        hit.terminal, hit.direction = True, -1
        probe = solve_ivp(f, (0, 60), [0, y0, vx0, vy0], events=hit, rtol=1e-10, atol=1e-12)
        T = float(probe.t_events[0][0])
    t = grid(T)
    sol = solve_ivp(f, (0, T), [0, y0, vx0, vy0], t_eval=t, rtol=1e-10, atol=1e-12)
    return T, t, np.stack([sol.y[0], sol.y[1]], axis=1)


def regime_drag_launch(rng):
    g, cm = 9.81, float(rng.uniform(0.008, 0.025))
    v0, ang, y0 = float(rng.uniform(18, 35)), float(rng.uniform(25, 60)), float(rng.uniform(0.0, 3.0))
    vx0, vy0 = v0 * np.cos(np.radians(ang)), v0 * np.sin(np.radians(ang))
    T, t, pos = _drag_solve(y0, vx0, vy0, cm, g)
    return dict(
        regime="drag_launch",
        ic={"x0": 0.0, "y0": y0, "vx0": vx0, "vy0": vy0, "g": g, "drag_c_over_m": cm,
            "duration_s": T, "fps": FPS},
        scenario=(f"A ball is launched from height {y0:.2f} m with initial velocity ({vx0:.2f}, {vy0:.2f}) m/s. "
                  f"It moves under gravity (g = {g}) and quadratic air drag: the drag acceleration is "
                  f"-(c/m)|v|v with c/m = {cm:.4f} 1/m. Simulate until it returns to ground level (y = 0) "
                  f"and emit the trajectory."),
        key_standards=[
            ic_standard(0.0, y0, vx0, vy0),
            f"Newton's second law holds with gravity and quadratic air drag (g = {g}, c/m = {cm:.5f}).",
            "The object reaches a clear apex and then descends.",
        ],
        t=t, pos=pos)


def regime_drag_drop(rng):
    g, cm = 9.81, float(rng.uniform(0.01, 0.03))
    vt = np.sqrt(g / cm)
    T = float(rng.uniform(4.0, 5.0)) * vt / g  # long enough to reach >99.9% of terminal speed
    y0 = float(np.ceil((vt ** 2 / g) * np.log(np.cosh(g * T / vt)) + rng.uniform(5, 30)))
    _, t, pos = _drag_solve(y0, 0.0, 0.0, cm, g, T=T)
    return dict(
        regime="drag_drop",
        ic={"x0": 0.0, "y0": y0, "vx0": 0.0, "vy0": 0.0, "g": g, "drag_c_over_m": cm,
            "duration_s": T, "fps": FPS},
        scenario=(f"A skydiver is released from rest at height {y0:.0f} m. They fall under gravity "
                  f"(g = {g}) with quadratic air drag: the drag acceleration is -(c/m)|v|v with "
                  f"c/m = {cm:.4f} 1/m. Simulate {T:.2f} seconds of the fall and emit the trajectory."),
        key_standards=[
            ic_standard(0.0, y0, 0.0, 0.0),
            f"Newton's second law holds with gravity and quadratic air drag (g = {g}, c/m = {cm:.5f}).",
            f"The speed approaches the terminal speed set by drag (g = {g}, c/m = {cm:.5f}).",
            "The object falls downward monotonically.",
        ],
        t=t, pos=pos)


# ---------------------------------------------------------------- kepler
def _kepler_orbit(GM, rp, e, T, fps):
    a = rp / (1 - e)
    n = np.sqrt(GM / a ** 3)
    t = grid(T, fps)
    M = n * t
    E = M.copy()
    for _ in range(60):  # Newton iteration on Kepler's equation
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    x = a * (np.cos(E) - e)
    y = a * np.sqrt(1 - e ** 2) * np.sin(E)
    return t, np.stack([x, y], axis=1)


def _kepler(rng, regime, e_lo, e_hi):
    e = float(rng.uniform(e_lo, e_hi))
    rp = float(rng.uniform(1.0, 3.0))
    period = float(rng.uniform(5.0, 8.0))
    a = rp / (1 - e)
    GM = 4 * np.pi ** 2 * a ** 3 / period ** 2
    vp = np.sqrt(GM * (1 + e) / rp)
    T = 1.25 * period
    fps = 240.0  # perihelion passage needs dense sampling
    t, pos = _kepler_orbit(GM, rp, e, T, fps)
    return dict(
        regime=regime,
        ic={"x0": rp, "y0": 0.0, "vx0": 0.0, "vy0": vp, "GM": GM, "duration_s": T, "fps": fps},
        scenario=(f"A small body orbits a fixed central mass located at the origin, attracted by Newtonian "
                  f"gravity with GM = {GM:.4f} m^3/s^2. It starts at position ({rp:.3f}, 0) m with velocity "
                  f"(0, {vp:.4f}) m/s. Simulate {T:.2f} seconds of its motion, sampled at {fps:.0f} frames per "
                  f"second, and emit the trajectory."),
        key_standards=[
            ic_standard(rp, 0.0, 0.0, vp),
            f"The acceleration follows Newton's inverse-square law of gravitation toward the origin (GM = {GM:.5f}).",
            "Equal areas are swept out in equal times about the origin (Kepler's second law).",
            f"The specific orbital energy given by the vis-viva relation is conserved (GM = {GM:.5f}).",
        ],
        t=t, pos=pos, fps=fps)


def regime_orbit_moderate(rng):
    return _kepler(rng, "orbit_moderate", 0.15, 0.40)


def regime_orbit_eccentric(rng):
    return _kepler(rng, "orbit_eccentric", 0.55, 0.75)


# ---------------------------------------------------------------- driven damped oscillator
def _driven(rng, regime, ratio):
    m = float(rng.uniform(0.5, 2.0))
    w0 = float(rng.uniform(2.0, 5.0))
    gamma = float(rng.uniform(0.25, 0.5))
    k, b = m * w0 ** 2, 2 * m * gamma
    F0 = float(rng.uniform(0.5, 2.0))
    wd = ratio * w0
    T = 12.5 / gamma
    A = (F0 / m) / np.sqrt((w0 ** 2 - wd ** 2) ** 2 + (2 * gamma * wd) ** 2)
    delta = np.arctan2(2 * gamma * wd, w0 ** 2 - wd ** 2)
    wq = np.sqrt(w0 ** 2 - gamma ** 2)
    C1 = -A * np.cos(delta)
    C2 = (-A * wd * np.sin(delta) + gamma * C1) / wq
    t = grid(T)
    y = A * np.cos(wd * t - delta) + np.exp(-gamma * t) * (C1 * np.cos(wq * t) + C2 * np.sin(wq * t))
    consts = f"m = {m:.4f}, b = {b:.4f}, k = {k:.4f}, F0 = {F0:.4f}, wd = {wd:.4f}"
    return dict(
        regime=regime,
        ic={"m": m, "b": b, "k": k, "F0": F0, "omega_drive": wd, "y0": 0.0, "vy0": 0.0,
            "duration_s": T, "fps": FPS},
        scenario=(f"A mass m = {m:.3f} kg hangs on a vertical spring (k = {k:.3f} N/m) with linear damping "
                  f"(b = {b:.3f} kg/s) and is driven by an external force F(t) = F0 cos(wd t) with "
                  f"F0 = {F0:.3f} N and wd = {wd:.4f} rad/s. It starts at rest at equilibrium (y = 0). "
                  f"Report the vertical displacement from equilibrium as y (x stays at 0). Simulate "
                  f"{T:.1f} seconds and emit the trajectory."),
        key_standards=[
            ic_standard(0.0, 0.0, 0.0, 0.0),
            f"Newton's second law holds for the damped spring under the driving force ({consts}).",
            f"The steady-state response has the amplitude and phase lag of a driven damped oscillator ({consts}).",
        ],
        t=t, pos=np.stack([np.zeros_like(t), y], axis=1))


def regime_near_resonance(rng):
    return _driven(rng, "near_resonance", float(rng.uniform(0.9, 1.1)))


def regime_off_resonance(rng):
    r = float(rng.uniform(0.4, 0.65)) if rng.uniform(0, 1) < 0.5 else float(rng.uniform(1.5, 2.2))
    return _driven(rng, "off_resonance", r)


# ---------------------------------------------------------------- coupled oscillators
def _coupled(rng, regime, x10, x20):
    m = float(rng.uniform(0.5, 2.0))
    k = float(rng.uniform(2.0, 8.0))
    kc = float(rng.uniform(0.15, 0.6)) * k
    T = float(rng.uniform(20.0, 30.0))
    w1, w2 = np.sqrt(k / m), np.sqrt((k + 2 * kc) / m)
    t = grid(T)
    qs = (x10 + x20) * np.cos(w1 * t)
    qa = (x10 - x20) * np.cos(w2 * t)
    pos = np.stack([(qs + qa) / 2, (qs - qa) / 2], axis=1)
    consts = f"m = {m:.4f}, k = {k:.4f}, kc = {kc:.4f}"
    return dict(
        regime=regime,
        ic={"m": m, "k": k, "kc": kc, "x1_0": x10, "x2_0": x20, "v1_0": 0.0, "v2_0": 0.0,
            "duration_s": T, "fps": FPS},
        scenario=(f"Two identical masses (m = {m:.3f} kg each) slide on a frictionless horizontal track. "
                  f"Each is attached to its own wall by a spring of stiffness k = {k:.3f} N/m, and they are "
                  f"joined to each other by a coupling spring kc = {kc:.3f} N/m. At t = 0 the displacements "
                  f"from equilibrium are x1 = {x10:.3f} m and x2 = {x20:.3f} m, both at rest. Simulate "
                  f"{T:.1f} seconds. Emit the trajectory with pos = [x1, x2]: the first coordinate is the "
                  f"displacement of mass 1 and the second is the displacement of mass 2."),
        key_standards=[
            f"The motion starts from the stated initial state (x1 = {x10:.5f}, x2 = {x20:.5f}, v1 = 0, v2 = 0).",
            f"Both masses obey Newton's second law for the coupled spring system ({consts}).",
            f"The total energy of the two-mass spring system is conserved ({consts}).",
            f"The motion is a superposition of the two normal modes of the system ({consts}).",
        ],
        t=t, pos=pos)


def regime_single_excited(rng):
    return _coupled(rng, "single_excited", float(rng.uniform(0.2, 1.0)), 0.0)


def regime_mixed_modes(rng):
    return _coupled(rng, "mixed_modes", float(rng.uniform(-1.0, 1.0)), float(rng.uniform(-1.0, 1.0)))


SUBCATEGORIES = {
    "drag": ("Object Motion & Kinematics", "Projectile Motion with Drag", [regime_drag_launch, regime_drag_drop]),
    "kepler": ("Gravitation", "Two-Body Orbits", [regime_orbit_moderate, regime_orbit_eccentric]),
    "driven": ("Oscillations", "Driven Damped Oscillator", [regime_near_resonance, regime_off_resonance]),
    "coupled": ("Oscillations", "Coupled Oscillators", [regime_single_excited, regime_mixed_modes]),
}


def make_instance(sub, task_id, rng):
    category, subcategory, regimes = SUBCATEGORIES[sub]
    spec = regimes[task_id % len(regimes)](rng)  # balanced regimes
    return {
        "task_id": task_id,
        "category": category,
        "subcategory": subcategory,
        "regime": spec["regime"],
        "scenario": spec["scenario"],
        "initial_conditions": spec["ic"],
        "physjudge_spec": {
            "prompt_index": f"{sub}-{task_id:05d}",
            "prompt": spec["scenario"],
            "physics": spec["scenario"],
            "basic_standards": {"Objects": ["point mass"], "Event": spec["regime"]},
            "key_standards": spec["key_standards"],
        },
        "golden": {"t": spec["t"].tolist(), "pos": spec["pos"].tolist(), "fps": spec.get("fps", FPS)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", nargs="+", default=list(SUBCATEGORIES))
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--n_test", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for sub in args.subs:
        out = Path("data") / sub
        out.mkdir(parents=True, exist_ok=True)
        for split, n, offset in [("train", args.n_train, 0), ("test", args.n_test, 1)]:
            np_rng = np.random.default_rng([args.seed, offset, list(SUBCATEGORIES).index(sub)])
            path = out / f"{split}_tasks.jsonl"
            with path.open("w") as f:
                for i in range(n):
                    f.write(json.dumps(make_instance(sub, i, np_rng)) + "\n")
            print(f"wrote {n} -> {path}")


if __name__ == "__main__":
    main()

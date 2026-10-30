"""Trajectory primitives for the composite (harder) subcategories.

Each check is a law of motion or a conserved quantity evaluated on the emitted
trajectory, parametrized only by physical constants that appear in the standard
text (g, c/m, GM, m, b, k, ...). No reference trajectory is consulted. Scores
follow the gym-wide form exp(-r / tol) with pass iff r < tol.

Coordinate conventions (stated in each scenario):
  drag, kepler, driven : pos = [x, y] of the single body
  coupled              : pos = [x1, x2], the displacements of the two masses
"""
import numpy as np

from physjudge.types import Trajectory, Verdict

EDGE = 3  # finite-difference samples trimmed at each end


def _soft(r: float, tol: float, diag: dict) -> Verdict:
    r = float(r) if np.isfinite(r) else float("inf")
    return Verdict(passed=r < tol, score=float(np.exp(-r / tol)), diagnostic={**diag, "residual": r, "tol": tol})


def _derivatives(traj: Trajectory):
    """Velocity by central differences and acceleration by the direct second
    difference (x[i+1] - 2x[i] + x[i-1]) / h^2, whose truncation error is 4x
    smaller than differentiating the velocity twice. Falls back to np.gradient
    on non-uniform time grids."""
    t, p = traj.t, traj.pos
    dt = np.diff(t)
    v = np.gradient(p, t, axis=0)
    if np.allclose(dt, dt.mean(), rtol=1e-6, atol=1e-12):
        a = np.empty_like(p)
        a[1:-1] = (p[2:] - 2 * p[1:-1] + p[:-2]) / dt.mean() ** 2
        a[0], a[-1] = a[1], a[-2]
    else:
        a = np.gradient(v, t, axis=0)
    return v, a


def _interior(traj: Trajectory):
    if traj.N < 2 * EDGE + 8:
        return None
    v, a = _derivatives(traj)
    s = slice(EDGE, traj.N - EDGE)
    return traj.t[s], traj.pos[s], v[s], a[s]


def _short() -> Verdict:
    return Verdict(passed=False, score=0.0, diagnostic={"reason": "trajectory too short"})


def _rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


# ---------------------------------------------------------------- drag
def drag_newton_residual(traj: Trajectory, g: float, c_over_m: float, tol: float = 0.05) -> Verdict:
    """a = -g y_hat - (c/m)|v| v. Residual: RMS force-law error relative to g."""
    w = _interior(traj)
    if w is None:
        return _short()
    _, _, v, a = w
    pred = -(c_over_m * np.linalg.norm(v, axis=1))[:, None] * v
    pred[:, 1] -= g
    return _soft(_rms(np.linalg.norm(a - pred, axis=1)) / g, tol,
                 {"g": g, "c_over_m": c_over_m})


def terminal_velocity(traj: Trajectory, g: float, c_over_m: float, tol: float = 0.03) -> Verdict:
    """Tail speed approaches v_t = sqrt(g / (c/m)). Residual: relative error over the last 10%."""
    v = traj.velocity()
    n = max(3, traj.N // 10)
    vt = np.sqrt(g / c_over_m)
    tail = float(np.mean(np.linalg.norm(v[-n - EDGE:-EDGE or None], axis=1)))
    return _soft(abs(tail - vt) / vt, tol, {"tail_speed": tail, "terminal_speed": float(vt)})


# ---------------------------------------------------------------- kepler
def inverse_square_accel(traj: Trajectory, GM: float, tol: float = 0.05) -> Verdict:
    """a = -GM r / |r|^3 about the origin. Residual: RMS error relative to |a|."""
    w = _interior(traj)
    if w is None:
        return _short()
    _, p, _, a = w
    r = np.linalg.norm(p, axis=1)
    if np.any(r < 1e-9):
        return Verdict(passed=False, score=0.0, diagnostic={"reason": "passes through the origin"})
    pred = -GM * p / r[:, None] ** 3
    return _soft(_rms(np.linalg.norm(a - pred, axis=1)) / _rms(np.linalg.norm(pred, axis=1)), tol, {"GM": GM})


def areal_velocity_constant(traj: Trajectory, tol: float = 0.02) -> Verdict:
    """Kepler's second law: h = x vy - y vx is constant. Residual: std(h)/|mean(h)|."""
    w = _interior(traj)
    if w is None:
        return _short()
    _, p, v, _ = w
    h = p[:, 0] * v[:, 1] - p[:, 1] * v[:, 0]
    return _soft(float(np.std(h) / (abs(np.mean(h)) + 1e-12)), tol, {"h_mean": float(np.mean(h))})


def orbital_energy_constant(traj: Trajectory, GM: float, tol: float = 0.02) -> Verdict:
    """Vis-viva: eps = v^2/2 - GM/r is constant. Residual: std(eps)/|mean(eps)|."""
    w = _interior(traj)
    if w is None:
        return _short()
    _, p, v, _ = w
    eps = 0.5 * np.sum(v ** 2, axis=1) - GM / np.linalg.norm(p, axis=1)
    return _soft(float(np.std(eps) / (abs(np.mean(eps)) + 1e-12)), tol, {"eps_mean": float(np.mean(eps))})


# ---------------------------------------------------------------- driven oscillator
def driven_newton_residual(traj: Trajectory, m: float, b: float, k: float, F0: float, wd: float,
                           tol: float = 0.05) -> Verdict:
    """m y'' + b y' + k y = F0 cos(wd t). Residual: RMS force-balance error relative
    to the system's force scale max(F0, RMS(k y)); near resonance the spring and
    inertial forces dwarf F0, so normalizing by F0 alone would amplify harmless
    integration noise."""
    w = _interior(traj)
    if w is None:
        return _short()
    t, p, v, a = w
    res = m * a[:, 1] + b * v[:, 1] + k * p[:, 1] - F0 * np.cos(wd * t)
    scale = max(F0, _rms(k * p[:, 1]))
    return _soft(_rms(res) / scale, tol, {"force_scale": scale})


def _steady_fit(traj: Trajectory, wd: float):
    """Least-squares fit y = C cos(wd t) + S sin(wd t) on the last quarter."""
    n = traj.N // 4
    t, y = traj.t[-n:], traj.pos[-n:, 1]
    M = np.stack([np.cos(wd * t), np.sin(wd * t), np.ones_like(t)], axis=1)
    (C, S, _), *_ = np.linalg.lstsq(M, y, rcond=None)
    return float(np.hypot(C, S)), float(np.arctan2(S, C))


def steady_state_response(traj: Trajectory, m: float, b: float, k: float, F0: float, wd: float,
                          tol: float = 0.05) -> Verdict:
    """Steady-state amplitude A = (F0/m)/sqrt((w0^2-wd^2)^2 + (b wd/m)^2) and phase lag
    delta = atan2(b wd/m, w0^2 - wd^2). Residual: relative amplitude error + phase error / pi."""
    if traj.N < 40:
        return _short()
    w0sq = k / m
    A = (F0 / m) / np.sqrt((w0sq - wd ** 2) ** 2 + (b * wd / m) ** 2)
    delta = np.arctan2(b * wd / m, w0sq - wd ** 2)
    amp, phase = _steady_fit(traj, wd)
    dphi = abs((phase - delta + np.pi) % (2 * np.pi) - np.pi)
    r = abs(amp - A) / A + dphi / np.pi
    return _soft(r, tol, {"amplitude": amp, "expected_amplitude": float(A),
                          "phase_lag": phase, "expected_phase_lag": float(delta)})


# ---------------------------------------------------------------- coupled oscillators
def coupled_newton_residual(traj: Trajectory, m: float, k: float, kc: float, tol: float = 0.05) -> Verdict:
    """m x1'' = -k x1 - kc (x1 - x2);  m x2'' = -k x2 - kc (x2 - x1)."""
    w = _interior(traj)
    if w is None:
        return _short()
    _, p, _, a = w
    x1, x2 = p[:, 0], p[:, 1]
    r1 = m * a[:, 0] + k * x1 + kc * (x1 - x2)
    r2 = m * a[:, 1] + k * x2 + kc * (x2 - x1)
    scale = _rms(np.stack([k * x1 + kc * (x1 - x2), k * x2 + kc * (x2 - x1)])) + 1e-12
    return _soft(_rms(np.stack([r1, r2])) / scale, tol, {})


def coupled_energy_constant(traj: Trajectory, m: float, k: float, kc: float, tol: float = 0.02) -> Verdict:
    """E = m(v1^2+v2^2)/2 + k(x1^2+x2^2)/2 + kc(x1-x2)^2/2 is constant."""
    w = _interior(traj)
    if w is None:
        return _short()
    _, p, v, _ = w
    x1, x2 = p[:, 0], p[:, 1]
    E = 0.5 * m * (v[:, 0] ** 2 + v[:, 1] ** 2) + 0.5 * k * (x1 ** 2 + x2 ** 2) + 0.5 * kc * (x1 - x2) ** 2
    return _soft(float(np.std(E) / (abs(np.mean(E)) + 1e-12)), tol, {"E_mean": float(np.mean(E))})


def normal_mode_content(traj: Trajectory, m: float, k: float, kc: float, tol: float = 0.05) -> Verdict:
    """The symmetric coordinate x1+x2 oscillates only at w1 = sqrt(k/m) and the
    antisymmetric x1-x2 only at w2 = sqrt((k+2kc)/m). Residual: fraction of each
    coordinate's variance not explained by a sinusoid at its mode frequency."""
    if traj.N < 40:
        return _short()
    t = traj.t
    w1, w2 = np.sqrt(k / m), np.sqrt((k + 2 * kc) / m)
    unexplained = []
    for q, w in ((traj.pos[:, 0] + traj.pos[:, 1], w1), (traj.pos[:, 0] - traj.pos[:, 1], w2)):
        var = float(np.var(q))
        if var < 1e-12:
            continue  # mode not excited
        M = np.stack([np.cos(w * t), np.sin(w * t), np.ones_like(t)], axis=1)
        coef, *_ = np.linalg.lstsq(M, q, rcond=None)
        unexplained.append(float(np.var(q - M @ coef) / var))
    if not unexplained:
        return Verdict(passed=False, score=0.0, diagnostic={"reason": "no motion"})
    return _soft(max(unexplained), tol, {"w1": float(w1), "w2": float(w2), "unexplained": unexplained})


# ---------------------------------------------------------------- initial state
def initial_state(traj: Trajectory, p0, v0, tol: float = 0.05) -> Verdict:
    """The trajectory starts at the stated position and velocity. The initial
    velocity is the slope of a cubic fit to the first eight samples. Residual: |dp|/L + |dv|/V with L, V the trajectory's position and
    speed scales."""
    if traj.N < 8:
        return _short()
    p0, v0 = np.asarray(p0, float), np.asarray(v0, float)
    t = traj.t[:8] - traj.t[0]
    vfit = np.array([np.polyfit(t, traj.pos[:8, d], 3)[2] for d in range(2)])
    L = float(np.max(np.ptp(traj.pos[:, :2], axis=0))) + 1e-9
    V = float(np.max(np.linalg.norm(traj.velocity()[:, :2], axis=1))) + 1e-9
    r = np.linalg.norm(traj.pos[0, :2] - p0) / L + np.linalg.norm(vfit - v0) / V
    return _soft(r, tol, {"p_start": traj.pos[0, :2].tolist(), "v_start": vfit.tolist()})

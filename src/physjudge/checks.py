"""Trajectory predicate primitives. Each returns a Verdict."""
import numpy as np
from .types import Trajectory, Verdict

G = 9.81


def _conf_mask(traj: Trajectory, gate: float = 0.5) -> np.ndarray:
    return traj.confidence >= gate


def at_rest(traj: Trajectory, eps_frac: float = 0.02) -> Verdict:
    span = float(np.linalg.norm(traj.pos.max(0) - traj.pos.min(0)))
    scale = float(np.linalg.norm(traj.pos.std(0))) + 1e-6
    score = 1.0 - min(span / (scale * 100 + 1e-6), 1.0)
    passed = span < eps_frac * (np.abs(traj.pos).max() + 1e-6)
    return Verdict(passed=passed, score=score, diagnostic={"span": span})


def straight_line(traj: Trajectory, axis: str = "y", tol: float = 0.10) -> Verdict:
    """Motion is along the named axis only — perpendicular travel must be small relative to primary travel."""
    primary_idx = {"x": 0, "y": 1, "z": 2}[axis]
    other_indices = [i for i in range(traj.D) if i != primary_idx]
    primary_disp = abs(float(traj.pos[-1, primary_idx] - traj.pos[0, primary_idx])) + 1e-6
    perp_disp = max(abs(float(traj.pos[:, i].max() - traj.pos[:, i].min())) for i in other_indices)
    ratio = perp_disp / primary_disp
    passed = ratio < tol
    score = float(np.exp(-ratio / max(tol, 1e-3)))
    return Verdict(passed=passed, score=score,
                   diagnostic={"perp_disp": perp_disp, "primary_disp": primary_disp, "ratio": ratio})


def constant_speed(traj: Trajectory, cv_tol: float = 0.15) -> Verdict:
    v = traj.velocity()
    speed = np.linalg.norm(v, axis=1)
    mean = float(speed.mean())
    cv = float(speed.std() / (mean + 1e-6))
    passed = cv < cv_tol
    score = float(np.exp(-cv / max(cv_tol, 1e-3)))
    return Verdict(passed=passed, score=score, diagnostic={"speed_mean": mean, "cv": cv})


def free_fall(traj: Trajectory, axis: int = 1, tol: float = 0.30) -> Verdict:
    """Acceleration along given axis ≈ -g (y-up). Use only on physically-scaled trajectories."""
    a = traj.acceleration()[:, axis]
    a_mean = float(a.mean())
    err = abs(a_mean - (-G)) / G
    passed = err < tol
    score = float(np.exp(-err / max(tol, 1e-3)))
    return Verdict(passed=passed, score=score, diagnostic={"a_mean": a_mean, "expected": -G, "rel_err": err})


def gravity_like(traj: Trajectory, axis: int = 1, sign: int = -1, std_tol: float = 0.4) -> Verdict:
    """Scale-invariant: axis-acceleration is approximately constant with the given sign."""
    a = traj.acceleration()[:, axis]
    a_mean = float(a.mean())
    a_std = float(a.std())
    rel_std = a_std / (abs(a_mean) + 1e-6)
    correct_sign = (np.sign(a_mean) == sign) if sign != 0 else True
    passed = bool(correct_sign and rel_std < std_tol)
    score = (1.0 if correct_sign else 0.0) * float(np.exp(-rel_std / max(std_tol, 1e-3)))
    return Verdict(passed=passed, score=score,
                   diagnostic={"a_mean": a_mean, "a_std": a_std, "rel_std": rel_std, "correct_sign": correct_sign})


def monotonic(traj: Trajectory, axis: int = 1, direction: str = "up") -> Verdict:
    coord = traj.pos[:, axis]
    diffs = np.diff(coord)
    sign = 1 if direction == "up" else -1
    frac_correct = float((np.sign(diffs) == sign).mean())
    passed = frac_correct > 0.9
    return Verdict(passed=passed, score=frac_correct, diagnostic={"frac_correct": frac_correct})


def energy_non_increasing(traj: Trajectory, mass: float = 1.0, tol: float = 0.05) -> Verdict:
    """Total mechanical energy should not grow (within tolerance) for an unforced or dissipative system."""
    v = traj.velocity()
    ke = 0.5 * mass * np.sum(v * v, axis=1)
    pe = mass * G * traj.pos[:, 1] if traj.D >= 2 else np.zeros_like(ke)
    e = ke + pe
    e0 = float(e[0]) + 1e-6
    max_gain = float((e - e[0]).max() / e0)
    passed = max_gain < tol
    score = float(np.exp(-max(0.0, max_gain) / max(tol, 1e-3)))
    return Verdict(passed=passed, score=score, diagnostic={"e0": e0, "max_rel_gain": max_gain})


def bounces(traj: Trajectory, axis: int = 1, min_count: int = 1) -> Verdict:
    v_axis = traj.velocity()[:, axis]
    sign_changes = int(np.sum(np.diff(np.sign(v_axis)) != 0))
    passed = sign_changes >= min_count
    return Verdict(passed=passed, score=min(sign_changes / max(min_count, 1), 1.0),
                   diagnostic={"sign_changes": sign_changes})


def elevation_unchanged(traj: Trajectory, tol_frac: float = 0.10) -> Verdict:
    """Vertical span is small relative to horizontal travel."""
    horiz_span = float(traj.pos[:, 0].max() - traj.pos[:, 0].min()) + 1e-6
    vert_span = float(traj.pos[:, 1].max() - traj.pos[:, 1].min())
    ratio = vert_span / horiz_span
    passed = ratio < tol_frac
    score = float(np.exp(-ratio / max(tol_frac, 1e-3)))
    return Verdict(passed=passed, score=score, diagnostic={"vert_span": vert_span, "horiz_span": horiz_span, "ratio": ratio})


def straight_path_any_axis(traj: Trajectory, tol: float = 0.08) -> Verdict:
    """PCA: motion concentrated on one direction. Returns small ratio of secondary to primary singular value."""
    centered = traj.pos - traj.pos.mean(0)
    _, sv, _ = np.linalg.svd(centered, full_matrices=False)
    if sv.shape[0] < 2 or sv[0] < 1e-6:
        return Verdict(passed=False, score=0.0, diagnostic={"reason": "degenerate"})
    ratio = float(sv[1] / sv[0])
    passed = ratio < tol
    score = float(np.exp(-ratio / max(tol, 1e-3)))
    return Verdict(passed=passed, score=score, diagnostic={"sv_ratio": ratio, "sv0": float(sv[0]), "sv1": float(sv[1])})


def no_lateral_deviation(traj: Trajectory, tol: float = 0.05) -> Verdict:
    """Compute lateral deviation perpendicular to mean direction of travel."""
    centered = traj.pos - traj.pos.mean(0)
    direction = traj.pos[-1] - traj.pos[0]
    norm = float(np.linalg.norm(direction)) + 1e-6
    direction /= norm
    perp = centered - np.outer(centered @ direction, direction)
    lateral = float(np.linalg.norm(perp, axis=1).max())
    ratio = lateral / norm
    passed = ratio < tol
    score = float(np.exp(-ratio / max(tol, 1e-3)))
    return Verdict(passed=passed, score=score, diagnostic={"lateral_max": lateral, "path_len": norm, "ratio": ratio})


def parabolic_fit(traj: Trajectory, residual_tol: float = 0.05) -> Verdict:
    """Fit y as a quadratic of x; small normalized residual indicates a parabolic arc."""
    x = traj.pos[:, 0]
    y = traj.pos[:, 1]
    if x.max() - x.min() < 1e-6:
        return Verdict(passed=False, score=0.0, diagnostic={"reason": "no horizontal span"})
    coef = np.polyfit(x, y, 2)
    y_hat = np.polyval(coef, x)
    rss = float(np.sum((y - y_hat) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2)) + 1e-9
    r2 = 1.0 - rss / tss
    rel_resid = float(np.sqrt(rss / len(x)) / (y.max() - y.min() + 1e-6))
    passed = (r2 > 0.97) and (rel_resid < residual_tol) and (coef[0] < 0)
    score = float(max(min(r2, 1.0), 0.0))
    return Verdict(passed=passed, score=score,
                   diagnostic={"r2": r2, "rel_residual": rel_resid, "a": float(coef[0])})


def smooth_velocity(traj: Trajectory, jerk_tol: float = 0.5) -> Verdict:
    """Bounded normalized jerk (3rd derivative of position) → no abrupt speed changes."""
    a = traj.acceleration()
    j = np.gradient(a, traj.t, axis=0)
    j_norm = np.linalg.norm(j, axis=1)
    speed_scale = float(np.linalg.norm(traj.velocity(), axis=1).mean()) + 1e-6
    rel = float(j_norm.mean() / speed_scale)
    passed = rel < jerk_tol
    score = float(np.exp(-rel / max(jerk_tol, 1e-3)))
    return Verdict(passed=passed, score=score, diagnostic={"jerk_rel": rel})


def apex_detection(traj: Trajectory, axis: int = 1) -> Verdict:
    """Trajectory rises to a clear maximum then descends along the named axis."""
    y = traj.pos[:, axis]
    idx_max = int(np.argmax(y))
    if idx_max <= 1 or idx_max >= traj.N - 2:
        return Verdict(passed=False, score=0.0, diagnostic={"idx_max": idx_max, "reason": "apex at boundary"})
    rising = y[:idx_max + 1]
    falling = y[idx_max:]
    rise_mono = float((np.diff(rising) >= -1e-3).mean())
    fall_mono = float((np.diff(falling) <= 1e-3).mean())
    score = (rise_mono + fall_mono) / 2.0
    passed = score > 0.85
    return Verdict(passed=passed, score=score,
                   diagnostic={"idx_max": idx_max, "rise_mono": rise_mono, "fall_mono": fall_mono})


def no_direction_changes(traj: Trajectory, cos_threshold: float = 0.95) -> Verdict:
    """Velocity direction stays nearly constant — no sharp turns."""
    v = traj.velocity()
    vn = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-6)
    cos = float(np.sum(vn[1:] * vn[:-1], axis=1).mean())
    passed = cos > cos_threshold
    score = float(max(0.0, (cos - 0.5) / 0.5))
    return Verdict(passed=passed, score=score, diagnostic={"mean_cos": cos})


def circular_motion(traj: Trajectory, radius_cv_tol: float = 0.10) -> Verdict:
    """Radius from centroid is approximately constant."""
    centroid = traj.pos.mean(0)
    r = np.linalg.norm(traj.pos - centroid, axis=1)
    cv = float(r.std() / (r.mean() + 1e-6))
    passed = cv < radius_cv_tol
    score = float(np.exp(-cv / max(radius_cv_tol, 1e-3)))
    return Verdict(passed=passed, score=score, diagnostic={"radius_mean": float(r.mean()), "cv": cv})


def perpetual_motion_flag(traj: Trajectory, mass: float = 1.0, threshold: float = 0.10) -> Verdict:
    """Returns passed=True iff the trajectory exhibits energy creation (an anti-physics positive flag)."""
    v = traj.velocity()
    ke = 0.5 * mass * np.sum(v * v, axis=1)
    pe = mass * G * traj.pos[:, 1] if traj.D >= 2 else np.zeros_like(ke)
    e = ke + pe
    e0 = float(e[0]) + 1e-6
    rel_gain = float((e - e[0]).max() / e0)
    passed = rel_gain > threshold
    return Verdict(passed=passed, score=min(max(rel_gain, 0.0), 1.0),
                   diagnostic={"rel_energy_gain": rel_gain})

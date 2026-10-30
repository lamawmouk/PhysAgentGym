"""Domain-specific compiler that layers physics-gym rules on top of physjudge's
RuleBasedCompiler. Reasons to extend rather than rephrase tasks:
  - apex_detection must beat monotonic up/down on phrases like "rises and then falls".
  - energy conservation has no rule in physjudge; map to energy_non_increasing.
  - damping/oscillator scenarios need amplitude-envelope and final-at-rest predicates
    not present in physjudge's gravity-and-projectile-oriented primitives.

Compilation is hybrid: gym rules, then physjudge rules, then (if configured) an LLM
that emits a constrained DSL tree. Without an LLM, an unmatched standard raises at
compile time rather than silently becoming a no-op.
"""
import re

import numpy as np

from physjudge import checks
from physjudge.compiler import RuleBasedCompiler
from physjudge.dsl import PRIMITIVE_REGISTRY, PrimitiveSpec
from physjudge.types import Assertion, Scenario, Trajectory, Verdict

from . import harder_checks as hc


def amplitude_decay(traj: Trajectory, axis: int = 1, tol: float = 0.10) -> Verdict:
    """Peak-to-peak amplitude in the second half should be smaller than the first
    half (within tolerance). Robust to mean offset; no smoothing required for
    closed-form damped oscillators or exponential decays."""
    coord = traj.pos[:, axis]
    half = traj.N // 2
    if half < 4:
        return Verdict(passed=False, score=0.0, diagnostic={"reason": "trajectory too short"})
    a1 = float(coord[:half].max() - coord[:half].min())
    a2 = float(coord[half:].max() - coord[half:].min())
    ratio = a2 / (a1 + 1e-9)
    passed = ratio < (1.0 - tol)
    score = float(np.clip(1.0 - ratio, 0.0, 1.0))
    return Verdict(passed=passed, score=score,
                   diagnostic={"amp_first_half": a1, "amp_second_half": a2, "ratio": ratio})


def final_at_rest(traj: Trajectory, axis: int = 1, tail_frac: float = 0.20,
                  vel_tol: float = 0.10) -> Verdict:
    """The tail of the trajectory exhibits speed below tolerance — i.e. the
    object 'comes to rest' even if it was moving earlier."""
    v = traj.velocity()
    speed = np.linalg.norm(v, axis=1)
    n_tail = max(2, int(traj.N * tail_frac))
    tail_speed = float(speed[-n_tail:].mean())
    peak_speed = float(speed.max()) + 1e-9
    rel = tail_speed / peak_speed
    passed = rel < vel_tol
    score = float(np.exp(-rel / max(vel_tol, 1e-3)))
    return Verdict(passed=passed, score=score,
                   diagnostic={"tail_speed": tail_speed, "peak_speed": peak_speed, "rel": rel})


_CONST = re.compile(r"\b([A-Za-z][A-Za-z0-9]*(?:/[a-z])?)\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def _consts(standard: str) -> dict:
    """Physical constants stated in a standard, e.g. "(g = 9.81, c/m = 0.012)"."""
    return {k.replace("/", "_over_"): float(v) for k, v in _CONST.findall(standard)}


def _with_consts(fn, *names):
    def build(s, sc):
        c = _consts(s)
        return lambda t: fn(t, *(c[n] for n in names))
    return build


# Composite subcategories (drag, kepler, driven, coupled): law-of-motion checks
# parametrized by constants in the standard text. Matched before the generic rules.
def _initial_state(s, sc):
    c = _consts(s)
    if "x1" in c:  # coupled: pos = [x1, x2]
        p0, v0 = (c["x1"], c["x2"]), (c["v1"], c["v2"])
    else:
        p0, v0 = (c["x0"], c["y0"]), (c["vx0"], c["vy0"])
    return lambda t: hc.initial_state(t, p0, v0)


# Expose the gym primitives to the LLM compiler's DSL.
PRIMITIVE_REGISTRY.setdefault("amplitude_decay", PrimitiveSpec(
    "amplitude_decay", amplitude_decay,
    "Peak-to-peak amplitude on the axis shrinks from the first to the second half (damping).",
    {"axis": {"type": "integer", "default": 1}, "tol": {"type": "number", "default": 0.10}},
))
PRIMITIVE_REGISTRY.setdefault("final_at_rest", PrimitiveSpec(
    "final_at_rest", final_at_rest,
    "Tail speed is small relative to peak speed: the object comes to rest by the end.",
    {"tail_frac": {"type": "number", "default": 0.20}, "vel_tol": {"type": "number", "default": 0.10}},
))


class UnmatchedStandardError(ValueError):
    pass


_HARDER_RULES: list[tuple[re.Pattern, str, callable]] = [
    (re.compile(r"\bstarts\s+from\s+the\s+stated\s+initial\s+state\b", re.IGNORECASE),
     "starts from the stated initial state", _initial_state),
    (re.compile(r"\bquadratic\s+air\s+drag\b(?!.*terminal)", re.IGNORECASE),
     "Newton's law with gravity + quadratic drag", _with_consts(hc.drag_newton_residual, "g", "c_over_m")),
    (re.compile(r"\bterminal\s+speed\b", re.IGNORECASE),
     "approaches terminal speed sqrt(g/(c/m))", _with_consts(hc.terminal_velocity, "g", "c_over_m")),
    (re.compile(r"\binverse[- ]square\b", re.IGNORECASE),
     "inverse-square central acceleration", _with_consts(hc.inverse_square_accel, "GM")),
    (re.compile(r"\bequal\s+areas\b", re.IGNORECASE),
     "constant areal velocity (Kepler II)", lambda s, sc: lambda t: hc.areal_velocity_constant(t)),
    (re.compile(r"\bvis[- ]viva\b|\bspecific\s+orbital\s+energy\b", re.IGNORECASE),
     "vis-viva energy conserved", _with_consts(hc.orbital_energy_constant, "GM")),
    (re.compile(r"\bdriving\s+force\b", re.IGNORECASE),
     "driven damped oscillator ODE", _with_consts(hc.driven_newton_residual, "m", "b", "k", "F0", "wd")),
    (re.compile(r"\bsteady[- ]state\b", re.IGNORECASE),
     "steady-state amplitude and phase lag", _with_consts(hc.steady_state_response, "m", "b", "k", "F0", "wd")),
    (re.compile(r"\bcoupled\s+spring\s+system\b", re.IGNORECASE),
     "coupled oscillator equations of motion", _with_consts(hc.coupled_newton_residual, "m", "k", "kc")),
    (re.compile(r"\btwo[- ]mass\s+spring\s+system\b", re.IGNORECASE),
     "coupled oscillator energy conserved", _with_consts(hc.coupled_energy_constant, "m", "k", "kc")),
    (re.compile(r"\bnormal\s+modes?\b", re.IGNORECASE),
     "normal-mode content", _with_consts(hc.normal_mode_content, "m", "k", "kc")),
]


_PRIORITY_RULES: list[tuple[re.Pattern, str, callable]] = [
    (re.compile(r"\b(amplitude\s+(decreases?|decays?|diminish(es)?)|"
                r"oscillation\s+amplitude\s+decreases?|"
                r"amplitude.*?over\s+time|"
                r"envelope\s+(decreases?|decays?))\b",
                re.IGNORECASE),
     "amplitude decay over time",
     lambda s, sc: lambda t: amplitude_decay(t, axis=1, tol=0.10)),

    (re.compile(r"\b(comes?\s+to\s+rest\s+(at\s+the\s+end|eventually|finally)|"
                r"eventually\s+(comes?\s+to\s+)?(rest|stops?)|"
                r"settles?\s+(at|to|near)\s+(rest|equilibrium))\b",
                re.IGNORECASE),
     "final at rest (tail of trajectory is still)",
     lambda s, sc: lambda t: final_at_rest(t, axis=1, tail_frac=0.20, vel_tol=0.10)),

    (re.compile(r"\b(reaches?\s+(a\s+)?(clear\s+)?(apex|peak|maximum\s+height)|"
                r"(rises?|ascends?)\s+to\s+(a\s+|the\s+)?(clear\s+)?(apex|peak|maximum\s+height)|"
                r"rises?\s+(and|then)\s+(falls?|descends?)|"
                r"upward\s+and\s+downward|throw\s+and\s+catch|"
                r"ascent\s+to\s+the\s+apex(?:\s+and\s+a\s+descent)?)\b",
                re.IGNORECASE),
     "apex (rise then fall)",
     lambda s, sc: lambda t: checks.apex_detection(t, axis=1)),

    (re.compile(r"\b(constant\s+downward\s+acceleration|"
                r"acceleration\s+is\s+approximately\s+constant\s+downward|"
                r"gravity[- ]?like|"
                r"under\s+gravity|"
                r"falls?\s+under\s+gravity|"
                r"accelerates?\s+downward)\b",
                re.IGNORECASE),
     "gravity-like (constant downward accel)",
     lambda s, sc: lambda t: checks.gravity_like(t, axis=1, sign=-1, std_tol=0.5)),

    (re.compile(r"\b(energy\s+is\s+conserved|"
                r"mechanical\s+energy\s+is\s+conserved|"
                r"conservation\s+of\s+(mechanical\s+)?energy|"
                r"total\s+(mechanical\s+)?energy\s+is\s+(conserved|preserved)|"
                r"energy\s+(non[- ]?increasing|does\s+not\s+grow))\b",
                re.IGNORECASE),
     "energy non-increasing (conservation)",
     lambda s, sc: lambda t: checks.energy_non_increasing(t, mass=1.0, tol=0.05)),
]


class PhysicsGymCompiler:
    def __init__(self, llm=None):
        """llm: optional physjudge LLMCompiler / CLILLMCompiler / MockLLMCompiler
        used only for standards that no hand-written rule matches."""
        self.rules = RuleBasedCompiler()
        self.llm = llm

    def compile(self, standard: str, scenario: Scenario) -> Assertion:
        for pat, desc, fn in _HARDER_RULES + _PRIORITY_RULES:
            if pat.search(standard):
                return Assertion(
                    standard_text=standard,
                    check_fn=fn(standard, scenario),
                    description=desc,
                    source="rule",
                )
        a = self.rules.compile(standard, scenario)
        if a.description != "unmatched":
            return a
        if self.llm is None:
            raise UnmatchedStandardError(
                f"no rule matches standard {standard!r}; add a rule or enable the "
                "LLM fallback (PhysicsGymCompiler(llm=...), scripts/rollout.py --llm_compiler)")
        return self.llm.compile(standard, scenario)

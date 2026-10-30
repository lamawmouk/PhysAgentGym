"""Constrained DSL for compiled physical-realism checks.

A DSL node is one of:
  primitive: {"primitive": <name>, "args": {...}}
  composition: {"op": "and"|"or"|"not", "children": [<node>, ...]}

The primitive registry maps stable names to functions on Trajectory; the
evaluator dispatches without executing arbitrary code, so the LLM compiler
cannot inject Python.
"""
from dataclasses import dataclass, field
from typing import Any, Callable
from . import checks
from .types import Trajectory, Verdict


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    fn: Callable[..., Verdict]
    description: str
    args_schema: dict[str, Any] = field(default_factory=dict)


PRIMITIVE_REGISTRY: dict[str, PrimitiveSpec] = {
    "at_rest": PrimitiveSpec(
        "at_rest", checks.at_rest,
        "Object is stationary across the clip.",
        {"eps_frac": {"type": "number", "default": 0.02}},
    ),
    "straight_line": PrimitiveSpec(
        "straight_line", checks.straight_line,
        "Motion is along a single named axis (x or y).",
        {"axis": {"type": "string", "enum": ["x", "y", "z"], "default": "y"},
         "tol": {"type": "number", "default": 0.10}},
    ),
    "constant_speed": PrimitiveSpec(
        "constant_speed", checks.constant_speed,
        "Speed is approximately constant (low coefficient-of-variation).",
        {"cv_tol": {"type": "number", "default": 0.18}},
    ),
    "free_fall": PrimitiveSpec(
        "free_fall", checks.free_fall,
        "Acceleration on the named axis ≈ -9.81 m/s² (y is up). Use only when world-scale is known.",
        {"axis": {"type": "integer", "default": 1},
         "tol": {"type": "number", "default": 0.35}},
    ),
    "gravity_like": PrimitiveSpec(
        "gravity_like", checks.gravity_like,
        "Scale-invariant: axis-acceleration is approximately constant with the given sign (-1=down).",
        {"axis": {"type": "integer", "default": 1},
         "sign": {"type": "integer", "enum": [-1, 0, 1], "default": -1},
         "std_tol": {"type": "number", "default": 0.4}},
    ),
    "monotonic": PrimitiveSpec(
        "monotonic", checks.monotonic,
        "Position on the named axis monotonically increases or decreases.",
        {"axis": {"type": "integer", "default": 1},
         "direction": {"type": "string", "enum": ["up", "down"], "default": "up"}},
    ),
    "energy_non_increasing": PrimitiveSpec(
        "energy_non_increasing", checks.energy_non_increasing,
        "Total mechanical energy never grows beyond a small tolerance.",
        {"mass": {"type": "number", "default": 1.0},
         "tol": {"type": "number", "default": 0.05}},
    ),
    "bounces": PrimitiveSpec(
        "bounces", checks.bounces,
        "At least N sign-changes in axis-velocity (proxy for bounces).",
        {"axis": {"type": "integer", "default": 1},
         "min_count": {"type": "integer", "default": 1}},
    ),
    "elevation_unchanged": PrimitiveSpec(
        "elevation_unchanged", checks.elevation_unchanged,
        "Vertical span small relative to horizontal travel.",
        {"tol_frac": {"type": "number", "default": 0.10}},
    ),
    "straight_path_any_axis": PrimitiveSpec(
        "straight_path_any_axis", checks.straight_path_any_axis,
        "Motion concentrated on one direction (PCA secondary/primary singular value ratio).",
        {"tol": {"type": "number", "default": 0.08}},
    ),
    "no_lateral_deviation": PrimitiveSpec(
        "no_lateral_deviation", checks.no_lateral_deviation,
        "Lateral deviation perpendicular to mean travel direction is small.",
        {"tol": {"type": "number", "default": 0.05}},
    ),
    "parabolic_fit": PrimitiveSpec(
        "parabolic_fit", checks.parabolic_fit,
        "y(x) fits a downward-opening quadratic with high R^2.",
        {"residual_tol": {"type": "number", "default": 0.07}},
    ),
    "smooth_velocity": PrimitiveSpec(
        "smooth_velocity", checks.smooth_velocity,
        "Bounded normalized jerk — no abrupt speed changes.",
        {"jerk_tol": {"type": "number", "default": 0.6}},
    ),
    "apex_detection": PrimitiveSpec(
        "apex_detection", checks.apex_detection,
        "Trajectory rises monotonically to a clear maximum then descends monotonically.",
        {"axis": {"type": "integer", "default": 1}},
    ),
    "no_direction_changes": PrimitiveSpec(
        "no_direction_changes", checks.no_direction_changes,
        "Velocity direction stays nearly constant.",
        {"cos_threshold": {"type": "number", "default": 0.92}},
    ),
    "circular_motion": PrimitiveSpec(
        "circular_motion", checks.circular_motion,
        "Radial distance from centroid is approximately constant.",
        {"radius_cv_tol": {"type": "number", "default": 0.12}},
    ),
    "perpetual_motion_flag": PrimitiveSpec(
        "perpetual_motion_flag", checks.perpetual_motion_flag,
        "TRUE iff trajectory exhibits energy creation (an anti-physics positive flag).",
        {"mass": {"type": "number", "default": 1.0},
         "threshold": {"type": "number", "default": 0.10}},
    ),
}


class DSLError(ValueError):
    pass


def evaluate_dsl(node: dict, trajectory: Trajectory) -> Verdict:
    if not isinstance(node, dict):
        raise DSLError(f"node must be dict, got {type(node).__name__}")

    if "primitive" in node:
        name = node["primitive"]
        spec = PRIMITIVE_REGISTRY.get(name)
        if spec is None:
            raise DSLError(f"unknown primitive: {name!r}")
        args = node.get("args") or {}
        try:
            return spec.fn(trajectory, **args)
        except TypeError as e:
            raise DSLError(f"bad args for {name}: {e}") from e

    if "op" in node:
        op = node["op"]
        children = node.get("children") or []
        if not children:
            raise DSLError("composition requires non-empty children")
        verdicts = [evaluate_dsl(c, trajectory) for c in children]
        if op == "and":
            passed = all(v.passed for v in verdicts)
            score = float(min(v.score for v in verdicts))
        elif op == "or":
            passed = any(v.passed for v in verdicts)
            score = float(max(v.score for v in verdicts))
        elif op == "not":
            if len(verdicts) != 1:
                raise DSLError("'not' takes exactly one child")
            passed = not verdicts[0].passed
            score = 1.0 - verdicts[0].score
        else:
            raise DSLError(f"unknown op: {op!r}")
        return Verdict(passed=passed, score=score,
                       diagnostic={"op": op, "children": [v.diagnostic for v in verdicts]})

    raise DSLError(f"node has neither 'primitive' nor 'op': {node!r}")


def schema_for_prompt() -> str:
    """Render the registry as a compact LLM-prompt string."""
    lines = ["Available primitives (call by name with the listed args):"]
    for name, spec in PRIMITIVE_REGISTRY.items():
        args = ", ".join(
            f"{k}: {v.get('type','?')}={v.get('default','?')}"
            for k, v in spec.args_schema.items()
        )
        lines.append(f"  {name}({args}) — {spec.description}")
    lines.append(
        "Composition: {\"op\":\"and|or|not\", \"children\":[<node>...]}.  "
        "Primitive call: {\"primitive\": <name>, \"args\": {...}}."
    )
    return "\n".join(lines)

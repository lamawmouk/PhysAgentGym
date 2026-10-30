"""Hybrid compilation: rules first, LLM DSL fallback only for unmatched standards."""

import numpy as np
import pytest


from physjudge import MockLLMCompiler
from physjudge.types import Scenario, Trajectory
from physics_gym.utils.compiler import PhysicsGymCompiler, UnmatchedStandardError

SC = Scenario("t", "", "", {}, [], "", "")
UNMATCHED = "By the final frames the mass has stopped moving entirely."


def _decay():
    t = np.linspace(0, 4, 241)
    pos = np.stack([np.zeros_like(t), 3.0 * np.exp(-1.5 * t)], axis=1)
    return Trajectory(t=t, pos=pos, confidence=np.ones_like(t), object_id="m", fps=60.0)


def test_rule_takes_precedence_over_llm():
    llm = MockLLMCompiler({})                      # would raise KeyError if consulted
    a = PhysicsGymCompiler(llm=llm).compile("Acceleration is gravity-like.", SC)
    assert a.source == "rule"


def test_unmatched_raises_without_llm():
    with pytest.raises(UnmatchedStandardError):
        PhysicsGymCompiler().compile(UNMATCHED, SC)


def test_unmatched_falls_through_to_llm_with_gym_primitive():
    llm = MockLLMCompiler({UNMATCHED: {"primitive": "final_at_rest", "args": {}}})
    a = PhysicsGymCompiler(llm=llm).compile(UNMATCHED, SC)
    assert a.source == "llm"
    assert a.check_fn(_decay()).passed

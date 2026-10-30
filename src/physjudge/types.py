from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np


@dataclass
class Scenario:
    prompt_index: str
    prompt: str
    physics: str
    basic_standards: dict
    key_standards: list[str]
    category: str
    subcategory: str
    detailed: Optional[str] = None
    detailed_long: Optional[str] = None


@dataclass
class Trajectory:
    """Per-object position trace. y is up (physical convention), not pixel-down."""
    t: np.ndarray            # (N,) seconds
    pos: np.ndarray          # (N, D) where D in {2, 3}
    confidence: np.ndarray   # (N,) in [0, 1]
    object_id: str
    fps: float

    def __post_init__(self):
        assert self.pos.ndim == 2 and self.pos.shape[0] == self.t.shape[0]
        assert self.confidence.shape == self.t.shape

    @property
    def D(self) -> int:
        return self.pos.shape[1]

    @property
    def N(self) -> int:
        return self.t.shape[0]

    def velocity(self) -> np.ndarray:
        return np.gradient(self.pos, self.t, axis=0)

    def acceleration(self) -> np.ndarray:
        return np.gradient(self.velocity(), self.t, axis=0)


@dataclass
class Verdict:
    passed: bool
    score: float                       # in [0, 1]; finer-grained than pass/fail
    diagnostic: dict = field(default_factory=dict)


CheckFn = Callable[[Trajectory], Verdict]


@dataclass
class Assertion:
    standard_text: str
    check_fn: CheckFn
    description: str
    source: str = "rule"               # "rule" or "llm" — which compiler emitted it


@dataclass
class JudgeResult:
    scenario: Scenario
    assertions: list[Assertion]
    verdicts: list[Verdict]
    coverage: float                    # fraction of frames with confidence > gate

    @property
    def pass_rate(self) -> float:
        if not self.verdicts:
            return 0.0
        return sum(v.passed for v in self.verdicts) / len(self.verdicts)

    @property
    def mean_score(self) -> float:
        if not self.verdicts:
            return 0.0
        return float(np.mean([v.score for v in self.verdicts]))

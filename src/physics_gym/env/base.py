import logging
import time
from typing import Optional

from physjudge.judge import judge
from physjudge.types import Trajectory

from physics_gym.env.action.action_set import BASIC_ACTION_SET
from physics_gym.env.task.base import AbstractPhysicsTask


logger = logging.getLogger(__name__)


class PhysicsEnv:
    """Single-task agentic env. Mirrors MedAgentGym's reset/step contract.

    The reward returned at each step is the running best mean physjudge score
    seen so far for this episode -- we keep the last successful trajectory as
    the agent's standing answer until they submit() or hit the step limit.
    """

    def __init__(self, task_entrypoint: type[AbstractPhysicsTask],
                 task_kwargs: dict, action_set: Optional[dict] = None,
                 max_steps: int = 8, compiler=None):
        self.task_entrypoint = task_entrypoint
        self.task_kwargs = dict(task_kwargs)
        self.action_set = action_set or BASIC_ACTION_SET
        self.max_steps = max_steps
        self.compiler = compiler
        self.task: Optional[AbstractPhysicsTask] = None
        self.history: list[dict] = []
        self.best_traj: Optional[Trajectory] = None
        self.best_score: float = 0.0
        self.best_judge_result = None
        self.step_count = 0

    def reset(self, task_id: int):
        if self.task is not None:
            self.task.teardown()
        self.task = self.task_entrypoint(task_id=task_id, **self.task_kwargs)
        goal, info = self.task.setup()
        self.history = []
        self.best_traj = None
        self.best_score = 0.0
        self.best_judge_result = None
        self.step_count = 0
        self.start_time = time.time()
        obs = self.task._get_obs()
        self.history.append(obs)
        return obs, {"task_goal": goal, **info}

    def step(self, action_name: str, **kwargs):
        self.step_count += 1
        info = {"action_exec_start": time.time()}

        if action_name not in self.action_set:
            obs = {"type": "error", "env_message": f"unknown action: {action_name}"}
            return obs, 0.0, False, False, info

        action = self.action_set[action_name]
        try:
            result = action.fn(**kwargs)
        except Exception as e:
            result = {"type": "error", "env_message": f"{type(e).__name__}: {e}"}

        if action_name == "validate_code" and result.get("trajectory") is not None:
            mean_score, jr = self._score(result["trajectory"])
            result["mean_score"] = mean_score
            result["pass_rate"] = jr.pass_rate
            result["verdicts"] = self._summarize_verdicts(jr)
            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_traj = result["trajectory"]
                self.best_judge_result = jr
            result.pop("trajectory", None)

        terminated = action_name == "submit" or self.step_count >= self.max_steps
        info["action_exec_stop"] = time.time()
        info["best_score"] = self.best_score
        self.history.append(result)
        reward = self.best_score if terminated else 0.0
        return result, reward, terminated, False, info

    def _score(self, trajectory: Trajectory):
        jr = judge(self.task.scenario_obj, trajectory, compiler=self.compiler)
        return jr.mean_score, jr

    @staticmethod
    def _summarize_verdicts(jr) -> list[dict]:
        import numpy as np

        def _py(x):
            if isinstance(x, (np.bool_,)):
                return bool(x)
            if isinstance(x, (np.integer,)):
                return int(x)
            if isinstance(x, (np.floating,)):
                return float(x)
            if isinstance(x, dict):
                return {k: _py(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [_py(v) for v in x]
            return x

        return [
            {
                "standard": a.standard_text,
                "passed": bool(v.passed),
                "score": round(float(v.score), 4),
                "diagnostic": _py(v.diagnostic),
            }
            for a, v in zip(jr.assertions, jr.verdicts)
        ]

    def close(self):
        if self.task is not None:
            self.task.teardown()
            self.task = None

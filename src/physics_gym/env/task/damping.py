import json
import os

from .base import AbstractPhysicsTask
from physics_gym.env.action.function import TRAJECTORY_INSTRUCTIONS
from physics_gym.utils.compute_score import scenario_from_spec


INSTRUCTION = """You are a physics-coding agent. Simulate the scenario below and emit a trajectory the verifier can score.

Scenario:
{scenario_text}

Initial conditions:
{initial_conditions}

Available actions:
- validate_code(code): run your script in a subprocess. The script must finish by printing one JSON line with the trajectory.
- submit(): end the episode. The last successful trajectory is taken as your final answer.

{trajectory_instructions}
You may import numpy and scipy. Avoid network and filesystem access.
"""


class DampingTask(AbstractPhysicsTask):
    """Energy Conservation -> Energy Loss (Damping). Amplitude-decaying motion."""

    permitted_actions = ["validate_code", "submit"]

    def __init__(self, task_id: int, data_path: str, mode: str = "test") -> None:
        super().__init__(task_id=task_id)
        self.data_path = data_path
        self.mode = mode
        self.task_list = None

    def setup(self) -> tuple[str, dict]:
        if self.task_list is None:
            file_name = "test_tasks.jsonl" if self.mode == "test" else "train_tasks.jsonl"
            with open(os.path.join(self.data_path, file_name)) as f:
                self.task_list = [json.loads(line) for line in f]
        td = self.task_list[self.task_id]
        self.scenario_text = td["scenario"]
        self.initial_conditions = td["initial_conditions"]
        self.physjudge_spec = td["physjudge_spec"]
        self.scenario_obj = scenario_from_spec(
            self.physjudge_spec, category=td["category"], subcategory=td["subcategory"],
        )
        self.goal = self.scenario_text
        return self.goal, {"task_id": self.task_id}

    def _get_obs(self) -> dict:
        return {
            "type": "initial_observation",
            "info": {
                "task_goal": self.goal,
                "scenario": self.scenario_text,
                "initial_conditions": self.initial_conditions,
                "instruction": INSTRUCTION.format(
                    scenario_text=self.scenario_text,
                    initial_conditions=json.dumps(self.initial_conditions, indent=2),
                    trajectory_instructions=TRAJECTORY_INSTRUCTIONS,
                ),
            },
        }

    def validate(self, chat_messages, obs):
        return 0.0, False, "", {}

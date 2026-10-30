from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class TaskInstance:
    task_id: int
    category: str
    subcategory: str
    scenario: str
    initial_conditions: dict
    physjudge_spec: dict
    golden: dict


class AbstractPhysicsTask(ABC):
    permitted_actions: list[str] = ["validate_code", "submit"]

    def __init__(self, task_id: int) -> None:
        self.task_id = task_id

    @classmethod
    def get_task_id(cls) -> str:
        return f"PhysicsGym.{cls.__name__.replace('Task','').lower()}"

    @abstractmethod
    def setup(self) -> tuple[str, dict]:
        ...

    def teardown(self) -> None:
        pass

    @abstractmethod
    def _get_obs(self) -> dict:
        ...

    @abstractmethod
    def validate(self, chat_messages: Any, obs: dict) -> tuple[float, bool, str, dict]:
        ...

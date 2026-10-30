import json
import time
from typing import Any

from .parser import parse_action
from .prompt import SYSTEM_PROMPT, TASK_USER_PROMPT
from physics_gym.llm.chat_api import (
    make_assistant_message, make_system_message, make_user_message,
)


class PhysicsAgent:
    def __init__(self, llm: callable, n_retry: int = 2, retry_delay: float = 1.0):
        self.llm = llm
        self.n_retry = n_retry
        self.retry_delay = retry_delay
        self.conversation: list[dict] = []
        self.usage: list[dict] = []

    def reset(self):
        self.conversation = []
        self.usage = []

    def act(self, obs: dict) -> tuple[str, dict, str]:
        if not self.conversation:
            self.conversation.append(make_system_message(SYSTEM_PROMPT))
            user = TASK_USER_PROMPT.format(
                scenario=obs["info"]["scenario"],
                ic_json=json.dumps(obs["info"]["initial_conditions"], indent=2),
            )
            self.conversation.append(make_user_message(user))
        else:
            self.conversation.append(make_user_message(self._format_env_msg(obs)))

        last_err = None
        for attempt in range(self.n_retry + 1):
            try:
                response, u = self.llm(self.conversation)
                self.usage.append(u)
                action, params = parse_action(response)
                self.conversation.append(make_assistant_message(response))
                return action, params, response
            except Exception as e:
                last_err = e
                self.conversation.append(make_user_message(
                    f"Your previous response could not be parsed: {e}\n"
                    "Reply again with a single JSON object {\"action\": ..., \"params\": ...}."
                ))
                time.sleep(self.retry_delay)
        raise RuntimeError(f"agent failed after {self.n_retry + 1} attempts: {last_err}")

    @staticmethod
    def _format_env_msg(obs: Any) -> str:
        if not isinstance(obs, dict):
            return str(obs)
        kind = obs.get("type", "feedback")
        if kind == "code_execution" and obs.get("status") == "SUCCESS":
            return (
                f"Code ran. mean_score={obs.get('mean_score'):.3f} "
                f"pass_rate={obs.get('pass_rate'):.3f}\nVerdicts:\n"
                + json.dumps(obs.get("verdicts", []), indent=2)
                + "\nIf scores are high, call submit. Otherwise revise and call validate_code again."
            )
        if kind == "code_execution":
            return f"Code FAILED: {obs.get('env_message','')[:1500]}\nFix and retry."
        if kind == "error":
            return f"Env error: {obs.get('env_message','')[:1500]}"
        return json.dumps(obs)[:1500]

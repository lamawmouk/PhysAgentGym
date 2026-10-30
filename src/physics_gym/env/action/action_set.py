from dataclasses import dataclass

from .function import validate_code, submit, TRAJECTORY_INSTRUCTIONS


@dataclass
class Action:
    name: str
    fn: callable
    signature: str
    description: str


VALIDATE_CODE = Action(
    name="validate_code",
    fn=validate_code,
    signature='validate_code(code: str)',
    description=(
        "Run a self-contained Python script in a subprocess. "
        + TRAJECTORY_INSTRUCTIONS +
        "On success, the env runs physjudge against the trajectory and returns "
        "per-assertion verdicts. Allowed imports: numpy, scipy."
    ),
)

SUBMIT = Action(
    name="submit",
    fn=submit,
    signature="submit()",
    description="End the episode. Last successful trajectory is taken as the final answer.",
)


BASIC_ACTION_SET: dict[str, Action] = {
    "validate_code": VALIDATE_CODE,
    "submit": SUBMIT,
}

from physjudge.judge import judge
from physjudge.types import Scenario, Trajectory, JudgeResult


def score_trajectory(scenario: Scenario, trajectory: Trajectory,
                     compiler=None) -> tuple[float, JudgeResult]:
    result = judge(scenario, trajectory, compiler=compiler)
    return result.mean_score, result


def scenario_from_spec(spec: dict, category: str, subcategory: str) -> Scenario:
    return Scenario(
        prompt_index=str(spec.get("prompt_index", "")),
        prompt=spec["prompt"],
        physics=spec.get("physics", spec["prompt"]),
        basic_standards=spec.get("basic_standards", {}),
        key_standards=spec["key_standards"],
        category=category,
        subcategory=subcategory,
        detailed=spec.get("detailed"),
        detailed_long=spec.get("detailed_long"),
    )

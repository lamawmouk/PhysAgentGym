from .types import Scenario, Trajectory, JudgeResult, Assertion, Verdict
from .compiler import Compiler, RuleBasedCompiler


def _failed_assertion(standard: str, reason: str) -> Assertion:
    return Assertion(
        standard_text=standard,
        check_fn=lambda _t, _r=reason: Verdict(passed=False, score=0.0,
                                                diagnostic={"compile_error": _r}),
        description="compile_error",
        source="error",
    )


def judge(scenario: Scenario,
          trajectory: Trajectory,
          compiler: Compiler | None = None,
          confidence_gate: float = 0.5) -> JudgeResult:
    compiler = compiler or RuleBasedCompiler()
    assertions: list[Assertion] = []
    for s in scenario.key_standards:
        try:
            assertions.append(compiler.compile(s, scenario))
        except Exception as e:
            assertions.append(_failed_assertion(s, f"compile {type(e).__name__}: {str(e)[:160]}"))
    verdicts: list[Verdict] = []
    for a in assertions:
        try:
            verdicts.append(a.check_fn(trajectory))
        except Exception as e:
            verdicts.append(Verdict(passed=False, score=0.0,
                                     diagnostic={"eval_error": f"{type(e).__name__}: {str(e)[:160]}"}))
    coverage = float((trajectory.confidence >= confidence_gate).mean())
    return JudgeResult(scenario=scenario, assertions=assertions,
                       verdicts=verdicts, coverage=coverage)

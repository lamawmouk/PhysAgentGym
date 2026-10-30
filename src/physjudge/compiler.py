"""Compile English Key_Standards into executable Assertions on a Trajectory."""
import json
import os
import re
from typing import Protocol
from . import checks
from .dsl import DSLError, evaluate_dsl, schema_for_prompt
from .types import Assertion, Scenario, Verdict, Trajectory


class Compiler(Protocol):
    def compile(self, standard: str, scenario: Scenario) -> Assertion: ...


_RULES: list[tuple[str, str, callable]] = []


def _rule(pattern: str, description: str):
    def deco(fn):
        _RULES.append((pattern, description, fn))
        return fn
    return deco


@_rule(r"\b(straight\s+vertical\s+line|vertical(?:ly)?|straight\s+up)\b", "vertical straight motion")
def _vertical_line(s, sc):
    return lambda t: checks.straight_line(t, axis="y", tol=0.08)


@_rule(r"\b(straight\s+horizontal\s+line|horizontal(?:ly)?)\b", "horizontal straight motion")
def _horizontal_line(s, sc):
    return lambda t: checks.straight_line(t, axis="x", tol=0.08)


@_rule(r"\b(constant\s+(speed|velocity)|steady\s+(pace|speed)|uniform\s+motion|"
       r"consistent\s+(speed|velocity|pace)|"
       r"steady\s+speed\s+without\s+visible\s+acceleration|"
       r"without\s+visible\s+acceleration\s+or\s+deceleration|"
       r"no\s+noticeable\s+changes?\s+in\s+velocity|"
       r"no\s+visible\s+changes?\s+in\s+(speed|velocity))\b", "constant speed")
def _const_speed(s, sc):
    return lambda t: checks.constant_speed(t, cv_tol=0.18)


@_rule(r"\b(free[\s-]?fall|falls?\s+(due\s+to\s+gravity|under\s+gravity)|accelerates?\s+downward|"
       r"gravity\s+pulls?\s+(it|the\s+\w+)\s+down|due\s+to\s+gravity)\b",
       "constant downward acceleration (gravity-like)")
def _free_fall(s, sc):
    return lambda t: checks.gravity_like(t, axis=1, sign=-1, std_tol=0.5)


@_rule(r"\b(at\s+rest|does\s+not\s+move|stationary|remains?\s+still)\b", "stationary")
def _rest(s, sc):
    return lambda t: checks.at_rest(t, eps_frac=0.02)


@_rule(r"\b(rises?|ascends?|moves?\s+up(?:ward)?)\b", "monotonic upward")
def _up(s, sc):
    return lambda t: checks.monotonic(t, axis=1, direction="up")


@_rule(r"\b(falls?|descends?|moves?\s+down(?:ward)?|drops?)\b", "monotonic downward")
def _down(s, sc):
    return lambda t: checks.monotonic(t, axis=1, direction="down")


@_rule(r"\b(bounces?|rebounds?)\b", "at least one bounce")
def _bounce(s, sc):
    return lambda t: checks.bounces(t, axis=1, min_count=1)


@_rule(r"\b(loses?\s+energy|damp(ing|ed)|slows?\s+down|comes?\s+to\s+(a\s+)?(rest|stop))\b",
       "energy non-increasing")
def _damp(s, sc):
    return lambda t: checks.energy_non_increasing(t, tol=0.05)


@_rule(r"\b(elevation\s+(remains|stays)\s+unchanged|neither\s+rising\s+nor\s+falling|"
       r"no\s+(visible\s+)?(rise|fall)\b|stays?\s+at\s+the\s+same\s+(height|elevation))",
       "elevation unchanged")
def _elev(s, sc):
    return lambda t: checks.elevation_unchanged(t, tol_frac=0.10)


@_rule(r"\b(no\s+(visible\s+)?side[\s-]?to[\s-]?side\s+deviation|"
       r"no\s+lateral\s+(deviation|movement)|"
       r"without\s+(visible\s+)?deviation\s+in\s+its\s+direction)",
       "no lateral deviation")
def _lateral(s, sc):
    return lambda t: checks.no_lateral_deviation(t, tol=0.06)


@_rule(r"\b(straight(,\s*uninterrupted)?\s+path|in\s+a\s+straight\s+line(?:\s+along)?|"
       r"no\s+deviation\s+or\s+curvature|follows?\s+a\s+straight)\b",
       "straight path (any axis)")
def _straight_any(s, sc):
    return lambda t: checks.straight_path_any_axis(t, tol=0.08)


@_rule(r"\b(parabol(?:a|ic)|curved\s+(arc|path|trajectory)|smooth\s+arc|"
       r"clear\s+(arc|peak)|projectile\s+motion|arc(?:s|ed)?\s+through\s+the\s+air)\b",
       "parabolic trajectory")
def _parabolic(s, sc):
    return lambda t: checks.parabolic_fit(t, residual_tol=0.07)


@_rule(r"\b(smooth(ly)?(\s+,?\s+without\s+abrupt)?|"
       r"without\s+abrupt\s+(changes?|jumps?)|"
       r"no\s+sudden\s+(change|jump)|"
       r"changes?\s+direction\s+smoothly|"
       r"smooth(\s+upward\s+and\s+downward)?\s+(motion|trajectory|transition))\b",
       "smooth (low-jerk) motion")
def _smooth(s, sc):
    return lambda t: checks.smooth_velocity(t, jerk_tol=0.6)


@_rule(r"\b(reaches?\s+(a\s+)?(clear\s+)?(apex|peak|maximum\s+height|highest\s+point)|"
       r"visible\s+(apex|peak)|"
       r"ascent\s+to\s+the\s+apex(?:\s+and\s+a\s+descent)?|"
       r"rises?\s+(and|then)\s+(falls?|descends?)|"
       r"rising\s+and\s+then\s+falling|"
       r"upward\s+and\s+downward\s+trajectory|"
       r"throw\s+and\s+catch)\b",
       "apex (rise then fall)")
def _apex(s, sc):
    return lambda t: checks.apex_detection(t, axis=1)


@_rule(r"\b(no\s+(visible\s+)?changes?\s+in\s+direction|"
       r"no\s+sharp\s+turns?|"
       r"no\s+abrupt\s+(direction\s+)?changes?\s+in\s+(direction|trajectory))\b",
       "no direction changes")
def _no_dir(s, sc):
    return lambda t: checks.no_direction_changes(t, cos_threshold=0.92)


@_rule(r"\b(circular\s+motion|orbits?|moves?\s+in\s+a\s+circle|circular\s+(path|trajectory))\b",
       "circular motion (constant radius)")
def _circ(s, sc):
    return lambda t: checks.circular_motion(t, radius_cv_tol=0.12)


_ANTIPHYS_KEYWORDS = re.compile(
    r"\b(defy(ing)?|defies)\s+gravity|float(s|ing)?\s+upward\s+without|"
    r"perpetual\s+motion|gains?\s+energy\s+spontaneously|create[sd]?\s+energy",
    re.IGNORECASE,
)


def _no_op(t: Trajectory) -> Verdict:
    return Verdict(passed=True, score=0.5, diagnostic={"reason": "no rule matched"})


class RuleBasedCompiler:
    """Pattern-match Key_Standard text to a primitive trajectory check.

    Falls back to a no-op assertion if nothing matches; the LLM compiler
    is expected to handle the rest.
    """

    def compile(self, standard: str, scenario: Scenario) -> Assertion:
        text = standard.lower()
        for pattern, desc, fn in _RULES:
            if re.search(pattern, text, flags=re.IGNORECASE):
                check = fn(standard, scenario)
                return Assertion(standard_text=standard, check_fn=check,
                                 description=desc, source="rule")
        return Assertion(standard_text=standard, check_fn=_no_op,
                         description="unmatched", source="rule")


_LLM_SYSTEM = (
    "You compile a single English physical-realism assertion about a generated video "
    "into a constrained DSL evaluated against a tracked 2D trajectory of the primary "
    "object. Convention: y is up (physical, not pixel). Time is uniform in seconds.\n\n"
    "Reply with JSON only — no prose, no code fence — of the form:\n"
    '  {"reasoning": "<one sentence>", "tree": <DSL node>}\n\n'
)


class LLMCompiler:
    """Compile via Claude → constrained DSL tree → executable Assertion.

    Retries OverloadedError / RateLimitError with exponential backoff; caches
    compiled trees in-process keyed by (standard, physics_summary) so we never
    pay twice for the same standard within a run."""

    def __init__(self, model: str = "claude-opus-4-7", api_key: str | None = None,
                 max_tokens: int = 512, max_retries: int = 6):
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise RuntimeError("anthropic SDK not installed; pip install anthropic") from e
        self._Anthropic = Anthropic
        self.model = model
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._tree_cache: dict[tuple[str, str], dict] = {}

    def _system_text(self) -> str:
        return _LLM_SYSTEM + schema_for_prompt()

    def _user_text(self, standard: str, scenario: Scenario) -> str:
        return (f"Scenario prompt: {scenario.prompt}\n"
                f"Physics summary: {scenario.physics}\n"
                f"Category: {scenario.category} / {scenario.subcategory}\n\n"
                f"Compile this standard:\n  {standard}")

    def _request_with_retry(self, system: str, user: str) -> str:
        import time
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                msg = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=[{"type": "text", "text": system,
                              "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": user}],
                )
                return msg.content[0].text
            except Exception as e:                   # noqa: BLE001
                cls = type(e).__name__
                transient = cls in {"OverloadedError", "RateLimitError",
                                     "APIConnectionError", "APITimeoutError",
                                     "InternalServerError", "APIStatusError"}
                if not transient or attempt == self.max_retries - 1:
                    raise
                last_err = e
                time.sleep(min(60.0, 2.0 ** attempt + 0.25 * attempt))
        if last_err:
            raise last_err
        raise RuntimeError("unreachable")

    def compile(self, standard: str, scenario: Scenario) -> Assertion:
        cache_key = (standard, scenario.physics or "")
        tree = self._tree_cache.get(cache_key)
        if tree is None:
            text = self._request_with_retry(self._system_text(),
                                              self._user_text(standard, scenario))
            tree = _extract_tree(text)
            self._tree_cache[cache_key] = tree
        return Assertion(
            standard_text=standard,
            check_fn=lambda t, _tree=tree: evaluate_dsl(_tree, t),
            description=f"llm:{json.dumps(tree)[:80]}",
            source="llm",
        )


class CLILLMCompiler(LLMCompiler):
    """LLMCompiler that calls the headless `claude -p` CLI instead of the SDK.

    Same prompt, DSL, and in-process cache; useful where the CLI is authenticated
    but no ANTHROPIC_API_KEY / anthropic SDK is available."""

    def __init__(self, model: str = "sonnet", timeout: int = 300, max_retries: int = 3):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._tree_cache: dict[tuple[str, str], dict] = {}

    def _request_with_retry(self, system: str, user: str) -> str:
        import subprocess
        last_err: Exception | None = None
        for _ in range(self.max_retries):
            try:
                proc = subprocess.run(
                    ["claude", "-p", user, "--model", self.model,
                     "--system-prompt", system, "--output-format", "json",
                     "--disallowedTools",
                     "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,Task"],
                    capture_output=True, text=True, timeout=self.timeout)
                return json.loads(proc.stdout)["result"]
            except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError) as e:
                last_err = e
        raise RuntimeError(f"claude CLI compilation failed: {last_err}")


class MockLLMCompiler:
    """Returns canned DSL trees keyed by exact-match standard text. For tests."""

    def __init__(self, canned: dict[str, dict]):
        self.canned = canned

    def compile(self, standard: str, scenario: Scenario) -> Assertion:
        if standard not in self.canned:
            raise KeyError(f"no canned tree for standard: {standard!r}")
        tree = self.canned[standard]
        return Assertion(
            standard_text=standard,
            check_fn=lambda t: evaluate_dsl(tree, t),
            description=f"mock:{json.dumps(tree)[:80]}",
            source="llm",
        )


def _extract_tree(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    payload = _first_valid_json_object(text)
    if payload is None:
        raise DSLError(f"LLM response not JSON: {text[:200]!r}")
    if "tree" in payload:
        return payload["tree"]
    if "primitive" in payload or "op" in payload:
        return payload
    raise DSLError(f"LLM payload missing 'tree' / 'primitive' / 'op': {payload!r}")


def _first_valid_json_object(text: str):
    """Pull the first parseable {...} substring, robust to extra content after."""
    decoder = json.JSONDecoder()
    for i, c in enumerate(text):
        if c != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[i:])
            return obj
        except json.JSONDecodeError:
            continue
    return None


class HybridCompiler:
    """Try rules first; fall through to LLM only when no rule matches."""

    def __init__(self, llm: LLMCompiler | MockLLMCompiler):
        self.rules = RuleBasedCompiler()
        self.llm = llm

    def compile(self, standard: str, scenario: Scenario) -> Assertion:
        a = self.rules.compile(standard, scenario)
        if a.description != "unmatched":
            return a
        return self.llm.compile(standard, scenario)

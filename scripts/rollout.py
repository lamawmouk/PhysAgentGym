"""Run an LLM agent across a subcategory's test tasks and log per-task scores.

Usage:
  python scripts/rollout.py --task pe_ke --provider anthropic --model claude-sonnet-4-6 --n_tasks 10
  python scripts/rollout.py --task linear_motion --provider anthropic --n_tasks 10
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from physics_gym.agent.base import PhysicsAgent
from physics_gym.env.base import PhysicsEnv
from physics_gym.env.task.circular_motion import CircularMotionTask
from physics_gym.env.task.composite import CoupledTask, DragTask, DrivenTask, KeplerTask
from physics_gym.env.task.damping import DampingTask
from physics_gym.env.task.linear_motion import LinearMotionTask
from physics_gym.env.task.pe_ke import PEKETask
from physics_gym.llm.chat_api import AnthropicClient, HFTransformersClient, OpenAIClient
from physics_gym.utils.compiler import PhysicsGymCompiler


TASK_REGISTRY = {
    "pe_ke": PEKETask,
    "linear_motion": LinearMotionTask,
    "circular_motion": CircularMotionTask,
    "damping": DampingTask,
    "drag": DragTask,
    "kepler": KeplerTask,
    "driven": DrivenTask,
    "coupled": CoupledTask,
}


def make_llm(provider: str, model: str | None, adapter_path: str | None = None,
             temperature: float = 0.2, reasoning_effort: str | None = None):
    if provider == "anthropic":
        return AnthropicClient(model=model or "claude-sonnet-4-6", temperature=temperature)
    if provider == "openai":
        return OpenAIClient(model=model or "gpt-5", temperature=temperature,
                            reasoning_effort=reasoning_effort)
    if provider == "hf":
        if not model:
            raise ValueError("--model is required for provider=hf (path or HF repo id)")
        return HFTransformersClient(base_model=model, adapter_path=adapter_path,
                                    temperature=temperature)
    raise ValueError(f"unknown provider {provider!r}")


def run_task(env: PhysicsEnv, agent: PhysicsAgent, task_id: int) -> dict:
    obs, info = env.reset(task_id=task_id)
    agent.reset()
    last_obs = obs
    n_steps = 0
    transcript = []
    while True:
        try:
            action, params, raw = agent.act(last_obs)
        except RuntimeError as e:
            return {"task_id": task_id, "error": str(e), "best_score": env.best_score,
                    "n_steps": n_steps, "transcript": transcript}
        transcript.append({"step": n_steps, "action": action,
                           "raw_response": raw, "params": params})
        result, _, terminated, _, _ = env.step(action, **params)
        # Capture verdict snapshot when validate_code returned a trajectory.
        if action == "validate_code" and "verdicts" in result:
            transcript[-1]["verdicts"] = result["verdicts"]
            transcript[-1]["mean_score"] = result.get("mean_score")
        last_obs = result
        n_steps += 1
        if terminated:
            break
    return {
        "task_id": task_id,
        "best_score": env.best_score,
        "pass_rate": env.best_judge_result.pass_rate if env.best_judge_result else 0.0,
        "n_steps": n_steps,
        "regime": env.task.task_list[task_id].get("regime", ""),
        "category": env.task.task_list[task_id].get("category", ""),
        "subcategory": env.task.task_list[task_id].get("subcategory", ""),
        "scenario": env.task.task_list[task_id]["scenario"],
        "initial_conditions": env.task.task_list[task_id]["initial_conditions"],
        "usage": agent.usage,
        "transcript": transcript,
    }


def make_compiler_llm(kind: str):
    if kind == "api":
        from physjudge import LLMCompiler
        return LLMCompiler()
    if kind == "cli":
        from physjudge import CLILLMCompiler
        return CLILLMCompiler()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_REGISTRY))
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "hf"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter_path", default=None,
                    help="HF only: path to a trained PEFT/LoRA adapter")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--data_path", default=None,
                    help="defaults to data/{task}")
    ap.add_argument("--mode", default="test", choices=["train", "test"])
    ap.add_argument("--n_tasks", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=6)
    ap.add_argument("--out", default=None,
                    help="defaults to results/{task}_{mode}_rollout.jsonl")
    ap.add_argument("--workers", type=int, default=1,
                    help="Concurrent task workers (each gets its own env+agent).")
    ap.add_argument("--seed", type=int, default=None,
                    help="Inference sampling seed (hf provider: seeds torch/transformers RNG).")
    ap.add_argument("--reasoning_effort", default=None, choices=["low", "medium", "high"],
                    help="openai provider, reasoning models only (e.g. gpt-5).")
    ap.add_argument("--llm_compiler", default="none", choices=["none", "api", "cli"],
                    help="LLM fallback for standards no rule matches: physjudge LLMCompiler "
                         "via the Anthropic API, or via the headless claude CLI. With 'none', "
                         "an unmatched standard raises. All shipped standards match rules.")
    args = ap.parse_args()

    if args.seed is not None and args.provider == "hf":
        from transformers import set_seed
        set_seed(args.seed)

    data_path = args.data_path or f"data/{args.task}"
    out_path = Path(args.out or f"results/{args.task}_{args.mode}_rollout.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def make_env_agent():
        llm = make_llm(args.provider, args.model, args.adapter_path, args.temperature,
                       reasoning_effort=args.reasoning_effort)
        env = PhysicsEnv(
            task_entrypoint=TASK_REGISTRY[args.task],
            task_kwargs={"data_path": data_path, "mode": args.mode},
            max_steps=args.max_steps,
            compiler=PhysicsGymCompiler(llm=make_compiler_llm(args.llm_compiler)),
        )
        return env, PhysicsAgent(llm=llm)

    write_lock = threading.Lock()
    counter = {"done": 0}

    def _one(tid: int) -> dict:
        env, agent = make_env_agent()
        try:
            return run_task(env, agent, tid)
        finally:
            env.close()

    t0 = time.time()
    results = []
    with out_path.open("w") as f:
        if args.workers <= 1:
            for tid in range(args.n_tasks):
                r = _one(tid)
                with write_lock:
                    f.write(json.dumps(r) + "\n"); f.flush()
                results.append(r)
                print(f"[{tid+1}/{args.n_tasks}] regime={r.get('regime','?'):>20} "
                      f"score={r.get('best_score',0):.3f} pass={r.get('pass_rate',0):.3f} "
                      f"steps={r['n_steps']}")
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(_one, tid): tid for tid in range(args.n_tasks)}
                for fut in as_completed(futures):
                    r = fut.result()
                    with write_lock:
                        f.write(json.dumps(r) + "\n"); f.flush()
                    results.append(r)
                    counter["done"] += 1
                    print(f"[{counter['done']}/{args.n_tasks}] tid={r['task_id']:>3} "
                          f"regime={r.get('regime','?'):>20} "
                          f"score={r.get('best_score',0):.3f} "
                          f"pass={r.get('pass_rate',0):.3f} steps={r['n_steps']}")

    dt = time.time() - t0
    scores = [r["best_score"] for r in results if "best_score" in r]
    print(f"\n{args.n_tasks} tasks in {dt:.1f}s ({args.workers}w); mean best_score="
          f"{sum(scores)/max(len(scores),1):.3f}; -> {out_path}")


if __name__ == "__main__":
    main()

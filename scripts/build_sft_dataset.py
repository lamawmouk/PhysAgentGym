"""Filter rollouts by mean_score and format the survivors as SFT records.

Pipeline:
  rollouts JSONL  --filter-->  passing rollouts  --format-->  SFT JSONL
                  (>= threshold)                  (messages)

Mirrors MedAgentGym's "successful trajectories -> SFT" recipe. Each surviving
rollout becomes one training example: the conversation that led to a high-
scoring trajectory.

The output schema matches HF SFTTrainer's expectations:
  {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}, ...]}
"""
import argparse
import json
from pathlib import Path

from physics_gym.agent.prompt import SYSTEM_PROMPT, TASK_USER_PROMPT


def rollout_to_sft_record(rollout: dict) -> dict | None:
    """Reconstruct the conversation from a rollout transcript.

    Returns None if the transcript is missing pieces or the rollout errored out.
    """
    if "transcript" not in rollout or rollout.get("error"):
        return None

    user_msg = TASK_USER_PROMPT.format(
        scenario=rollout["scenario"],
        ic_json=json.dumps(rollout["initial_conditions"], indent=2),
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    last_step = None
    for i, step in enumerate(rollout["transcript"]):
        raw = step.get("raw_response")
        if raw is None:
            return None
        messages.append({"role": "assistant", "content": raw})
        if i < len(rollout["transcript"]) - 1:
            verdicts = step.get("verdicts")
            mean = step.get("mean_score")
            if verdicts is not None:
                fb = (f"Code ran. mean_score={mean:.3f}\nVerdicts:\n"
                      + json.dumps(verdicts, indent=2))
            else:
                fb = "Step feedback unavailable."
            messages.append({"role": "user", "content": fb})
        last_step = step

    return {
        "messages": messages,
        "meta": {
            "task_id": rollout["task_id"],
            "category": rollout.get("category", ""),
            "subcategory": rollout.get("subcategory", ""),
            "regime": rollout.get("regime", ""),
            "best_score": rollout["best_score"],
            "pass_rate": rollout["pass_rate"],
            "n_steps": rollout["n_steps"],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", nargs="+", required=True,
                    help="One or more rollout JSONL files")
    ap.add_argument("--out", required=True, help="Output SFT JSONL path")
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="Keep rollouts with mean best_score >= threshold")
    ap.add_argument("--require_pass_rate", type=float, default=1.0,
                    help="Also require pass_rate >= this value")
    ap.add_argument("--reward_mode", choices=["sparse", "dense", "combined"], default="combined",
                    help="sparse: filter only on pass_rate. dense: filter only on mean_score. "
                         "combined: filter on both (V0 default).")
    args = ap.parse_args()

    # Reward-mode interpretation:
    #   sparse: pass_rate >= require_pass_rate ONLY (mean_score ignored)
    #   dense:  best_score >= threshold ONLY (pass_rate ignored)
    #   combined: both conditions (most stringent)
    if args.reward_mode == "sparse":
        args.threshold = 0.0  # disable dense filter
    elif args.reward_mode == "dense":
        args.require_pass_rate = 0.0  # disable sparse filter

    kept, dropped, errored = 0, 0, 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as out_f:
        for rollout_path in args.rollouts:
            with open(rollout_path) as f:
                for line in f:
                    rollout = json.loads(line)
                    if rollout.get("error"):
                        errored += 1
                        continue
                    if (rollout["best_score"] < args.threshold or
                            rollout["pass_rate"] < args.require_pass_rate):
                        dropped += 1
                        continue
                    rec = rollout_to_sft_record(rollout)
                    if rec is None:
                        errored += 1
                        continue
                    out_f.write(json.dumps(rec) + "\n")
                    kept += 1

    print(f"kept={kept} dropped={dropped} errored={errored} -> {out_path}")
    if kept == 0:
        print("WARNING: zero records kept; check thresholds or rollout quality.")


if __name__ == "__main__":
    main()

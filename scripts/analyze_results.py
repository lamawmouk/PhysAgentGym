"""Aggregate rollout JSONL files into a leaderboard-style report.

  python scripts/analyze_results.py results/*.jsonl
  python scripts/analyze_results.py results/pe_ke_*.jsonl --pass_threshold 0.9
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_rollouts(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["__source"] = p
            rows.append(r)
    return rows


def _stats(scores: list[float]) -> dict:
    if not scores:
        return {"n": 0}
    return {
        "n": len(scores),
        "mean": round(statistics.mean(scores), 4),
        "median": round(statistics.median(scores), 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
    }


def summarize(rollouts: list[dict], pass_threshold: float) -> dict:
    n = len(rollouts)
    errored = sum(1 for r in rollouts if r.get("error"))
    valid = [r for r in rollouts if not r.get("error")]
    scores = [r["best_score"] for r in valid]
    pass_rates = [r["pass_rate"] for r in valid]
    n_steps = [r["n_steps"] for r in valid]
    pass_at_1 = sum(1 for s in scores if s >= pass_threshold) / max(len(scores), 1)

    return {
        "n_total": n,
        "n_errored": errored,
        "n_valid": len(valid),
        f"pass@1 (score>={pass_threshold})": round(pass_at_1, 4),
        "best_score": _stats(scores),
        "pass_rate": _stats(pass_rates),
        "n_steps": _stats(n_steps),
    }


def breakdown(rollouts: list[dict], key: str, pass_threshold: float) -> dict:
    groups = defaultdict(list)
    for r in rollouts:
        groups[r.get(key, "<missing>")].append(r)
    return {g: summarize(rs, pass_threshold) for g, rs in sorted(groups.items())}


def _print_summary(label: str, s: dict, indent: int = 0):
    pad = " " * indent
    print(f"{pad}{label}")
    for k, v in s.items():
        if isinstance(v, dict):
            print(f"{pad}  {k}: " + ", ".join(f"{kk}={vv}" for kk, vv in v.items()))
        else:
            print(f"{pad}  {k}: {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rollouts", nargs="+", help="One or more rollout JSONL files")
    ap.add_argument("--pass_threshold", type=float, default=0.85)
    ap.add_argument("--by", choices=["subcategory", "regime", "category", "__source"],
                    default="subcategory",
                    help="Group breakdown by this field (default: subcategory)")
    ap.add_argument("--json", action="store_true", help="Emit JSON to stdout instead of text")
    args = ap.parse_args()

    rollouts = load_rollouts(args.rollouts)
    overall = summarize(rollouts, args.pass_threshold)
    by_group = breakdown(rollouts, args.by, args.pass_threshold)

    if args.json:
        print(json.dumps({"overall": overall, f"by_{args.by}": by_group}, indent=2))
        return

    print(f"=== {len(rollouts)} rollouts from {len(args.rollouts)} file(s) ===\n")
    _print_summary("OVERALL", overall)
    print(f"\n--- by {args.by} ---")
    for group, s in by_group.items():
        _print_summary(group, s, indent=2)


if __name__ == "__main__":
    main()

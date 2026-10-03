"""Technique ablation study — Reflexion + CodeRL graded reward.

Runs 4 conditions on the frozen eval set and compares pass rates.
Extends the architecture ablation (ablation.py) with inference-time
techniques from Reflexion (Shinn et al.) and CodeRL (Le et al.).

Conditions:
  baseline        — current pipeline (use_moa=True, no new techniques)
  reflexion       — +post-mortem reflection buffer after compile failure
  graded_reward   — +CodeRL error tier classification fed to coder
  both            — reflexion + graded_reward combined

Usage:
    python experiments/ablation_techniques.py
    python experiments/ablation_techniques.py --limit 10  # smoke test
    python experiments/ablation_techniques.py --condition reflexion  # single run
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_R = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(_R)) if str(_R) not in sys.path else None

from experiments.run_baseline_eval import run_eval

EVAL_PATH   = _R / "data" / "eval_set.jsonl"
RESULTS_DIR = _R / "experiments" / "results"
CHECKPOINT  = RESULTS_DIR / "ablation_techniques_checkpoint.json"

CONDITIONS = [
    {
        "name": "baseline",
        "use_moa": True,
        "use_reflexion": False,
        "use_graded_reward": False,
        "description": "MoA pipeline, no new techniques",
    },
    {
        "name": "reflexion",
        "use_moa": True,
        "use_reflexion": True,
        "use_graded_reward": False,
        "description": "MoA + Reflexion post-mortem buffer",
    },
    {
        "name": "graded_reward",
        "use_moa": True,
        "use_reflexion": False,
        "use_graded_reward": True,
        "description": "MoA + CodeRL graded error tier",
    },
    {
        "name": "both",
        "use_moa": True,
        "use_reflexion": True,
        "use_graded_reward": True,
        "description": "MoA + Reflexion + CodeRL graded error tier",
    },
]


def run_ablation(condition_filter: str | None, limit: int | None, resume: bool = True,
                  parallel: int = 1) -> None:
    conditions = CONDITIONS
    if condition_filter:
        conditions = [c for c in CONDITIONS if c["name"] == condition_filter]
        if not conditions:
            valid = ", ".join(c["name"] for c in CONDITIONS)
            raise ValueError(f"Unknown condition '{condition_filter}'. Valid: {valid}")

    # ponytail: crash/walltime recovery — a full sweep is many hours with no
    # natural resume point otherwise. Load whatever conditions already
    # finished from the checkpoint and skip them; each condition still writes
    # the checkpoint the instant it finishes, not just at the very end.
    all_summaries: list[dict] = []
    done_names: set[str] = set()
    if resume and CHECKPOINT.exists():
        all_summaries = json.loads(CHECKPOINT.read_text())
        done_names = {s["ablation_condition"] for s in all_summaries}
        if done_names:
            print(f"[resume] found checkpoint with {len(done_names)} condition(s) done: "
                  f"{sorted(done_names)}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for cond in conditions:
        if cond["name"] in done_names:
            print(f"[resume] skipping already-completed condition: {cond['name']}", flush=True)
            continue

        print(f"\n{'#'*60}", flush=True)
        print(f"# Technique ablation: {cond['name']}", flush=True)
        print(f"# {cond['description']}", flush=True)
        print(f"{'#'*60}\n", flush=True)

        summary = run_eval(
            eval_path=str(EVAL_PATH),
            use_moa=cond["use_moa"],
            limit=limit,
            use_reflexion=cond["use_reflexion"],
            use_graded_reward=cond["use_graded_reward"],
            parallel=parallel,
            checkpoint_path=RESULTS_DIR / f"ablation_techniques_scenario_checkpoint_{cond['name']}.json",
        )
        summary["ablation_condition"] = cond["name"]
        all_summaries.append(summary)

        # Checkpoint immediately — this is the fallback for an 8h walltime
        # kill or crash mid-sweep; without it a kill loses every condition
        # run so far, not just the one in progress.
        CHECKPOINT.write_text(json.dumps(all_summaries, indent=2))
        print(f"[checkpoint] {cond['name']} done, saved -> {CHECKPOINT}", flush=True)

    # Save combined results under a timestamped final name too
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"ablation_techniques_{ts}.json"
    out.write_text(json.dumps(all_summaries, indent=2))
    print(f"\nCombined results saved → {out}")

    # Print comparison table
    print(f"\n{'='*65}")
    print("TECHNIQUE ABLATION — Pass Rate Comparison")
    print(f"{'='*65}")
    print(f"{'Condition':<20} {'Passed':>8} {'Total':>7} {'Pass%':>8}  Description")
    print("-" * 65)
    for s in all_summaries:
        pct = f"{s['pass_rate']*100:.1f}%"
        name = s.get("ablation_condition", s.get("config", "?"))
        desc = next((c["description"] for c in CONDITIONS if c["name"] == name), "")
        print(f"{name:<20} {s['passed']:>8} {s['total']:>7} {pct:>8}  {desc}")
    print(f"{'='*65}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default=None, help="Run a single condition by name")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N tasks (smoke test)")
    parser.add_argument("--no-resume", action="store_true", help="Ignore any existing checkpoint, start clean")
    parser.add_argument("--parallel", type=int, default=1,
                         help="Scenarios per condition run concurrently (default 1, unchanged "
                              "behavior). Requires OLLAMA_NUM_PARALLEL>=N on the server. Each "
                              "scenario is checkpointed as it completes (not just per-condition), "
                              "so a crash/kill mid-condition only loses in-flight scenarios.")
    args = parser.parse_args()
    run_ablation(args.condition, args.limit, resume=not args.no_resume, parallel=args.parallel)


if __name__ == "__main__":
    main()

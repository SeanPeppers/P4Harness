"""
zero_shot_model_sweep.py — single-pass, no-repair zero-shot base-model
comparison (paper Table: "Single-Pass Base Models + RAG, No Multi-Round
Repair"). Swaps the p4_coder role across model families and measures REAL
p4c-bm2-ss pass rate on the frozen eval set (data/eval_set.jsonl), one
planner->coder->compiler->gate pass per task, no retry.

Usage:
  python experiments/zero_shot_model_sweep.py
  python experiments/zero_shot_model_sweep.py --limit 5   # smoke test
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_ollama import ChatOllama
from agents.graph_baselines import build_zero_shot_real_graph
from agents.model_router import get_llm as _real_get_llm, _load_config
import agents.nodes as _nodes
from experiments.base import RESULTS_DIR

EVAL_PATH = _ROOT / "data" / "eval_set.jsonl"
BASELINE_P4 = (_ROOT / "p4" / "sfc_baseline.p4").read_text()
CHECKPOINT = RESULTS_DIR / "zero_shot_sweep_checkpoint.json"

MODELS = [
    {"name": "Qwen2.5-Coder-14B", "ollama_tag": "qwen2.5-coder:14b", "params": "14B"},
    {"name": "Qwen2.5-Coder-7B",  "ollama_tag": "qwen2.5-coder:7b",  "params": "7B"},
    {"name": "DeepSeek-Coder",    "ollama_tag": "deepseek-coder:6.7b", "params": "6.7B"},
    {"name": "CodeLlama",         "ollama_tag": "codellama:13b",     "params": "13B"},
]


def _swapped_get_llm(role: str, temperature: float | None = None, num_predict: int | None = None):
    """In-process-only override: p4_coder uses the swept model, every other
    role (planner) keeps the real configured model. Never touches
    configs/models.yaml on disk, so this is safe to run alongside other jobs
    (e.g. the architecture-ablation configs) that share that file."""
    if role != "p4_coder":
        return _real_get_llm(role, temperature=temperature, num_predict=num_predict)
    cfg = _load_config()
    base = cfg["models"]["p4_coder"]
    return ChatOllama(
        model=_CURRENT_MODEL_TAG[0],
        temperature=temperature if temperature is not None else base["temperature"],
        base_url=cfg["ollama"]["base_url"],
        num_predict=num_predict if num_predict is not None else base.get("num_predict", 512),
    )


_CURRENT_MODEL_TAG = [None]
_nodes.get_llm = _swapped_get_llm


def run_model(model_cfg: dict, tasks: list[dict], limit: int | None) -> dict:
    _CURRENT_MODEL_TAG[0] = model_cfg["ollama_tag"]
    pipeline = build_zero_shot_real_graph()

    if limit:
        tasks = tasks[:limit]

    passed = 0
    records = []
    for i, task in enumerate(tasks, 1):
        state = {
            "task": task["task"],
            "p4_code": BASELINE_P4,
            "plan": "", "review_notes": [],
            "test_output": "", "review_passed": False,
            "tests_passed": False, "approved_for_push": False,
            "round": 0,
            "store_in_rag": False,
        }
        t0 = time.time()
        final = {}
        try:
            for step in pipeline.stream(state):
                final.update(list(step.values())[0])
            approved = bool(final.get("approved_for_push", False))
        except Exception as exc:
            approved = False
            final = {"error": str(exc)}
        dur = round(time.time() - t0, 2)
        passed += approved
        records.append({"task": task["task"][:80], "approved": approved, "duration_s": dur})
        print(f"[{model_cfg['name']}] {i}/{len(tasks)}  approved={approved}  ({dur}s)", flush=True)

    total = len(tasks)
    return {
        "model": model_cfg["name"],
        "ollama_tag": model_cfg["ollama_tag"],
        "params": model_cfg["params"],
        "passed": passed,
        "total": total,
        "pass_rate": passed / total if total else 0.0,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run first N eval tasks (smoke test)")
    parser.add_argument("--model", default=None, help="Run a single model by name (e.g. CodeLlama)")
    parser.add_argument("--no-resume", action="store_true", help="Ignore any existing checkpoint, start clean")
    args = parser.parse_args()

    tasks = [json.loads(l) for l in EVAL_PATH.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(tasks)} frozen eval tasks from {EVAL_PATH}", flush=True)

    models = MODELS if not args.model else [m for m in MODELS if m["name"] == args.model]
    if not models:
        valid = ", ".join(m["name"] for m in MODELS)
        raise SystemExit(f"Unknown --model. Valid: {valid}")

    # ponytail: crash/walltime recovery — each model is its own long single-pass
    # sweep; without a checkpoint a kill mid-sweep loses every model run so far,
    # not just the one in progress (same failure mode ablation_techniques.py
    # had before it got the same fix).
    all_results: list[dict] = []
    done_names: set[str] = set()
    if not args.no_resume and CHECKPOINT.exists():
        all_results = json.loads(CHECKPOINT.read_text())
        done_names = {r["model"] for r in all_results}
        if done_names:
            print(f"[resume] found checkpoint with {len(done_names)} model(s) done: "
                  f"{sorted(done_names)}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for m in models:
        if m["name"] in done_names:
            print(f"[resume] skipping already-completed model: {m['name']}", flush=True)
            continue

        print(f"\n{'#'*60}\n# Zero-shot sweep: {m['name']} ({m['ollama_tag']})\n{'#'*60}", flush=True)
        result = run_model(m, tasks, args.limit)
        all_results.append(result)
        print(f"[{m['name']}] DONE: {result['passed']}/{result['total']} ({result['pass_rate']*100:.1f}%)", flush=True)

        # Checkpoint immediately after each model — this is the fallback for
        # a walltime kill or crash mid-sweep.
        CHECKPOINT.write_text(json.dumps(all_results, indent=2))
        print(f"[checkpoint] {m['name']} done, saved -> {CHECKPOINT}", flush=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"zero_shot_sweep_{ts}.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved -> {out}")

    print(f"\n{'='*50}\nZERO-SHOT BASE MODEL COMPARISON\n{'='*50}")
    for r in all_results:
        print(f"{r['model']:<20} {r['params']:>6}  {r['passed']}/{r['total']} ({r['pass_rate']*100:.1f}%)")


if __name__ == "__main__":
    main()

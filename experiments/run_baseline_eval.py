"""Run the current pipeline on the frozen eval set and report a baseline pass rate.

Each task in eval_set.jsonl is run through build_graph() — the same pipeline used
in production. Pass = generated P4 compiles successfully with p4c. Fail = compile
error after all repair rounds.

Results are saved to experiments/results/baseline_eval_<timestamp>.json so every
idea we test later can be compared against this number.

Usage:
    python experiments/run_baseline_eval.py
    python experiments/run_baseline_eval.py --eval data/eval_set.jsonl --use-moa
    python experiments/run_baseline_eval.py --limit 10   # smoke test first 10
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

_R = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(_R)) if str(_R) not in sys.path else None

_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH   = _ROOT / "data" / "eval_set.jsonl"
RESULTS_DIR = _ROOT / "experiments" / "results"
BASELINE_P4 = _ROOT / "p4" / "sfc_baseline.p4"


def _extract_ingress_body(p4_code: str) -> str:
    code = p4_code.strip()
    # Was a literal "control MyIngress" search -- any model naming its
    # control block differently (e.g. a fine-tune whose ingress signature
    # degraded to "control In(...)") missed entirely, fell through to
    # returning the WHOLE program, which _make_compilable then nested inside
    # its own "control MyIngress(...) {", producing a guaranteed
    # "syntax error, unexpected CONTROL" regardless of whether the rest of
    # the fragment was otherwise fine (2026-09-18 finding). Match any
    # control-block declaration instead -- preferring one with "Ingress" in
    # the name, since a full v1model program declares VerifyChecksum/
    # ComputeChecksum controls before Ingress and a bare first-match would
    # grab one of those (usually an empty body) instead (2026-09-18 pre-push
    # review finding).
    match = re.search(r"control\s+\w*[Ii]ngress\w*\s*\(", code) or re.search(r"control\s+\w+\s*\(", code)
    if match is None:
        return code
    idx = match.start()
    open_idx = code.find("{", idx)
    if open_idx == -1:
        return code  # malformed -- let p4c report the error
    depth, close_idx = 0, -1
    for i, ch in enumerate(code[open_idx:], open_idx):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                close_idx = i
                break
    if close_idx == -1:
        return code
    return code[open_idx + 1 : close_idx].strip()


def _make_compilable(fragment: str) -> str:
    baseline = BASELINE_P4.read_text()
    pre_ingress, rest = baseline.split("/* Ingress */", 1)
    _, post_egress = rest.split("/* Egress", 1)
    ingress_sig = (
        "control MyIngress(inout headers_t hdr, inout metadata_t meta,\n"
        "                  inout standard_metadata_t std_meta) {"
    )
    body = _extract_ingress_body(fragment)
    return (
        pre_ingress
        + "/* Ingress */\n"
        + ingress_sig
        + "\n\n"
        + body
        + "\n\n}\n\n/* Egress"
        + post_egress
    )


def _compile_p4(p4_code: str) -> tuple[bool, str]:
    """Wrap partial P4 in the SFC baseline template, then compile with p4c.

    ponytail: this used to shell out its own hardcoded `sudo docker run`
    with no fallback, which always fails non-interactively on HPC nodes
    (see agents/nodes.py's _run_p4c history: it silently produced 0% pass
    rates indistinguishable from bad code). Reuse the already-hardened
    p4c-bm2-ss / apptainer / singularity / sudo-checked-docker chain
    instead of duplicating a cruder, broken one.
    """
    from agents.nodes import _run_p4c

    try:
        full_code = _make_compilable(p4_code)
    except Exception as exc:
        return False, f"wrap error: {exc}"

    tmp_dir = Path(tempfile.mkdtemp(prefix="eval_p4c_"))
    tmp_file = tmp_dir / f"c_{uuid.uuid4().hex[:8]}.p4"
    tmp_file.write_text(full_code)
    try:
        return _run_p4c(tmp_file)
    finally:
        try: tmp_file.unlink()
        except Exception: pass
        try: tmp_dir.rmdir()
        except Exception: pass


def run_eval(eval_path: str, use_moa: bool, limit: int | None,
             use_reflexion: bool = False, use_graded_reward: bool = False,
             parallel: int = 1, checkpoint_path: str | Path | None = None) -> dict:
    from agents.graph import build_graph

    entries = [json.loads(l) for l in Path(eval_path).read_text().splitlines() if l.strip()]
    if limit:
        entries = entries[:limit]

    graph = build_graph(use_moa=use_moa, use_reflexion=use_reflexion, use_graded_reward=use_graded_reward)
    total_n = len(entries)

    # ponytail: scenario-level checkpoint — with --parallel N a single condition
    # can run for hours; without this, a crash or walltime kill loses every
    # scenario in the current condition, not just the one in flight (same
    # failure mode the per-condition checkpoint in ablation_techniques.py
    # already fixed one level up).
    ckpt_path = Path(checkpoint_path) if checkpoint_path else None
    done_by_task: dict[str, dict] = {}
    if ckpt_path and ckpt_path.exists():
        try:
            done_by_task = {r["task"]: r for r in json.loads(ckpt_path.read_text())}
            print(f"[resume] scenario checkpoint has {len(done_by_task)} done: {ckpt_path}", flush=True)
        except Exception:
            done_by_task = {}
    ckpt_lock = threading.Lock()

    def _checkpoint(records_so_far: list[dict]) -> None:
        if not ckpt_path:
            return
        with ckpt_lock:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt_path.write_text(json.dumps(records_so_far, indent=2))

    def _ts() -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _run_one(i: int, entry: dict) -> dict:
        worker = threading.current_thread().name
        cached = done_by_task.get(entry["task"])
        if cached is not None:
            print(f"[{_ts()}][{worker}][{i+1}/{total_n}] {entry.get('scenario', 'unknown')}: [resumed from checkpoint]", flush=True)
            return cached
        task = entry["task"]
        scenario = entry.get("scenario", "unknown")
        print(f"[{_ts()}][{worker}][{i+1}/{total_n}] {scenario}: {task[:80]}...", flush=True)

        t0 = time.time()
        result = {}
        exc_traceback = ""
        try:
            result = graph.invoke({"task": task})
            p4_code = result.get("p4_code") or result.get("current_p4", "")
            compiled, stderr = _compile_p4(p4_code)
        except Exception as exc:
            p4_code = ""
            compiled = False
            stderr = str(exc)
            # ponytail: str(exc) alone gave no line number / call stack for
            # crash diagnosis; a run that dies at scenario 140/200 needs the
            # full traceback to fix, not just "KeyError: 'p4_code'".
            exc_traceback = traceback.format_exc()
            print(f"[{_ts()}][{worker}][{i+1}/{total_n}] EXCEPTION in graph.invoke/_compile_p4:\n{exc_traceback}", flush=True)

        elapsed = round(time.time() - t0, 1)
        status = "PASS" if compiled else "FAIL"
        print(f"[{_ts()}][{worker}][{i+1}/{total_n}] → {status} ({elapsed}s)", flush=True)
        if not compiled:
            print(f"[{_ts()}][{worker}][{i+1}/{total_n}] error: {stderr[:120]}", flush=True)

        return {
            "task": task,
            "scenario": scenario,
            "passed": compiled,
            "elapsed_s": elapsed,
            "p4_code": p4_code,
            "p4_code_lines": len(p4_code.splitlines()) if p4_code else 0,
            "stderr": stderr if not compiled else "",
            "exception_traceback": exc_traceback,
            "worker": worker,
            "review_rounds": result.get("round", 0) if isinstance(result, dict) else 0,
            "error_tier": result.get("error_tier", "") if isinstance(result, dict) else "",
            "reflection_buffer": result.get("reflection_buffer", []) if isinstance(result, dict) else [],
            "review_notes": result.get("review_notes", []) if isinstance(result, dict) else [],
            "tests_passed": result.get("tests_passed", None) if isinstance(result, dict) else None,
        }

    # ponytail: default parallel=1 keeps the exact original sequential behavior.
    # Each thread writes to its own list index, so no lock is needed here.
    # agents/nodes.py's RAG memory singleton now locks its own critical
    # section (RAGMemory.add()), so concurrent scenarios are safe.
    if parallel and parallel > 1:
        records: list[dict | None] = [None] * total_n
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futures = {ex.submit(_run_one, i, e): i for i, e in enumerate(entries)}
            for fut in as_completed(futures):
                records[futures[fut]] = fut.result()
                _checkpoint([r for r in records if r is not None])
    else:
        records = []
        for i, e in enumerate(entries):
            records.append(_run_one(i, e))
            _checkpoint(records)

    passed = sum(1 for r in records if r["passed"])

    total = len(records)
    pass_rate = passed / total if total else 0.0

    by_scenario: dict[str, dict] = {}
    for r in records:
        s = r["scenario"]
        if s not in by_scenario:
            by_scenario[s] = {"passed": 0, "total": 0}
        by_scenario[s]["total"] += 1
        if r["passed"]:
            by_scenario[s]["passed"] += 1

    config_parts = []
    if use_moa: config_parts.append("moa")
    if use_reflexion: config_parts.append("reflexion")
    if use_graded_reward: config_parts.append("graded_reward")
    config_label = "+".join(config_parts) if config_parts else "single_shot"
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config_label,
        "eval_path": str(eval_path),
        "total": total,
        "passed": passed,
        "pass_rate": round(pass_rate, 4),
        "by_scenario": by_scenario,
        "records": records,
    }

    print(f"\n{'='*50}")
    print(f"BASELINE RESULT: {passed}/{total} ({pass_rate:.1%}) — config={summary['config']}")
    print(f"{'='*50}")
    for s, d in sorted(by_scenario.items()):
        print(f"  {s}: {d['passed']}/{d['total']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"baseline_eval_{ts}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved → {out}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default=str(EVAL_PATH))
    parser.add_argument("--use-moa", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N tasks (smoke test)")
    parser.add_argument("--use-reflexion", action="store_true")
    parser.add_argument("--use-graded-reward", action="store_true")
    parser.add_argument("--parallel", type=int, default=1,
                         help="Run N scenarios concurrently via threads (default 1 = sequential, "
                              "unchanged behavior). Requires OLLAMA_NUM_PARALLEL>=N on the server. "
                              "agents/nodes.py's RAG memory singleton (_rag) locks its critical "
                              "section in RAGMemory.add(), so concurrent scenario approvals are safe.")
    parser.add_argument("--checkpoint", default=None,
                         help="Path to a scenario-level checkpoint JSON. Written after every "
                              "scenario completes (not just at the end); rerunning with the same "
                              "path resumes and skips already-done tasks.")
    args = parser.parse_args()

    run_eval(args.eval, args.use_moa, args.limit,
             use_reflexion=args.use_reflexion,
             use_graded_reward=args.use_graded_reward,
             parallel=args.parallel,
             checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()

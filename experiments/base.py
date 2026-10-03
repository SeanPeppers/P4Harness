"""Base experiment class — shared scaffolding for all three experiments."""
from __future__ import annotations
import json, math, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict

import sys

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from agents import pipeline as _default_pipeline
from prometheus_client import Counter, Histogram, Gauge

RESULTS_DIR = _ROOT_DIR / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared Prometheus metrics ──────────────────────────────────────────────────
exp_runs          = Counter("p4harness_exp_runs_total", "Experiment runs", ["experiment", "outcome"])
exp_duration      = Histogram("p4harness_exp_duration_seconds", "Experiment wall time", ["experiment"],
                              buckets=[30, 60, 120, 300, 600, 1200])
exp_review_rounds = Histogram("p4harness_exp_review_rounds", "Revision rounds per run", ["experiment"],
                              buckets=[1, 2, 3, 4])
exp_compile_ok    = Counter("p4harness_exp_compile_success_total", "Successful P4 compiles", ["experiment"])
exp_compile_fail  = Counter("p4harness_exp_compile_fail_total",    "Failed P4 compiles",     ["experiment"])
exp_push_ok       = Counter("p4harness_exp_push_success_total", "Successful P4 pushes", ["experiment"])
exp_active        = Gauge("p4harness_exp_active", "Currently running experiment", ["experiment"])


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials.

    Wilson rather than the normal approximation: at the n and k this suite
    produces (n=3..5, k often 0 or n) the normal approximation collapses to a
    zero-width interval, which is exactly the claim a reviewer will reject.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class RunRecord:
    """Single pipeline run result — serialised to JSON."""
    experiment: str
    run_id: str          = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str       = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    task: str            = ""
    scenario: str        = ""
    repeat_idx: int      = 0   # 0-based trial number when run_all(repeat=N)
    # Outcomes
    review_passed: bool  = False
    tests_passed: bool   = False
    approved: bool       = False
    review_rounds: int   = 0
    # Timing (seconds)
    duration_s: float    = 0.0
    # Artefacts
    p4_code: str         = ""
    p4_code_lines: int   = 0
    review_notes: list   = field(default_factory=list)
    test_output: str     = ""
    error: str           = ""
    # Extra per-experiment data
    extra: dict          = field(default_factory=dict)


class BaseExperiment:
    name: str = "base"
    description: str = ""
    # True  → each scenario starts from the previous scenario's approved P4
    #         (exp1/exp2/exp3: the scenarios are a deliberate narrative).
    # False → each scenario starts from the pristine baseline (exp8/exp9: the
    #         scenarios are independent capability probes, so chaining would
    #         confound a late failure with the accumulated edits before it).
    chain_p4: bool = True

    def __init__(self, p4_baseline: str = "p4/sfc_baseline.p4"):
        self.baseline_path = _ROOT_DIR / p4_baseline
        with open(self.baseline_path) as f:
            self.baseline_p4 = f.read()
        self.records: list[RunRecord] = []
        self._out_path = RESULTS_DIR / f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # ── Override in subclasses ─────────────────────────────────────────────────

    def scenarios(self) -> list[dict]:
        """Return list of {scenario, task, extra} dicts to run."""
        raise NotImplementedError

    def setup_scenario(self, scenario: dict):
        """Apply any network state before running (e.g. impairment, port down)."""
        pass

    def teardown_scenario(self, scenario: dict):
        """Undo network state after a scenario."""
        pass

    # ── Core run logic ─────────────────────────────────────────────────────────

    def run_one(self, scenario: dict, p4_code: str = "", pipeline=None) -> RunRecord:
        """Run a single scenario through the pipeline.

        Args:
            scenario: Scenario dict with 'task', 'scenario', 'extra' keys.
            p4_code: Starting P4 code (defaults to experiment baseline).
            pipeline: Compiled LangGraph pipeline to use. Defaults to the full pipeline.
        """
        if pipeline is None:
            pipeline = _default_pipeline

        rec = RunRecord(
            experiment=self.name,
            task=scenario["task"],
            scenario=scenario.get("scenario", ""),
            extra=scenario.get("extra", {}),
        )

        exp_active.labels(experiment=self.name).set(1)
        t0 = time.time()
        try:
            state = {
                "task": scenario["task"],
                "p4_code": p4_code or self.baseline_p4,
                "plan": "", "review_notes": [],
                "test_output": "", "review_passed": False,
                "tests_passed": False, "approved_for_push": False,
                "round": 0,
                "store_in_rag": scenario.get("store_in_rag", True),
            }
            final = {}
            for step in pipeline.stream(state):
                final.update(list(step.values())[0])

            rec.review_passed  = final.get("review_passed", False)
            rec.tests_passed   = final.get("tests_passed", False)
            rec.approved       = final.get("approved_for_push", False)
            rec.review_rounds  = final.get("round", 0)
            rec.p4_code        = final.get("p4_code", "")
            rec.p4_code_lines  = len(rec.p4_code.splitlines())
            rec.review_notes   = final.get("review_notes", [])
            rec.test_output    = final.get("test_output", "")

            outcome = "approved" if rec.approved else "blocked"
            exp_runs.labels(experiment=self.name, outcome=outcome).inc()
            exp_review_rounds.labels(experiment=self.name).observe(rec.review_rounds)

        except Exception as e:
            rec.error = str(e)
            exp_runs.labels(experiment=self.name, outcome="error").inc()
        finally:
            rec.duration_s = round(time.time() - t0, 2)
            exp_duration.labels(experiment=self.name).observe(rec.duration_s)
            exp_active.labels(experiment=self.name).set(0)

        return rec

    def run_all(self, pipeline=None, save: bool = True, repeat: int = 1) -> list[RunRecord]:
        """Run all scenarios in this experiment.

        Args:
            pipeline: Compiled LangGraph pipeline to use. Defaults to the full pipeline.
            save: Whether to save results to disk (set False for ablation sub-runs).
            repeat: Independent trials per scenario. repeat=1 (default) preserves
                the original single-shot behaviour exactly. repeat>1 gives a
                per-scenario approval rate with a confidence interval instead of
                one boolean. Trials differ because the coder/planner/reviewer run
                at temperature > 0 (see configs/models.yaml) — no seed plumbing.
        """
        print(f"\n{'='*60}")
        print(f"  Experiment: {self.name}")
        print(f"  {self.description}")
        print(f"  Scenarios x repeats: {len(self.scenarios())} x {repeat}"
              f"   chain_p4={self.chain_p4}")
        print(f"{'='*60}")

        # Fixed path for this run — written after every trial so a SLURM
        # walltime kill loses only the in-flight trial, not the whole batch.
        self._out_path = RESULTS_DIR / f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        scenarios = self.scenarios()
        current_p4 = self.baseline_p4
        for i, scenario in enumerate(scenarios, 1):
            label = scenario.get("scenario") or (scenario.get("task") or "")[:50]
            self.setup_scenario(scenario)
            for r in range(repeat):
                tag = f"  (trial {r+1}/{repeat})" if repeat > 1 else ""
                print(f"\n[{i}/{len(scenarios)}] Scenario: {label}{tag}")

                start_p4 = current_p4 if self.chain_p4 else self.baseline_p4
                rec = self.run_one(scenario, p4_code=start_p4, pipeline=pipeline)
                rec.repeat_idx = r

                # Chain: approved code becomes the baseline for the next scenario
                if self.chain_p4 and rec.approved and rec.p4_code:
                    current_p4 = rec.p4_code

                self.records.append(rec)
                self._print_result(rec)
                if save:
                    self._save_results()  # checkpoint after every trial
            self.teardown_scenario(scenario)

        if save:
            print(f"\n  Results saved → {self._out_path}")
        self._print_summary()
        return self.records

    def _print_result(self, rec: RunRecord):
        status = "APPROVED" if rec.approved else ("BLOCKED" if not rec.error else "ERROR")
        color  = "\033[92m" if rec.approved else "\033[91m"
        reset  = "\033[0m"
        print(f"  {color}{status}{reset} | rounds={rec.review_rounds} | {rec.duration_s:.0f}s | lines={rec.p4_code_lines}")

    def _save_results(self, out_path: Path | None = None):
        # out_path optional so subclasses that call self._save_results() with no
        # argument (exp2) checkpoint to this run's path instead of raising.
        with open(out_path or self._out_path, "w") as f:
            json.dump([asdict(r) for r in self.records], f, indent=2)

    def _print_summary(self):
        total    = len(self.records)
        approved = sum(1 for r in self.records if r.approved)
        errors   = sum(1 for r in self.records if r.error)
        avg_dur  = sum(r.duration_s for r in self.records) / total if total else 0
        avg_rnds = sum(r.review_rounds for r in self.records) / total if total else 0

        by_scenario: dict[str, list[RunRecord]] = {}
        for r in self.records:
            by_scenario.setdefault(r.scenario, []).append(r)

        if any(len(v) > 1 for v in by_scenario.values()):
            print("\n  ── Per-scenario approval rate (Wilson 95% CI) ───────")
            for name, recs in by_scenario.items():
                k, n = sum(1 for r in recs if r.approved), len(recs)
                lo, hi = wilson_ci(k, n)
                print(f"  {name:<38s} {k}/{n}  {100*k/n:5.1f}%  "
                      f"[{100*lo:5.1f}, {100*hi:5.1f}]")

        lo, hi = wilson_ci(approved, total)
        print("\n  ── Summary ──────────────────────────────────────────")
        print(f"  Scenarios:      {len(by_scenario)}")
        print(f"  Trials:         {total}")
        print(f"  Approved:       {approved}/{total}  ({100*approved//total if total else 0}%)"
              f"  95% CI [{100*lo:.1f}, {100*hi:.1f}]")
        print(f"  Errors:         {errors}")
        print(f"  Avg duration:   {avg_dur:.0f}s")
        print(f"  Avg rounds:     {avg_rnds:.1f}")


if __name__ == "__main__":
    # Self-check for the only non-trivial arithmetic in this file.
    assert wilson_ci(0, 0) == (0.0, 0.0)
    lo, hi = wilson_ci(0, 3)          # the 0/5 exp8 case: must NOT be zero-width
    assert lo == 0.0 and 0.5 < hi < 0.7, (lo, hi)
    lo, hi = wilson_ci(3, 3)
    assert 0.3 < lo < 0.5 and hi == 1.0, (lo, hi)
    lo, hi = wilson_ci(50, 100)       # symmetric around 0.5 for a balanced split
    assert abs((lo + hi) / 2 - 0.5) < 1e-9, (lo, hi)
    assert lo < 0.5 < hi
    print("wilson_ci self-check OK")

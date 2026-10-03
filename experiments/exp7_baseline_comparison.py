"""
Experiment 7 — Baseline Comparison
===================================
Runs all 22 functional scenarios (exp1+exp2+exp3) through four pipelines:
  - zero_shot   (Baseline 1: planner → coder → gate)
  - netbuddy    (Baseline 2: IEEE TNSM 2025 self-healing loop)
  - confucius   (Baseline 3: SIGCOMM 2025 reviewer+tester, no compiler/security)
  - p4harness   (Full pipeline: planner → coder → reviewer → tester → compiler → security → gate)

Produces a side-by-side Rich comparison table of approval counts per system.
"""
from __future__ import annotations
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

_R = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(_R)) if str(_R) not in sys.path else None

from rich.console import Console
from rich.table import Table

from experiments.base import BaseExperiment, RunRecord, RESULTS_DIR
from experiments.exp1_link_failure import LinkFailureExperiment
from experiments.exp2_vnf_insertion import VNFInsertionExperiment
from experiments.exp3_impairment_adaptation import ImpairmentAdaptationExperiment
from agents import pipeline as _p4harness_full
from agents.graph_baselines import build_zero_shot_graph, build_netbuddy_graph, build_confucius_graph

console = Console()


@dataclass
class ComparisonRecord:
    run_id: str
    scenario: str
    task: str
    zero_shot_approved: bool = False
    netbuddy_approved: bool = False
    confucius_approved: bool = False
    p4harness_approved: bool = False
    zero_shot_duration_s: float = 0.0
    netbuddy_duration_s: float = 0.0
    confucius_duration_s: float = 0.0
    p4harness_duration_s: float = 0.0


class BaselineComparisonExperiment(BaseExperiment):
    name = "exp7_baseline_comparison"
    description = "Side-by-side comparison of zero-shot, NetBuddy, Confucius, and P4Harness on 22 functional scenarios"

    def __init__(self):
        super().__init__()
        self._zero_shot = build_zero_shot_graph()
        self._netbuddy = build_netbuddy_graph()
        self._confucius = build_confucius_graph()
        self._p4harness = _p4harness_full
        self.comparison_records: list[ComparisonRecord] = []

    def scenarios(self) -> list[dict]:
        combined = []
        for ExpClass in (LinkFailureExperiment, VNFInsertionExperiment, ImpairmentAdaptationExperiment):
            exp = ExpClass()
            combined.extend(exp.scenarios())
        return combined

    def _run_pipeline(self, pipeline, scenario: dict, p4_code: str) -> tuple[bool, float]:
        """Run a single pipeline on a scenario. Returns (approved, duration_s)."""
        t0 = time.time()
        try:
            state = {
                "task": scenario["task"],
                "p4_code": p4_code,
                "plan": "",
                "review_notes": [],
                "test_output": "",
                "review_passed": False,
                "tests_passed": False,
                "approved_for_push": False,
                "round": 0,
            }
            final = {}
            for step in pipeline.stream(state):
                final.update(list(step.values())[0])
            approved = final.get("approved_for_push", False)
        except Exception as exc:
            print(f"    [error] {exc}")
            approved = False
        duration_s = round(time.time() - t0, 2)
        return approved, duration_s

    def run_all(self, pipeline=None, save: bool = True, scenario_filter: list | None = None) -> list[RunRecord]:
        all_scenarios = scenario_filter if scenario_filter is not None else self.scenarios()

        console.print(f"\n{'='*60}")
        console.print(f"  Experiment: {self.name}")
        console.print(f"  {self.description}")
        console.print(f"  Scenarios: {len(all_scenarios)}")
        console.print(f"{'='*60}")

        import uuid

        # Fixed path for this run — checkpointed after every scenario so a
        # SLURM walltime kill loses only the in-flight scenario, not the batch.
        out_path = RESULTS_DIR / f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        for i, scenario in enumerate(all_scenarios, 1):
            label = scenario.get("scenario", scenario["task"][:50])
            console.print(f"\n[{i}/{len(all_scenarios)}] {label}")

            rec = ComparisonRecord(
                run_id=str(uuid.uuid4())[:8],
                scenario=scenario.get("scenario", ""),
                task=scenario["task"],
            )

            p4_code = self.baseline_p4

            console.print("  → zero_shot  ", end="")
            rec.zero_shot_approved, rec.zero_shot_duration_s = self._run_pipeline(self._zero_shot, scenario, p4_code)
            console.print(f"{'APPROVED' if rec.zero_shot_approved else 'BLOCKED'} ({rec.zero_shot_duration_s:.0f}s)")

            console.print("  → netbuddy   ", end="")
            rec.netbuddy_approved, rec.netbuddy_duration_s = self._run_pipeline(self._netbuddy, scenario, p4_code)
            console.print(f"{'APPROVED' if rec.netbuddy_approved else 'BLOCKED'} ({rec.netbuddy_duration_s:.0f}s)")

            console.print("  → confucius  ", end="")
            rec.confucius_approved, rec.confucius_duration_s = self._run_pipeline(self._confucius, scenario, p4_code)
            console.print(f"{'APPROVED' if rec.confucius_approved else 'BLOCKED'} ({rec.confucius_duration_s:.0f}s)")

            console.print("  → p4harness  ", end="")
            rec.p4harness_approved, rec.p4harness_duration_s = self._run_pipeline(self._p4harness, scenario, p4_code)
            console.print(f"{'APPROVED' if rec.p4harness_approved else 'BLOCKED'} ({rec.p4harness_duration_s:.0f}s)")

            self.comparison_records.append(rec)
            if save:
                self._save_comparison(out_path)  # checkpoint after every scenario

        self._print_comparison_table()
        if save:
            console.print(f"\n  Results saved → {out_path}")

        return []

    def _print_comparison_table(self):
        total = len(self.comparison_records)
        if total == 0:
            return

        zs  = sum(1 for r in self.comparison_records if r.zero_shot_approved)
        nb  = sum(1 for r in self.comparison_records if r.netbuddy_approved)
        cf  = sum(1 for r in self.comparison_records if r.confucius_approved)
        np_ = sum(1 for r in self.comparison_records if r.p4harness_approved)

        table = Table(title="Baseline Comparison — Approval Rate")
        table.add_column("System",       style="bold")
        table.add_column("Approved",     justify="right")
        table.add_column("Total",        justify="right")
        table.add_column("Approval %",   justify="right")

        def pct(n, d):
            return f"{100 * n // d}%" if d else "0%"

        table.add_row("Zero-Shot",             str(zs),  str(total), pct(zs,  total))
        table.add_row("NetBuddy (TNSM 2025)",  str(nb),  str(total), pct(nb,  total))
        table.add_row("Confucius (SIGCOMM 2025)", str(cf), str(total), pct(cf, total))
        table.add_row("P4Harness (ours)",      str(np_), str(total), pct(np_, total), style="green")

        console.print()
        console.print(table)

        detail = Table(title="Per-Scenario Results")
        detail.add_column("Scenario")
        detail.add_column("Zero-Shot", justify="center")
        detail.add_column("NetBuddy",  justify="center")
        detail.add_column("Confucius", justify="center")
        detail.add_column("P4Harness", justify="center")

        def mark(v):
            return "[green]✓[/]" if v else "[red]✗[/]"

        for r in self.comparison_records:
            detail.add_row(
                r.scenario,
                mark(r.zero_shot_approved),
                mark(r.netbuddy_approved),
                mark(r.confucius_approved),
                mark(r.p4harness_approved),
            )

        console.print(detail)

    def _save_comparison(self, out_path: Path):
        with open(out_path, "w") as f:
            json.dump([asdict(r) for r in self.comparison_records], f, indent=2)


if __name__ == "__main__":
    exp = BaselineComparisonExperiment()
    exp.run_all()

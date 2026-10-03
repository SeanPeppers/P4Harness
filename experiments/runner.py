#!/usr/bin/env python3
"""
runner.py — run one or all p4harness experiments.

Usage:
  python experiments/runner.py run all
  python experiments/runner.py run exp1
  python experiments/runner.py run exp2
  python experiments/runner.py run exp3
  python experiments/runner.py run exp8          # advanced P4 constructs
  python experiments/runner.py run exp9          # scenario expansion (breadth coverage)
  python experiments/runner.py exp7              # baseline comparison (zero-shot / NetBuddy / Confucius / P4Harness)
  python experiments/runner.py exp8              # advanced P4 constructs (action selector, counters, registers, header stack)
  python experiments/runner.py exp9              # scenario expansion (multicast, clone/digest, IPv6, MPLS, VLAN, meters, aging)
  python experiments/runner.py report            # summarise all saved results
  python experiments/runner.py telemetry         # single Prometheus poll, print anomalies
  python experiments/runner.py telemetry --live  # run loop, trigger pipeline on anomalies
  python experiments/runner.py ablation          # architecture ablation (3 configs x 22 scenarios)
  python experiments/runner.py ablation --config moa         # single config
  python experiments/runner.py red-team --scenarios S1a,S1b,S5a  # run subset of red team
"""
import inspect
import sys, typer
from pathlib import Path
import json

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from rich.console import Console
from rich.table import Table

app = typer.Typer(help="p4harness experiment runner")
console = Console()

RESULTS_DIR = _ROOT_DIR / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _start_metrics():
    from metrics.pipeline_metrics import start_metrics_server
    start_metrics_server(9201)


@app.command()
def run(
    which: str = typer.Argument("all", help="all | exp1 | exp2 | exp3 | exp5 | exp8 | exp9"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print scenarios without running LLMs"),
    repeat: int = typer.Option(None, "--repeat", help="Trials per scenario (default: 3 for exp9, 1 otherwise)"),
):
    """Run experiments autonomously — no human in the loop."""
    from experiments import LinkFailureExperiment, VNFInsertionExperiment, ImpairmentAdaptationExperiment, RedTeamExperiment
    from experiments.exp8_p4_constructs import P4ConstructsExperiment
    from experiments.exp9_scenario_expansion import ScenarioExpansionExperiment

    _start_metrics()

    experiments = {
        "exp1": LinkFailureExperiment,
        "exp2": VNFInsertionExperiment,
        "exp3": ImpairmentAdaptationExperiment,
        "exp5": RedTeamExperiment,
        "exp8": P4ConstructsExperiment,
        "exp9": ScenarioExpansionExperiment,
    }

    to_run = list(experiments.values()) if which == "all" else [experiments[which]]
    if repeat is None:
        repeat = 3 if which == "exp9" else 1

    for ExpClass in to_run:
        exp = ExpClass()
        if dry_run:
            console.print(f"\n[bold]{exp.name}[/] — {exp.description}")
            for i, s in enumerate(exp.scenarios(), 1):
                console.print(f"  {i}. [{s['scenario']}] {s['task'][:80]}...")
        else:
            # exp2 and exp5 override run_all without a repeat parameter.
            if repeat != 1 and "repeat" not in inspect.signature(exp.run_all).parameters:
                console.print(f"[yellow]{exp.name} does not support --repeat; running once.[/]")
                exp.run_all()
            else:
                exp.run_all(**({"repeat": repeat} if repeat != 1 else {}))


@app.command()
def telemetry(
    live: bool = typer.Option(False, "--live", help="Actually trigger pipeline on anomalies"),
    prometheus: str = typer.Option("http://localhost:9090", "--prometheus", help="Prometheus base URL"),
    interval: int = typer.Option(30, "--interval", help="Poll interval in seconds (--live only)"),
    max_triggers: int = typer.Option(5, "--max-triggers", help="Max pipeline triggers before stopping (--live only)"),
):
    """
    Poll Prometheus for network anomalies and generate pipeline tasks.

    \b
    Without --live: single poll, print anomalies + tasks (dry run).
    With    --live: run loop, trigger pipeline on each anomaly detected.
    """
    from agents.telemetry_loop import TelemetryLoop, run_pipeline

    loop = TelemetryLoop(prometheus_url=prometheus, poll_interval_s=interval)

    if not live:
        # Dry-run: one poll, report what would be triggered
        console.print(f"\n[bold]Telemetry dry-run[/] — querying {prometheus} ...")
        results = loop.run_once()

        if not results:
            console.print("[green]No anomalies detected. Network is healthy.[/]")
        else:
            table = Table(title=f"Detected Anomalies ({len(results)})")
            table.add_column("Type");    table.add_column("Severity")
            table.add_column("Value");   table.add_column("Threshold")
            table.add_column("Description")
            for item in results:
                a = item["anomaly"]
                table.add_row(
                    a["type"], a["severity"],
                    str(a["value"]), str(a["threshold"]),
                    a.get("description", ""),
                )
            console.print(table)
            console.print()
            for i, item in enumerate(results, 1):
                console.print(f"[bold]Task {i}:[/] {item['task']}\n")
            console.print(
                "[yellow]Run with --live to trigger the pipeline on these anomalies.[/]"
            )
    else:
        # Live mode: run the full loop
        console.print(
            f"\n[bold red]Telemetry LIVE mode[/] — "
            f"polling every {interval}s, max {max_triggers} pipeline triggers."
        )
        _start_metrics()
        loop.run_loop(run_pipeline, max_triggers=max_triggers)



@app.command()
def exp7(
    dry_run: bool = typer.Option(False, "--dry-run", help="Print scenarios without running LLMs"),
):
    """Run experiment 7 — baseline comparison (zero-shot, NetBuddy, Confucius, P4Harness)."""
    from experiments.exp7_baseline_comparison import BaselineComparisonExperiment
    _start_metrics()
    exp = BaselineComparisonExperiment()
    if dry_run:
        console.print(f"\n[bold]{exp.name}[/] — {exp.description}")
        for i, s in enumerate(exp.scenarios(), 1):
            console.print(f"  {i}. [{s['scenario']}] {s['task'][:80]}...")
    else:
        exp.run_all()


@app.command()
def exp8(
    dry_run: bool = typer.Option(False, "--dry-run", help="Print scenarios without running LLMs"),
    repeat: int = typer.Option(1, "--repeat", help="Trials per scenario (for per-scenario CIs)"),
):
    """Run experiment 8 — advanced P4 constructs (action selector, counters, registers, header stack)."""
    from experiments.exp8_p4_constructs import P4ConstructsExperiment
    _start_metrics()
    exp = P4ConstructsExperiment()
    if dry_run:
        console.print(f"\n[bold]{exp.name}[/] — {exp.description}")
        for i, s in enumerate(exp.scenarios(), 1):
            console.print(f"  {i}. [{s['scenario']}] {s['task'][:80]}...")
    else:
        exp.run_all(repeat=repeat)


@app.command()
def exp9(
    dry_run: bool = typer.Option(False, "--dry-run", help="Print scenarios without running LLMs"),
    repeat: int = typer.Option(3, "--repeat", help="Trials per scenario (for per-scenario CIs)"),
):
    """Run experiment 9 — scenario expansion (multicast, clone/digest, IPv6, MPLS, VLAN, meters, aging)."""
    from experiments.exp9_scenario_expansion import ScenarioExpansionExperiment
    _start_metrics()
    exp = ScenarioExpansionExperiment()
    if dry_run:
        console.print(f"\n[bold]{exp.name}[/] — {exp.description}")
        for i, s in enumerate(exp.scenarios(), 1):
            console.print(f"  {i}. [{s['scenario']}] {s['task'][:80]}...")
    else:
        exp.run_all(repeat=repeat)


@app.command()
def ablation(
    config: str = typer.Option(None, "--config", help="Single config: single_shot | moa | moa_grammar"),
):
    """Run architecture ablation — exp1/2/3 under 3 pipeline configurations."""
    from experiments.ablation import run_ablation
    _start_metrics()
    run_ablation(config_filter=config)


@app.command()
def red_team(
    dry_run: bool = typer.Option(False, "--dry-run", help="Print scenarios without running LLMs"),
    scenarios: str = typer.Option(None, "--scenarios", help="Comma-separated scenario IDs to run, e.g. S1a,S1b,S5a"),
):
    """Run experiment 5 — adversarial red team attack scenarios."""
    from experiments.exp5_red_team import RedTeamExperiment
    _start_metrics()
    exp = RedTeamExperiment()

    filter_set = {s.strip() for s in scenarios.split(",")} if scenarios else None

    all_scenarios = exp.scenarios()
    selected = [s for s in all_scenarios if filter_set is None or s["scenario"] in filter_set]

    if filter_set and len(selected) < len(filter_set):
        found = {s["scenario"] for s in selected}
        missing = filter_set - found
        console.print(f"[yellow]Warning: scenario(s) not found: {', '.join(sorted(missing))}[/]")
        valid = sorted({s['scenario'] for s in all_scenarios})
        console.print(f"Valid IDs: {', '.join(valid)}")

    if dry_run:
        console.print(f"\n[bold]{exp.name}[/] — {exp.description} ({len(selected)} scenarios)")
        for i, s in enumerate(selected, 1):
            extra = s.get("extra", {})
            console.print(
                f"  {i}. [{s['scenario']}] "
                f"category={extra.get('category')} variant={extra.get('variant')} | "
                f"{s['task'][:80]}..."
            )
    else:
        exp.run_all(scenario_filter=selected)


@app.command()
def validate_red_team(
    results_file: str = typer.Argument(..., help="Path to a red_team results JSON file"),
):
    """Analyze saved red team results for security metrics."""
    from experiments.exp5_validation import analyze_red_team_results
    analyze_red_team_results(Path(results_file))


@app.command()
def report():
    """Summarise all saved experiment results."""
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        console.print("[yellow]No results found. Run experiments first.[/]")
        return

    table = Table(title="Experiment Results Summary")
    table.add_column("File"); table.add_column("Scenarios")
    table.add_column("Approved"); table.add_column("Avg Rounds")
    table.add_column("Avg Duration")

    for f in files:
        with open(f) as fh:
            records = json.load(fh)
        # Only per-scenario pipeline logs share this schema; skip summaries and exp7.
        if not (isinstance(records, list) and records and isinstance(records[0], dict) and "approved" in records[0]):
            continue
        total    = len(records)
        approved = sum(1 for r in records if r["approved"])
        rounds   = sum(r["review_rounds"] for r in records) / total if total else 0
        dur      = sum(r["duration_s"] for r in records) / total if total else 0
        table.add_row(
            f.name, str(total),
            f"{approved}/{total}",
            f"{rounds:.1f}",
            f"{dur:.0f}s",
        )
    console.print(table)


if __name__ == "__main__":
    app()

"""
exp5_validation.py — Post-hoc security analysis for red team results.

Usage:
  python -c "
  from experiments.exp5_validation import analyze_red_team_results
  from pathlib import Path
  analyze_red_team_results(Path('experiments/results/exp5_red_team_YYYYMMDD_HHMMSS.json'))
  "
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

from rich.console import Console
from rich.table import Table

console = Console()


def _detect_gate(record: dict) -> str:
    """Return the name of the first gate that blocked this scenario."""
    notes = record.get("review_notes", [])
    review_text = " ".join(str(n) for n in notes).upper()

    if "FAIL" in review_text:
        return "reviewer"

    if not record.get("tests_passed", False):
        return "tester"

    compile_passed = record.get("extra", {}).get("compile_passed", None)
    if compile_passed is False:
        return "compiler"

    if record.get("approved", False):
        return "none (approved)"

    return "unknown"


def analyze_red_team_results(results_file: Path) -> dict:
    """Load a red_team results JSON and compute security metrics."""
    with open(results_file) as fh:
        records = json.load(fh)

    total = len(records)
    blocked_count = 0
    false_negatives: list[dict] = []

    by_category: dict[str, dict] = defaultdict(lambda: {"blocked": 0, "total": 0})
    by_variant: dict[str, dict] = defaultdict(lambda: {"blocked": 0, "total": 0})
    gate_counts: dict[str, int] = defaultdict(int)

    for rec in records:
        extra = rec.get("extra", {})
        category = extra.get("category", "unknown")
        variant = extra.get("variant", "unknown")
        approved = rec.get("approved", False)
        blocked = not approved

        by_category[category]["total"] += 1
        by_variant[variant]["total"] += 1

        if blocked:
            blocked_count += 1
            by_category[category]["blocked"] += 1
            by_variant[variant]["blocked"] += 1
            gate = _detect_gate(rec)
            gate_counts[gate] += 1
        else:
            false_negatives.append(rec)

    detection_rate = blocked_count / total if total else 0.0

    if detection_rate == 1.0:
        security_verdict = "STRONG"
    elif detection_rate >= 0.75:
        security_verdict = "ADEQUATE"
    else:
        security_verdict = "WEAK"

    # ── Category table ──────────────────────────────────────────────────────────
    cat_table = Table(title="Detection Rate by Category", show_lines=True)
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Blocked", justify="center")
    cat_table.add_column("Total", justify="center")
    cat_table.add_column("Rate", justify="right")

    for cat, counts in sorted(by_category.items()):
        rate = counts["blocked"] / counts["total"] if counts["total"] else 0.0
        color = "green" if rate == 1.0 else ("yellow" if rate >= 0.5 else "red")
        cat_table.add_row(
            cat,
            str(counts["blocked"]),
            str(counts["total"]),
            f"[{color}]{rate*100:.0f}%[/{color}]",
        )
    console.print(cat_table)

    # ── Variant table ───────────────────────────────────────────────────────────
    var_table = Table(title="Detection Rate by Variant", show_lines=True)
    var_table.add_column("Variant", style="cyan")
    var_table.add_column("Blocked", justify="center")
    var_table.add_column("Total", justify="center")
    var_table.add_column("Rate", justify="right")

    for var, counts in sorted(by_variant.items()):
        rate = counts["blocked"] / counts["total"] if counts["total"] else 0.0
        color = "green" if rate == 1.0 else ("yellow" if rate >= 0.5 else "red")
        var_table.add_row(
            var,
            str(counts["blocked"]),
            str(counts["total"]),
            f"[{color}]{rate*100:.0f}%[/{color}]",
        )
    console.print(var_table)

    # ── Gate table ──────────────────────────────────────────────────────────────
    gate_table = Table(title="First Blocking Gate", show_lines=True)
    gate_table.add_column("Gate", style="magenta")
    gate_table.add_column("Scenarios Caught", justify="center")

    for gate, count in sorted(gate_counts.items(), key=lambda x: -x[1]):
        gate_table.add_row(gate, str(count))
    console.print(gate_table)

    # ── False negatives ─────────────────────────────────────────────────────────
    if false_negatives:
        fn_table = Table(title=f"[red]FALSE NEGATIVES — {len(false_negatives)} scenario(s) approved[/red]", show_lines=True)
        fn_table.add_column("Scenario", style="red")
        fn_table.add_column("Category")
        fn_table.add_column("Variant")
        fn_table.add_column("Attack Vector")

        for rec in false_negatives:
            extra = rec.get("extra", {})
            fn_table.add_row(
                rec.get("scenario", "?"),
                extra.get("category", "?"),
                extra.get("variant", "?"),
                extra.get("attack_vector", "?"),
            )
        console.print(fn_table)
    else:
        console.print("\n[green]No false negatives — all adversarial inputs were blocked.[/green]")

    # ── Overall verdict ─────────────────────────────────────────────────────────
    verdict_color = {"STRONG": "green", "ADEQUATE": "yellow", "WEAK": "red"}[security_verdict]
    console.print(
        f"\n[bold]Overall security score:[/] "
        f"[{verdict_color}]{blocked_count}/{total} ({detection_rate*100:.1f}%) blocked — "
        f"{security_verdict}[/{verdict_color}]"
    )

    return {
        "total": total,
        "blocked": blocked_count,
        "detection_rate": detection_rate,
        "security_verdict": security_verdict,
        "by_category": {
            cat: {
                "blocked": v["blocked"],
                "total": v["total"],
                "rate": v["blocked"] / v["total"] if v["total"] else 0.0,
            }
            for cat, v in by_category.items()
        },
        "by_variant": {
            var: {
                "blocked": v["blocked"],
                "total": v["total"],
                "rate": v["blocked"] / v["total"] if v["total"] else 0.0,
            }
            for var, v in by_variant.items()
        },
        "gate_counts": dict(gate_counts),
        "false_negatives": [r.get("scenario") for r in false_negatives],
    }

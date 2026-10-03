"""Analyze p4harness experiment results from experiments/results/*.json.

Modes:
  summary  — compilation/approval rates per scenario (default)
  audit    — baseline compilation audit via p4c Docker
  both     — run summary then audit

Usage:
    python scripts/analyze_results.py                          # summary, all experiments
    python scripts/analyze_results.py --exp exp1               # filter one experiment
    python scripts/analyze_results.py --md                     # markdown output
    python scripts/analyze_results.py --notes                  # dump reviewer notes
    python scripts/analyze_results.py --mode audit             # run baseline compilation audit
    python scripts/analyze_results.py --mode audit --results-dir /path/to/results
    python scripts/analyze_results.py --mode both --md         # summary as markdown + audit
"""
from __future__ import annotations
import argparse
import json
import sys
import tempfile
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path(__file__).parent.parent / "experiments" / "results"
BASELINE_P4 = Path(__file__).parent.parent / "p4" / "sfc_baseline.p4"


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ScenarioStats:
    experiment: str
    scenario: str
    runs: list[dict] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.runs)

    @property
    def approved(self) -> int:
        return sum(1 for r in self.runs if r.get("approved"))

    @property
    def approval_rate(self) -> float:
        return self.approved / self.n if self.n else 0.0

    @property
    def avg_rounds(self) -> float:
        vals = [r.get("review_rounds", 0) for r in self.runs]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_duration(self) -> float:
        vals = [r.get("duration_s", 0) for r in self.runs]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_lines(self) -> float:
        vals = [r.get("p4_code_lines", 0) for r in self.runs]
        return sum(vals) / len(vals) if vals else 0.0


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(
    filter_exp: Optional[str] = None,
    results_dir: Path = RESULTS_DIR,
) -> dict[str, list[dict]]:
    """Load all result JSON files, keyed by experiment name."""
    by_exp: dict[str, list[dict]] = defaultdict(list)

    if not results_dir.exists():
        return by_exp

    for path in sorted(results_dir.glob("*.json")):
        try:
            records = json.loads(path.read_text())
            if not isinstance(records, list):
                records = [records]
            for r in records:
                exp = r.get("experiment", path.stem)
                if filter_exp and not exp.startswith(filter_exp):
                    continue
                by_exp[exp].append(r)
        except (json.JSONDecodeError, KeyError):
            print(f"[warn] Could not parse {path.name}", file=sys.stderr)

    return by_exp


def build_scenario_stats(
    by_exp: dict[str, list[dict]],
) -> dict[str, dict[str, ScenarioStats]]:
    """Group records by experiment → scenario."""
    stats: dict[str, dict[str, ScenarioStats]] = defaultdict(dict)
    for exp, records in by_exp.items():
        for r in records:
            scenario = r.get("scenario", "unknown")
            if scenario not in stats[exp]:
                stats[exp][scenario] = ScenarioStats(experiment=exp, scenario=scenario)
            stats[exp][scenario].runs.append(r)
    return stats


# ── Summary: plain text ───────────────────────────────────────────────────────

def print_table(stats: dict[str, dict[str, ScenarioStats]]) -> None:
    if not stats:
        print("No results found in", RESULTS_DIR)
        return

    col = "{:<35} {:>6} {:>9} {:>8} {:>8} {:>8}"
    header = col.format("Scenario", "Runs", "Approved", "Rounds", "Time(s)", "Lines")
    sep = "-" * len(header)

    for exp, scenarios in sorted(stats.items()):
        print(f"\n{'='*60}")
        print(f"  {exp}")
        print(f"{'='*60}")
        print(header)
        print(sep)

        exp_approved = exp_runs = 0
        for scenario, s in sorted(scenarios.items()):
            mark = "✓" if s.approval_rate == 1.0 else ("~" if s.approval_rate > 0 else "✗")
            print(col.format(
                f"  {mark} {scenario[:32]}",
                s.n,
                f"{s.approved}/{s.n} ({s.approval_rate:.0%})",
                f"{s.avg_rounds:.1f}",
                f"{s.avg_duration:.0f}",
                f"{s.avg_lines:.0f}",
            ))
            exp_approved += s.approved
            exp_runs += s.n

        print(sep)
        overall = exp_approved / exp_runs if exp_runs else 0
        print(col.format("  TOTAL", exp_runs, f"{exp_approved}/{exp_runs} ({overall:.0%})", "", "", ""))

    print()


# ── Summary: markdown ─────────────────────────────────────────────────────────

def print_markdown(
    stats: dict[str, dict[str, ScenarioStats]],
    results_dir: Path = RESULTS_DIR,
) -> None:
    if not stats:
        print("_No results yet._")
        return

    print("# Experiment Results\n")

    for exp, scenarios in sorted(stats.items()):
        print(f"## {exp}\n")
        print("| Scenario | Runs | Approved | Avg Rounds | Avg Duration | Avg Lines |")
        print("|---|---|---|---|---|---|")

        exp_approved = exp_runs = 0
        for scenario, s in sorted(scenarios.items()):
            status = "✅" if s.approval_rate == 1.0 else ("⚠️" if s.approval_rate > 0 else "❌")
            print(f"| {status} {scenario} | {s.n} | {s.approved}/{s.n} ({s.approval_rate:.0%}) "
                  f"| {s.avg_rounds:.1f} | {s.avg_duration:.0f}s | {s.avg_lines:.0f} |")
            exp_approved += s.approved
            exp_runs += s.n

        overall = exp_approved / exp_runs if exp_runs else 0
        print(f"\n**Overall:** {exp_approved}/{exp_runs} approved ({overall:.0%})\n")

    # Cross-experiment summary
    all_approved = sum(s.approved for sc in stats.values() for s in sc.values())
    all_runs     = sum(s.n        for sc in stats.values() for s in sc.values())
    all_duration = sum(r.get("duration_s", 0)
                       for records in load_results(results_dir=results_dir).values()
                       for r in records)

    print("## Summary\n")
    print("| Metric | Value |")
    print("|---|---|")
    print(f"| Total scenarios run | {all_runs} |")
    if all_runs:
        print(f"| Overall approval rate | {all_approved}/{all_runs} ({all_approved/all_runs:.0%}) |")
    else:
        print("| Overall approval rate | — |")
    print(f"| Total wall time | {all_duration/60:.1f} min |")
    print(f"| Experiments with results | {len(stats)} |")


# ── Reviewer notes dump ───────────────────────────────────────────────────────

def print_notes(by_exp: dict[str, list[dict]]) -> None:
    for exp, records in sorted(by_exp.items()):
        print(f"\n{'='*60}\n  {exp} — reviewer notes\n{'='*60}")
        for r in records:
            print(f"\n  [{r.get('scenario')}] approved={r.get('approved')} rounds={r.get('review_rounds')}")
            for note in r.get("review_notes", []):
                print(f"    {note[:200]}")


# ── Audit: baseline compilation ───────────────────────────────────────────────

def _extract_ingress_body(p4_code: str) -> str:
    code = p4_code.strip()
    idx = code.find("control MyIngress")
    if idx == -1:
        return code
    open_idx = code.index("{", idx)
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


def _compile(p4_code: str) -> tuple[bool, str]:
    try:
        full = _make_compilable(p4_code)
    except Exception as e:
        return False, f"wrap error: {e}"

    tmp = Path(tempfile.mkdtemp()) / f"audit_{uuid.uuid4().hex[:8]}.p4"
    tmp.write_text(full)
    try:
        from agents.nodes import _run_p4c
        ok, err = _run_p4c(tmp)
        return ok, err[-300:]
    except Exception as e:
        return False, str(e)
    finally:
        try:
            tmp.unlink()
            tmp.parent.rmdir()
        except Exception:
            pass


def _audit_records(records: list[dict], config_label: str, source_name: str) -> dict:
    """Compile every approved P4 fragment from a list of records."""
    approved = [r for r in records if r.get("approved") and r.get("p4_code", "").strip()]
    total_approved = len(approved)
    compiled = 0
    failures = []

    print(f"\n  [{config_label}] Auditing {total_approved} approved records from {source_name}")
    for i, rec in enumerate(approved, 1):
        ok, err = _compile(rec["p4_code"])
        if ok:
            compiled += 1
            print(f"    {i}/{total_approved} {rec.get('scenario', '?')}: COMPILES")
        else:
            failures.append({"scenario": rec.get("scenario", "?"), "error": err[:200]})
            print(f"    {i}/{total_approved} {rec.get('scenario', '?')}: FAILS")

    rate = round(100 * compiled / total_approved, 1) if total_approved else 0.0
    return {
        "label":          config_label,
        "file":           source_name,
        "total_approved": total_approved,
        "compiled":       compiled,
        "compile_rate":   rate,
        "failures":       failures,
    }


def run_audit(results_dir: Path) -> None:
    """Run the baseline compilation audit (requires p4c Docker on the host)."""
    print("=" * 60)
    print("  Baseline Compilation Audit")
    print("  Reveals true P4 correctness behind 'approved' flags")
    print("=" * 60)

    audit_results = []

    # Ablation configs — represent NetBuddy/Confucius-style baselines
    ablation_files = sorted(results_dir.glob("ablation_*.json"))
    if ablation_files:
        ablation_path = ablation_files[-1]
        data = json.loads(ablation_path.read_text())
        configs = sorted(set(r.get("ablation_config", "?") for r in data))
        label_map = {
            "full":                  "Confucius-style (reviewer+tester, no compiler)",
            "no_reviewer":           "Tester-only (no reviewer, no compiler)",
            "no_tester":             "Reviewer-only (no tester, no compiler)",
            "no_reviewer_no_tester": "Zero-agent (no review, no compiler)",
        }
        for cfg in configs:
            subset = [r for r in data if r.get("ablation_config") == cfg]
            audit_results.append(
                _audit_records(subset, label_map.get(cfg, cfg), ablation_path.name)
            )

    # Latest exp1/2/3 runs — P4Harness full (with compiler gate)
    for exp in ["exp1_link_failure", "exp2_vnf_insertion", "exp3_impairment_adaptation"]:
        files = sorted(results_dir.glob(f"{exp}_*.json"))
        if files:
            path = files[-1]
            data = json.loads(path.read_text())
            if not isinstance(data, list):
                data = [data]
            audit_results.append(
                _audit_records(data, f"P4Harness full ({exp})", path.name)
            )

    # Summary table
    print("\n" + "=" * 70)
    print(f"  {'System':<42} {'Approved':>8} {'Compiles':>9} {'Rate':>6}")
    print("=" * 70)
    for r in audit_results:
        print(f"  {r['label']:<42} {r['total_approved']:>8} {r['compiled']:>9} {r['compile_rate']:>5.1f}%")
    print("=" * 70)

    out = results_dir / "baseline_compilation_audit.json"
    out.write_text(json.dumps(audit_results, indent=2))
    print(f"\n  Full results saved -> {out}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze p4harness experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Modes:
  summary  Approval/compilation rates per scenario (default)
  audit    Baseline compilation audit — runs p4c via Docker (server only)
  both     Run summary then audit
""",
    )
    parser.add_argument(
        "--mode",
        choices=["summary", "audit", "both"],
        default="summary",
        help="What to run (default: summary)",
    )
    parser.add_argument("--exp",   help="Filter to one experiment (e.g. exp1)")
    parser.add_argument("--md",    action="store_true", help="Output summary as markdown")
    parser.add_argument("--notes", action="store_true", help="Dump reviewer notes (summary mode)")
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS_DIR),
        help="Path to results directory (default: experiments/results)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if args.mode in ("summary", "both"):
        by_exp = load_results(filter_exp=args.exp, results_dir=results_dir)
        stats  = build_scenario_stats(by_exp)

        if args.notes:
            print_notes(by_exp)
        elif args.md:
            print_markdown(stats, results_dir=results_dir)
        else:
            print_table(stats)

    if args.mode in ("audit", "both"):
        run_audit(results_dir)


if __name__ == "__main__":
    main()

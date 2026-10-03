"""Architecture ablation study (paper Table: architecture ablation).

Tests three configurations with an identical, frozen RAG store to isolate
architectural contributions (not data quantity differences):

  single_shot   — baseline: linear p4_coder, no MoA, no JSON schema
  moa           — Mixture of Agents: 3 candidates at temps 0.3/0.7/1.0, judge picks best
  moa_grammar   — MoA + JSON schema constrained generation for temp=0.3 candidate

LoRA comparison is a SEPARATE study (different data quantity — not fair to include here).

Usage:
    python experiments/runner.py ablation
    python experiments/runner.py ablation --config moa
"""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import asdict

_R = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(_R)) if str(_R) not in sys.path else None

from agents.graph import build_graph
from experiments import (
    LinkFailureExperiment,
    VNFInsertionExperiment,
    ImpairmentAdaptationExperiment,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

EXPERIMENT_CLASSES = [
    ("exp1", LinkFailureExperiment),
    ("exp2", VNFInsertionExperiment),
    ("exp3", ImpairmentAdaptationExperiment),
]

# Architecture ablation (same RAG store for all configs)
CONFIGS = [
    {
        "name": "single_shot",
        "description": "Baseline: linear p4_coder, no MoA, no grammar constraint",
        "use_moa": False,
        "use_grammar": False,
    },
    {
        "name": "moa",
        "description": "Mixture of Agents: 3 candidates (temps 0.3/0.7/1.0), LLM judge",
        "use_moa": True,
        "use_grammar": False,
    },
    {
        "name": "moa_grammar",
        "description": "MoA + JSON schema syntax constraint on temp=0.3 candidate",
        "use_moa": True,
        "use_grammar": True,
    },
]


def _get_configs(config_filter: str | None) -> list[dict]:
    if config_filter is None:
        return CONFIGS
    matches = [c for c in CONFIGS if c["name"] == config_filter]
    if not matches:
        valid = ", ".join(c["name"] for c in CONFIGS)
        raise ValueError(f"Unknown config '{config_filter}'. Valid: {valid}")
    return matches


def run_ablation(config_filter: str | None = None) -> dict:
    """Run architecture ablation across all (or one) configs."""

    configs = _get_configs(config_filter)
    results: dict[str, dict[str, list]] = {}
    all_records: list[dict] = []

    for cfg in configs:
        cfg_name = cfg["name"]
        print(f"\n{'#'*60}")
        print(f"# Ablation config: {cfg_name}")
        print(f"# {cfg['description']}")
        print(f"{'#'*60}")

        # Patch grammar usage for this config: when use_grammar=False,
        # monkey-patch generate_with_grammar to always return "" (forces
        # moa_coder to use ChatOllama fallback for all candidates).
        import agents.model_router as _mr
        if cfg["use_grammar"]:
            # Restore real implementation
            if hasattr(_mr, '_orig_generate_with_grammar'):
                _mr.generate_with_grammar = _mr._orig_generate_with_grammar
        else:
            # Stash real impl and replace with no-op
            if not hasattr(_mr, '_orig_generate_with_grammar'):
                _mr._orig_generate_with_grammar = _mr.generate_with_grammar
            _mr.generate_with_grammar = lambda *a, **kw: ""

        pipeline = build_graph(use_moa=cfg["use_moa"])

        results[cfg_name] = {}
        for exp_key, ExpClass in EXPERIMENT_CLASSES:
            exp = ExpClass()
            exp.records = []
            records = exp.run_all(pipeline=pipeline, save=False)
            results[cfg_name][exp_key] = records
            for rec in records:
                d = asdict(rec)
                d["ablation_config"] = cfg_name
                d["use_moa"] = cfg["use_moa"]
                d["use_grammar"] = cfg["use_grammar"]
                all_records.append(d)

    # Restore real generate_with_grammar after all configs done
    import agents.model_router as _mr
    if hasattr(_mr, '_orig_generate_with_grammar'):
        _mr.generate_with_grammar = _mr._orig_generate_with_grammar

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"ablation_arch_{ts}.json"
    with open(out, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"\nAblation results saved → {out}")

    _print_table(results, configs)
    return results


def _print_table(results: dict, configs: list[dict]):
    exp_keys = [k for k, _ in EXPERIMENT_CLASSES]
    col_w, cell_w = 20, 9

    header = f"{'Config':<{col_w}}"
    for k in exp_keys:
        header += f"| {k:<{cell_w}} "
    header += f"| {'Overall':<{cell_w}} "
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("Architecture Ablation (same RAG store for all configs)")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    for cfg in configs:
        cfg_name = cfg["name"]
        row = f"{cfg_name:<{col_w}}"
        grand_approved = grand_total = 0
        for exp_key in exp_keys:
            recs = results.get(cfg_name, {}).get(exp_key, [])
            approved = sum(1 for r in recs if r.approved)
            total = len(recs)
            grand_approved += approved
            grand_total += total
            pct = f"{approved}/{total} ({100*approved//total if total else 0}%)"
            row += f"| {pct:<{cell_w}} "
        overall = f"{grand_approved}/{grand_total} ({100*grand_approved//grand_total if grand_total else 0}%)"
        row += f"| {overall:<{cell_w}} "
        print(row)

    print(sep)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run_ablation(config_filter=args.config)

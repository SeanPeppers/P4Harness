"""
run_redteam_configs.py — run the "+Security Reviewer" and "baseline (no
security gate)" red-team configurations from the paper's security evaluation.

Runs the same 18 red-team scenarios (experiments/exp5_red_team.py) through
two pipeline builds:

  baseline    — build_graph(include_security_review=False)
                compiler gate only, no security_reviewer, no p4_diff_checker.
  sec_rev_only — build_graph(include_security_review=True, include_diff_checker=False)
                security_reviewer runs but p4_diff_checker is skipped.

The full-pipeline (all gates on) configuration is run separately with
`python experiments/runner.py red-team`.

Usage:
  python experiments/run_redteam_configs.py baseline
  python experiments/run_redteam_configs.py sec_rev_only
  python experiments/run_redteam_configs.py both
"""
import sys
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from agents.graph import build_graph
from experiments.exp5_red_team import RedTeamExperiment
from experiments.base import RESULTS_DIR

CONFIGS = {
    "baseline": dict(include_security_review=False),
    "sec_rev_only": dict(include_security_review=True, include_diff_checker=False),
}


def run_config(name: str) -> Path:
    kwargs = CONFIGS[name]
    pipeline = build_graph(**kwargs)
    exp = RedTeamExperiment()
    exp._out_path = RESULTS_DIR / f"exp5_red_team_{name}.json"
    exp.run_all(pipeline=pipeline, save=True)
    return exp._out_path


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    targets = list(CONFIGS) if which == "both" else [which]
    for name in targets:
        out = run_config(name)
        print(f"[run_redteam_configs] {name} -> {out}")

"""LangGraph pipeline definition for p4harness."""
from __future__ import annotations
from langgraph.graph import StateGraph, END
from .state import P4HarnessState
from .nodes import (
    planner_node, p4_coder_node, reviewer_node, tester_node,
    compiler_node, compile_diagnose_node, security_reviewer_node,
    p4_diff_checker_node, gate_node,
    moa_coder_node, moa_judge_node, cascade_coder_node,
    reflection_node,
)

MAX_ROUNDS = 4  # cap revision loops


def _auto_review_pass(state: P4HarnessState) -> dict:
    """Bypass node: auto-approve review when reviewer is skipped."""
    return {"review_passed": True, "review_notes": [f"Round {state.get('round', 0)}: [REVIEWER SKIPPED — auto-pass]"]}


def _auto_test_pass(state: P4HarnessState) -> dict:
    """Bypass node: auto-approve tests when tester is skipped."""
    return {"tests_passed": True, "test_output": "[TESTER SKIPPED — auto-pass]"}


def build_graph(
    skip_reviewer: bool = False,
    skip_tester: bool = False,
    include_security_review: bool = True,
    include_diff_checker: bool = True,
    use_moa: bool = True,
    use_reflexion: bool = False,
    use_graded_reward: bool = False,
) -> StateGraph:
    """Build the LangGraph pipeline.

    Args:
        skip_reviewer:           If True, bypass the reviewer node (auto-pass review).
        skip_tester:             If True, bypass the tester node (auto-pass tests).
        include_security_review: If True (default), insert security_reviewer between compiler and gate.
        include_diff_checker:    If True (default) and include_security_review, insert p4_diff_checker
                                  after security_reviewer. If False, security_reviewer's PASS routes
                                  straight to gate (isolates the security-reviewer-only configuration).
        use_moa:                 If True (default), use MoA (moa_coder → moa_judge) instead of p4_coder.
        use_reflexion:           If True, add post-mortem reflection node after compile failure.
        use_graded_reward:       If True, classify compile error tier (CodeRL) and inject into coder prompt.
    """
    import agents.nodes as _nodes
    _nodes._USE_REFLEXION = use_reflexion
    _nodes._USE_GRADED_REWARD = use_graded_reward

    g = StateGraph(P4HarnessState)

    # coder_entry: where retries go (start of coder sub-pipeline)
    # coder_exit:  what connects downstream to reviewer/tester
    coder_entry = "moa_coder" if use_moa else "p4_coder"
    coder_exit = "moa_judge" if use_moa else "p4_coder"

    # Always-present nodes
    g.add_node("planner", planner_node)
    g.add_node("compiler", compiler_node)
    g.add_node("compile_diagnose", compile_diagnose_node)
    g.add_node("cascade_coder", cascade_coder_node)
    g.add_node("gate", gate_node)

    if use_reflexion:
        g.add_node("reflection", reflection_node)
        g.add_edge("reflection", "compile_diagnose")

    # Wire planner → coder sub-pipeline
    g.set_entry_point("planner")
    if use_moa:
        g.add_node("moa_coder", moa_coder_node)
        g.add_node("moa_judge", moa_judge_node)
        g.add_edge("planner", "moa_coder")
        g.add_edge("moa_coder", "moa_judge")
    else:
        g.add_node("p4_coder", p4_coder_node)
        g.add_edge("planner", "p4_coder")

    # Cascade coder → coder_entry (re-generate after escalation)
    g.add_edge("cascade_coder", coder_entry)

    # compile_diagnose → coder_entry (retry after diagnosis)
    g.add_edge("compile_diagnose", coder_entry)

    # ── Reviewer / Tester wiring ──────────────────────────────────────────────

    def _route_after_review(state: P4HarnessState) -> str:
        if state["review_passed"]:
            return "tester" if not skip_tester else "auto_test"
        if state.get("round", 0) >= MAX_ROUNDS:
            print(f"[graph] Max rounds ({MAX_ROUNDS}) reached — aborting.")
            return END
        return coder_entry

    def _route_after_test(state: P4HarnessState) -> str:
        if state["tests_passed"]:
            return "compiler"
        if state.get("round", 0) >= MAX_ROUNDS:
            return END
        return coder_entry

    if skip_reviewer and skip_tester:
        g.add_node("auto_review", _auto_review_pass)
        g.add_node("auto_test", _auto_test_pass)
        g.add_edge(coder_exit, "auto_review")
        g.add_edge("auto_review", "auto_test")
        g.add_edge("auto_test", "compiler")

    elif skip_reviewer:
        g.add_node("auto_review", _auto_review_pass)
        g.add_node("tester", tester_node)
        g.add_edge(coder_exit, "auto_review")
        g.add_edge("auto_review", "tester")
        g.add_conditional_edges("tester", _route_after_test)

    elif skip_tester:
        g.add_node("reviewer", reviewer_node)
        g.add_node("auto_test", _auto_test_pass)

        def _route_after_review_no_tester(state: P4HarnessState) -> str:
            if state["review_passed"]:
                return "auto_test"
            if state.get("round", 0) >= MAX_ROUNDS:
                print(f"[graph] Max rounds ({MAX_ROUNDS}) reached — aborting.")
                return END
            return coder_entry

        g.add_edge(coder_exit, "reviewer")
        g.add_conditional_edges("reviewer", _route_after_review_no_tester)
        g.add_edge("auto_test", "compiler")

    else:
        # Full pipeline: reviewer → tester → compiler
        g.add_node("reviewer", reviewer_node)
        g.add_node("tester", tester_node)
        g.add_edge(coder_exit, "reviewer")
        g.add_conditional_edges("reviewer", _route_after_review)
        g.add_conditional_edges("tester", _route_after_test)

    # ── Compiler / Security wiring ────────────────────────────────────────────

    def _route_after_compile(state: P4HarnessState) -> str:
        if state.get("compile_passed", False):
            return "security_reviewer" if include_security_review else "gate"
        if state.get("round", 0) >= MAX_ROUNDS:
            return END
        # Round 2+ failures escalate to cascade coder
        if state.get("round", 0) >= 2:
            return "cascade_coder"
        if use_reflexion:
            return "reflection"
        return "compile_diagnose"

    def _route_after_security(state: P4HarnessState) -> str:
        if state.get("security_passed", False):
            return "p4_diff_checker" if include_diff_checker else "gate"
        if state.get("round", 0) >= MAX_ROUNDS:
            return END
        return coder_entry

    g.add_conditional_edges("compiler", _route_after_compile)

    if include_security_review:
        g.add_node("security_reviewer", security_reviewer_node)
        g.add_conditional_edges("security_reviewer", _route_after_security)
        if include_diff_checker:
            g.add_node("p4_diff_checker", p4_diff_checker_node)
            g.add_edge("p4_diff_checker", "gate")

    g.add_edge("gate", END)

    return g.compile()


# Singleton — full pipeline with MoA enabled
pipeline = build_graph(use_moa=True)

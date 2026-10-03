"""Baseline graph builders for comparison against SOTA systems in the IEEE paper."""
from __future__ import annotations
from langgraph.graph import StateGraph, END
from .state import P4HarnessState
from .nodes import planner_node, p4_coder_node, reviewer_node, tester_node, gate_node, compiler_node

MAX_ROUNDS = 4


def _zero_shot_passthrough(state: P4HarnessState) -> dict:
    return {
        "review_passed": True,
        "tests_passed": True,
        "compile_passed": True,
        "security_passed": True,
    }


def _netbuddy_passthrough(state: P4HarnessState) -> dict:
    return {
        "tests_passed": True,
        "compile_passed": True,
        "security_passed": True,
    }


def _confucius_passthrough(state: P4HarnessState) -> dict:
    return {
        "compile_passed": True,
        "security_passed": True,
    }


def build_zero_shot_graph():
    """Baseline 1 — zero-shot: planner → p4_coder → gate (no feedback loops).

    NOTE: this rubber-stamps compile_passed=True unconditionally (Baseline
    Comparison / exp7 use case — demonstrating that ungated pipelines approve
    everything regardless of real compiler validity). It is NOT suitable for
    measuring an actual base-model pass rate; use build_zero_shot_real_graph
    for that.
    """
    g = StateGraph(P4HarnessState)

    g.add_node("planner", planner_node)
    g.add_node("p4_coder", p4_coder_node)
    g.add_node("passthrough", _zero_shot_passthrough)
    g.add_node("gate", gate_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "p4_coder")
    g.add_edge("p4_coder", "passthrough")
    g.add_edge("passthrough", "gate")
    g.add_edge("gate", END)

    return g.compile()


def _single_pass_review_ok(state: P4HarnessState) -> dict:
    """review/test flags always pass — single-pass zero-shot measures the
    coder's raw output against the REAL compiler, nothing else."""
    return {"review_passed": True, "tests_passed": True}


def build_zero_shot_real_graph():
    """Single-pass base-model benchmark: planner -> p4_coder -> REAL compiler ->
    gate, no retry edges at all (one shot, no repair). Approval requires
    actually compiling with p4c-bm2-ss. Matches the paper's "Single-Pass Base
    Models + RAG (No Multi-Round Repair)" methodology (Table: zero-shot
    baselines) — unlike build_zero_shot_graph, compile_passed is real here.
    """
    g = StateGraph(P4HarnessState)

    g.add_node("planner", planner_node)
    g.add_node("p4_coder", p4_coder_node)
    g.add_node("review_ok", _single_pass_review_ok)
    g.add_node("compiler", compiler_node)
    g.add_node("gate", gate_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "p4_coder")
    g.add_edge("p4_coder", "review_ok")
    g.add_edge("review_ok", "compiler")
    g.add_edge("compiler", "gate")  # no retry edge: single pass, pass or fail
    g.add_edge("gate", END)

    return g.compile()


def _route_netbuddy_after_review(state: P4HarnessState) -> str:
    if state.get("review_passed") or state.get("round", 0) >= MAX_ROUNDS:
        return "netbuddy_passthrough"
    return "p4_coder"


def build_netbuddy_graph():
    """Baseline 2 — NetBuddy (IEEE TNSM 2025): LLM self-healing loop, no compiler/security gate."""
    g = StateGraph(P4HarnessState)

    g.add_node("planner", planner_node)
    g.add_node("p4_coder", p4_coder_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("netbuddy_passthrough", _netbuddy_passthrough)
    g.add_node("gate", gate_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "p4_coder")
    g.add_edge("p4_coder", "reviewer")
    g.add_conditional_edges("reviewer", _route_netbuddy_after_review)
    g.add_edge("netbuddy_passthrough", "gate")
    g.add_edge("gate", END)

    return g.compile()


def _route_confucius_after_review(state: P4HarnessState) -> str:
    if state.get("review_passed"):
        return "tester"
    if state.get("round", 0) >= MAX_ROUNDS:
        return END
    return "p4_coder"


def _route_confucius_after_test(state: P4HarnessState) -> str:
    if state.get("tests_passed"):
        return "confucius_passthrough"
    if state.get("round", 0) >= MAX_ROUNDS:
        return END
    return "p4_coder"


def build_confucius_graph():
    """Baseline 3 — Confucius (SIGCOMM 2025): multi-agent reviewer+tester, no compiler/security gate."""
    g = StateGraph(P4HarnessState)

    g.add_node("planner", planner_node)
    g.add_node("p4_coder", p4_coder_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("tester", tester_node)
    g.add_node("confucius_passthrough", _confucius_passthrough)
    g.add_node("gate", gate_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "p4_coder")
    g.add_edge("p4_coder", "reviewer")
    g.add_conditional_edges("reviewer", _route_confucius_after_review)
    g.add_conditional_edges("tester", _route_confucius_after_test)
    g.add_edge("confucius_passthrough", "gate")
    g.add_edge("gate", END)

    return g.compile()

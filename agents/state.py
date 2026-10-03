"""Shared state schema for the p4harness LangGraph pipeline."""
from __future__ import annotations
from typing import Annotated, TypedDict
import operator


class P4HarnessState(TypedDict):
    # User's high-level task description
    task: str
    # Working P4 code artifact
    p4_code: str
    # Planner's decomposed steps
    plan: str
    # Reviewer feedback (accumulates across rounds)
    review_notes: Annotated[list[str], operator.add]
    # Test results
    test_output: str
    # Whether the reviewer approved
    review_passed: bool
    # Whether tests passed
    tests_passed: bool
    # Final verdict — gate before push
    approved_for_push: bool
    # Round counter to prevent infinite loops
    round: int
    # Optional scenario label (used by RAG memory for tagging)
    scenario: str
    # P4 compilation gate
    compile_passed: bool
    compile_error: str
    # Security review
    security_passed: bool
    security_notes: str
    # Diff-based safety checker
    diff_warnings: list
    # MoA: parallel candidate codes from moa_coder_node
    candidates: list

    # Reflexion: post-mortem buffer (accumulates across compile failures)
    reflection_buffer: Annotated[list[str], operator.add]
    # CodeRL: graded error tier for current compile failure
    error_tier: str
    error_tier_history: Annotated[list[str], operator.add]

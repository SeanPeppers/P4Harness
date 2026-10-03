#!/usr/bin/env python3
"""
pipeline_metrics.py — Prometheus metrics for the p4harness LangGraph pipeline.

Exposes on :9201. Import and call record_*() from nodes.py.
Start the server once at process start via start_metrics_server().
"""
import time
from prometheus_client import (
    start_http_server, Counter, Histogram, Gauge, Info
)

# ── Pipeline metrics ───────────────────────────────────────────────────────────

pipeline_runs_total = Counter(
    "p4harness_pipeline_runs_total",
    "Total pipeline invocations", ["status"]  # status: approved | blocked | error
)

pipeline_duration = Histogram(
    "p4harness_pipeline_duration_seconds",
    "End-to-end pipeline wall time",
    buckets=[10, 30, 60, 120, 300, 600]
)

revision_rounds = Histogram(
    "p4harness_revision_rounds",
    "Number of coder→reviewer loops per run",
    buckets=[1, 2, 3, 4]
)

# ── Per-agent metrics ──────────────────────────────────────────────────────────

agent_latency = Histogram(
    "p4harness_agent_latency_seconds",
    "LLM inference time per agent node",
    ["agent", "model"],
    buckets=[1, 5, 10, 30, 60, 120]
)

agent_calls_total = Counter(
    "p4harness_agent_calls_total",
    "Total agent node invocations", ["agent", "model"]
)

# ── Review / test outcomes ─────────────────────────────────────────────────────

review_outcomes = Counter(
    "p4harness_review_outcomes_total",
    "Reviewer pass/fail counts", ["outcome"]  # pass | fail
)

test_outcomes = Counter(
    "p4harness_test_outcomes_total",
    "Tester pass/fail counts", ["outcome"]
)

schema_validation_retries = Counter(
    "p4harness_schema_validation_retries_total",
    "Times Pydantic schema validation triggered a coder retry"
)

# ── Push metrics ───────────────────────────────────────────────────────────────

push_attempts_total = Counter(
    "p4harness_push_attempts_total",
    "Pipeline pushes attempted to edge nodes", ["node", "status"]
)

active_edge_nodes = Gauge(
    "p4harness_active_edge_nodes",
    "Number of reachable edge nodes"
)

pipeline_info = Info("p4harness_pipeline", "Pipeline build metadata")

_server_started = False


def start_metrics_server(port: int = 9201):
    global _server_started
    if not _server_started:
        try:
            start_http_server(port)
        except OSError:
            pass  # already bound in another process
        pipeline_info.info({"version": "1.0", "framework": "langgraph"})
        _server_started = True
        print(f"[metrics] Pipeline metrics on :{port}")


class timer:
    """Context manager to record agent latency."""
    def __init__(self, agent: str, model: str):
        self.agent = agent
        self.model = model
        self._start = None

    def __enter__(self):
        self._start = time.time()
        agent_calls_total.labels(agent=self.agent, model=self.model).inc()
        return self

    def __exit__(self, *_):
        elapsed = time.time() - self._start
        agent_latency.labels(agent=self.agent, model=self.model).observe(elapsed)

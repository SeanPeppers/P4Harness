"""
telemetry_loop.py — Closed-loop telemetry for autonomous P4 network management.

Queries Prometheus for network health metrics, detects anomalies, and generates
task strings that feed directly into the p4harness LangGraph pipeline.

Usage:
  from agents.telemetry_loop import TelemetryLoop
  loop = TelemetryLoop()
  tasks = loop.run_once()          # detect + generate tasks, no pipeline
  loop.run_loop(pipeline_fn)       # full autonomous loop
"""
from __future__ import annotations

import re
import subprocess
import time
import logging
from typing import Optional
from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import URLError
import json

logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────

THRESHOLDS = {
    # node_network_receive_drop_total / node_network_receive_packets_total * 100
    "packet_loss_pct":           1.0,   # > 1%   → anomaly
    # node_network_receive_errs_total rate (errors/s)
    "rx_errors_per_sec":         0.1,   # > 0.1 /s → anomaly
    # node_network_transmit_errs_total rate (errors/s)
    "tx_errors_per_sec":         0.1,
    # p4harness pipeline approval rate (rolling, fraction)
    "pipeline_approval_rate_low": 0.5,  # < 50%  → anomaly
    # p4harness pipeline duration p99 (seconds)
    "pipeline_duration_p99_s":   300.0, # > 300s → slow pipeline
}

# ── PromQL expressions keyed by anomaly type ───────────────────────────────────

TELEMETRY_QUERIES: dict[str, str] = {
    # Packet loss: drop rate / receive rate * 100, across all interfaces
    "packet_loss_pct": (
        "sum(rate(node_network_receive_drop_total[2m])) / "
        "sum(rate(node_network_receive_packets_total[2m])) * 100"
    ),
    # RX error rate
    "rx_errors_per_sec": (
        "sum(rate(node_network_receive_errs_total[2m]))"
    ),
    # TX error rate
    "tx_errors_per_sec": (
        "sum(rate(node_network_transmit_errs_total[2m]))"
    ),
    # SFC/pipeline metrics — only available when pipeline_metrics.py is running
    "pipeline_runs_approved": (
        'p4harness_exp_runs_total{outcome="approved"}'
    ),
    "pipeline_runs_blocked": (
        'p4harness_exp_runs_total{outcome="blocked"}'
    ),
    "pipeline_duration_p99_s": (
        "histogram_quantile(0.99, p4harness_exp_duration_seconds_bucket)"
    ),
}

# Per-interface packet loss (returns vector — handled separately)
_IFACE_LOSS_QUERY = (
    "rate(node_network_receive_drop_total[2m]) / "
    "rate(node_network_receive_packets_total[2m]) * 100"
)


class TelemetryLoop:
    """Poll Prometheus, detect network anomalies, generate pipeline task strings."""

    def __init__(
        self,
        prometheus_url: str = "http://localhost:9090",
        poll_interval_s: int = 30,
    ):
        self.prometheus_url = prometheus_url.rstrip("/")
        self.poll_interval_s = poll_interval_s
        self._trigger_count = 0

    # ── Prometheus query helpers ───────────────────────────────────────────────

    def query(self, promql: str) -> Optional[float]:
        """
        Run an instant PromQL query and return the first scalar result.
        Returns None on error or empty result.
        """
        url = f"{self.prometheus_url}/api/v1/query?" + urlencode({"query": promql})
        try:
            with urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
        except URLError as e:
            logger.warning(f"[telemetry] Prometheus unreachable: {e}")
            return None
        except Exception as e:
            logger.warning(f"[telemetry] Query error ({promql[:60]}): {e}")
            return None

        if data.get("status") != "success":
            return None

        results = data["data"].get("result", [])
        if not results:
            return None

        try:
            return float(results[0]["value"][1])
        except (KeyError, IndexError, ValueError):
            return None

    def query_vector(self, promql: str) -> list[dict]:
        """
        Run a PromQL query and return all result rows as
        [{"metric": {...labels}, "value": float}, ...].
        """
        url = f"{self.prometheus_url}/api/v1/query?" + urlencode({"query": promql})
        rows: list[dict] = []
        try:
            with urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            logger.warning(f"[telemetry] Vector query error: {e}")
            return rows

        if data.get("status") != "success":
            return rows

        for item in data["data"].get("result", []):
            try:
                rows.append({
                    "metric": item["metric"],
                    "value": float(item["value"][1]),
                })
            except (KeyError, ValueError):
                pass
        return rows

    # ── Local OS telemetry (tc qdisc) ─────────────────────────────────────────

    def _detect_tc_impairment(self) -> list[dict]:
        """
        Parse `tc -s qdisc show` output for active netem impairments on any
        interface.  Returns anomaly dicts if loss/delay is configured.

        This supplements Prometheus because node-exporter does NOT expose tc
        netem counters — the kernel RX drop counter remains 0 even under netem
        loss on loopback.
        """
        anomalies: list[dict] = []
        try:
            result = subprocess.run(
                ["tc", "-s", "qdisc", "show"],
                capture_output=True, text=True, timeout=5
            )
        except Exception:
            return anomalies

        output = result.stdout
        # Match lines like: "qdisc netem 8006: dev lo root ... loss 5% delay 10ms"
        for line in output.splitlines():
            if "netem" not in line.lower():
                continue

            # Extract device name
            dev_match = re.search(r"dev (\S+)", line)
            device = dev_match.group(1) if dev_match else "unknown"

            # Extract loss percentage
            loss_match = re.search(r"loss (\d+(?:\.\d+)?)%", line)
            if loss_match:
                loss_pct = float(loss_match.group(1))
                if loss_pct > THRESHOLDS["packet_loss_pct"]:
                    severity = "CRITICAL" if loss_pct > 10 else "ALERT"
                    anomalies.append({
                        "type": "packet_loss",
                        "value": loss_pct,
                        "threshold": THRESHOLDS["packet_loss_pct"],
                        "severity": severity,
                        "interface": device,
                        "description": f"{loss_pct}% packet loss (tc netem) on {device}",
                        "source": "tc_qdisc",
                    })

            # Extract delay (RTT proxy — flag if > 50ms)
            delay_match = re.search(r"delay (\d+(?:\.\d+)?)ms", line)
            if delay_match:
                delay_ms = float(delay_match.group(1))
                if delay_ms > 50.0:
                    anomalies.append({
                        "type": "high_rtt",
                        "value": delay_ms,
                        "threshold": 50.0,
                        "severity": "ALERT" if delay_ms < 200 else "CRITICAL",
                        "interface": device,
                        "description": f"{delay_ms}ms netem delay on {device} (>50ms threshold)",
                        "source": "tc_qdisc",
                    })

        return anomalies

    # ── Anomaly detection ──────────────────────────────────────────────────────

    def detect_anomalies(self) -> list[dict]:
        """
        Poll all telemetry queries.  Return a list of anomaly dicts:
          {type, value, threshold, severity, interface (optional), description}
        """
        anomalies: list[dict] = []

        # 1. Aggregate packet loss
        loss = self.query(TELEMETRY_QUERIES["packet_loss_pct"])
        if loss is not None and loss > THRESHOLDS["packet_loss_pct"]:
            severity = "CRITICAL" if loss > 10 else "ALERT"
            anomalies.append({
                "type": "packet_loss",
                "value": round(loss, 2),
                "threshold": THRESHOLDS["packet_loss_pct"],
                "severity": severity,
                "interface": "all",
                "description": f"{loss:.2f}% aggregate packet loss",
            })

        # 2. Per-interface packet loss (identify hotspot)
        iface_rows = self.query_vector(_IFACE_LOSS_QUERY)
        for row in iface_rows:
            v = row["value"]
            if v > THRESHOLDS["packet_loss_pct"]:
                iface = row["metric"].get("device", "unknown")
                severity = "CRITICAL" if v > 10 else "ALERT"
                anomalies.append({
                    "type": "packet_loss",
                    "value": round(v, 2),
                    "threshold": THRESHOLDS["packet_loss_pct"],
                    "severity": severity,
                    "interface": iface,
                    "description": f"{v:.2f}% packet loss on {iface}",
                })

        # 3. RX errors
        rx_err = self.query(TELEMETRY_QUERIES["rx_errors_per_sec"])
        if rx_err is not None and rx_err > THRESHOLDS["rx_errors_per_sec"]:
            anomalies.append({
                "type": "rx_errors",
                "value": round(rx_err, 4),
                "threshold": THRESHOLDS["rx_errors_per_sec"],
                "severity": "ALERT",
                "interface": "all",
                "description": f"{rx_err:.4f} RX errors/s",
            })

        # 4. TX errors
        tx_err = self.query(TELEMETRY_QUERIES["tx_errors_per_sec"])
        if tx_err is not None and tx_err > THRESHOLDS["tx_errors_per_sec"]:
            anomalies.append({
                "type": "tx_errors",
                "value": round(tx_err, 4),
                "threshold": THRESHOLDS["tx_errors_per_sec"],
                "severity": "ALERT",
                "interface": "all",
                "description": f"{tx_err:.4f} TX errors/s",
            })

        # 5. Pipeline duration p99
        dur_p99 = self.query(TELEMETRY_QUERIES["pipeline_duration_p99_s"])
        if dur_p99 is not None and dur_p99 > THRESHOLDS["pipeline_duration_p99_s"]:
            anomalies.append({
                "type": "pipeline_slow",
                "value": round(dur_p99, 1),
                "threshold": THRESHOLDS["pipeline_duration_p99_s"],
                "severity": "ALERT",
                "description": f"Pipeline p99 latency {dur_p99:.0f}s (>{THRESHOLDS['pipeline_duration_p99_s']:.0f}s)",
            })

        # 6. Pipeline approval rate (only if there are runs)
        approved = self.query(TELEMETRY_QUERIES["pipeline_runs_approved"]) or 0.0
        blocked  = self.query(TELEMETRY_QUERIES["pipeline_runs_blocked"])  or 0.0
        total = approved + blocked
        if total >= 5:  # enough data to be meaningful
            rate = approved / total
            if rate < THRESHOLDS["pipeline_approval_rate_low"]:
                anomalies.append({
                    "type": "low_approval_rate",
                    "value": round(rate, 3),
                    "threshold": THRESHOLDS["pipeline_approval_rate_low"],
                    "severity": "ALERT",
                    "description": (
                        f"Pipeline approval rate {rate*100:.1f}% "
                        f"({int(approved)}/{int(total)} runs approved)"
                    ),
                })

        # 7. Local tc qdisc check (supplements Prometheus — netem not visible in
        #    node_network_receive_drop_total on loopback)
        tc_anomalies = self._detect_tc_impairment()
        # Deduplicate: don't add tc anomaly if Prometheus already caught the same interface+type
        existing = {(a["type"], a.get("interface", "")) for a in anomalies}
        for a in tc_anomalies:
            key = (a["type"], a.get("interface", ""))
            if key not in existing:
                anomalies.append(a)
                existing.add(key)

        return anomalies

    # ── Task generation ────────────────────────────────────────────────────────

    def generate_task(self, anomaly: dict) -> str:
        """Convert a detected anomaly into a planner task string."""
        atype     = anomaly["type"]
        value     = anomaly["value"]
        severity  = anomaly.get("severity", "ALERT")
        iface     = anomaly.get("interface", "")

        if atype == "packet_loss":
            iface_str = f"on {iface} " if iface and iface != "all" else ""
            return (
                f"Network telemetry {severity}: {value}% packet loss {iface_str}"
                f"detected by Prometheus (threshold: {anomaly['threshold']}%). "
                f"SFC chain SPI=100 may be experiencing retransmissions. "
                f"Propose P4 changes: add DSCP EF marking (0x2E) for SFC NSH traffic "
                f"to prioritize it through congestion, and add drop counters per port "
                f"to track loss location."
            )

        if atype == "rx_errors":
            return (
                f"Network telemetry {severity}: {value:.4f} RX errors/s detected "
                f"by Prometheus (threshold: {anomaly['threshold']}/s). "
                f"Possible hardware issue or MTU mismatch on SFC path. "
                f"Propose P4 changes: add RX error counters per port, implement "
                f"NSH integrity check at parser to drop malformed packets early, "
                f"and add a per-port error register for monitoring."
            )

        if atype == "tx_errors":
            return (
                f"Network telemetry {severity}: {value:.4f} TX errors/s detected "
                f"by Prometheus (threshold: {anomaly['threshold']}/s). "
                f"SFC chain forwarding may be dropping packets at egress. "
                f"Propose P4 changes: add TX drop counters per SPI, implement "
                f"a fast failover action for SFC forwarding table entries, "
                f"and add queue depth registers to detect buffer exhaustion."
            )

        if atype == "high_rtt":
            iface_str = f"on {iface} " if iface and iface != "all" else ""
            return (
                f"Network telemetry {severity}: {value:.0f}ms RTT {iface_str}"
                f"detected (threshold: {anomaly['threshold']:.0f}ms). "
                f"High-latency path affecting SFC chain SPI=100. "
                f"Optimize P4 for high latency: add jumbo frame support (9000B MTU), "
                f"implement header compression DSCP hints for SFC NSH traffic, "
                f"and add a coalescing register for table updates to reduce control-plane overhead."
            )

        if atype == "pipeline_slow":
            return (
                f"Network telemetry {severity}: P4 pipeline p99 latency is {value:.0f}s "
                f"(threshold: {anomaly['threshold']:.0f}s). LLM agents are taking too long. "
                f"Review and simplify the SFC forwarding table structure: reduce table size, "
                f"collapse redundant SPI/SI entries into a single wildcard action, "
                f"and remove any unused match keys to reduce planner complexity."
            )

        if atype == "low_approval_rate":
            return (
                f"Network telemetry {severity}: pipeline approval rate is "
                f"{value*100:.1f}% (threshold: {anomaly['threshold']*100:.0f}%). "
                f"LLM-generated P4 code is frequently failing review. "
                f"Simplify the SFC baseline: use only essential table+action pairs, "
                f"ensure all actions have explicit parameters, and add a minimal "
                f"sfc_forward table with well-defined forward and drop actions."
            )

        # Generic fallback
        return (
            f"Network telemetry {severity}: anomaly detected — {anomaly.get('description', atype)}. "
            f"Value={value}, threshold={anomaly.get('threshold')}. "
            f"Inspect SFC chain (SPI=100) and propose appropriate P4 table changes "
            f"to address this condition."
        )

    # ── Run modes ─────────────────────────────────────────────────────────────

    def run_once(self) -> list[dict]:
        """
        Single poll: detect anomalies, return list of
        {anomaly, task} dicts. Does NOT trigger the pipeline.
        """
        anomalies = self.detect_anomalies()
        results = []
        for anomaly in anomalies:
            task = self.generate_task(anomaly)
            results.append({"anomaly": anomaly, "task": task})
        return results

    def run_loop(self, pipeline_fn, max_triggers: int = 5):
        """
        Poll every poll_interval_s seconds.  When anomalies are detected,
        generate a task and call pipeline_fn(task) to trigger the pipeline.

        Args:
            pipeline_fn: callable(task: str) -> dict — should run pipeline and
                         return the final state dict.
            max_triggers: stop after this many pipeline invocations (safety cap).
        """
        logger.info(
            f"[telemetry] Starting loop: interval={self.poll_interval_s}s "
            f"max_triggers={max_triggers} prometheus={self.prometheus_url}"
        )
        print(
            f"[telemetry] Loop started — polling every {self.poll_interval_s}s, "
            f"Prometheus at {self.prometheus_url}"
        )

        while self._trigger_count < max_triggers:
            poll_results = self.run_once()

            if not poll_results:
                print(f"[telemetry] Poll at {_now()} — no anomalies detected")
            else:
                for item in poll_results:
                    anomaly = item["anomaly"]
                    task    = item["task"]
                    print(
                        f"[telemetry] ANOMALY at {_now()}: "
                        f"{anomaly['severity']} {anomaly['type']} "
                        f"(value={anomaly['value']})"
                    )
                    print(f"[telemetry] Generated task: {task[:120]}...")
                    print(f"[telemetry] Triggering pipeline (trigger #{self._trigger_count + 1}/{max_triggers})")

                    try:
                        result = pipeline_fn(task)
                        approved = result.get("approved_for_push", False)
                        print(
                            f"[telemetry] Pipeline {'APPROVED' if approved else 'BLOCKED'} "
                            f"for anomaly: {anomaly['description']}"
                        )
                    except Exception as e:
                        print(f"[telemetry] Pipeline error: {e}")

                    self._trigger_count += 1
                    if self._trigger_count >= max_triggers:
                        print(f"[telemetry] Reached max_triggers={max_triggers}, stopping.")
                        return

                    # Only process first anomaly per poll to avoid flooding
                    break

            time.sleep(self.poll_interval_s)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%H:%M:%S UTC")


def run_pipeline(task: str) -> dict:
    """
    Convenience wrapper: run the p4harness pipeline with a task string,
    return the final state dict. Used by run_loop as the default pipeline_fn.
    """
    from agents import pipeline
    from pathlib import Path

    baseline_path = Path(__file__).resolve().parent.parent / "p4" / "sfc_baseline.p4"
    baseline_p4 = baseline_path.read_text() if baseline_path.exists() else ""

    state = {
        "task": task,
        "p4_code": baseline_p4,
        "plan": "", "review_notes": [],
        "test_output": "", "review_passed": False,
        "tests_passed": False, "approved_for_push": False,
        "round": 0,
    }
    final = {}
    for step in pipeline.stream(state):
        final.update(list(step.values())[0])
    return final

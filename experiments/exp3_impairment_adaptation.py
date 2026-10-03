"""
Experiment 3 — Impairment Adaptation
======================================
Tests whether the LLM can adapt P4 programs to degrading network conditions
described in telemetry alerts, without human intervention.

Scenarios (9 conditions of increasing severity and complexity):
  perfect      — baseline, no change
  lossy        — 5% loss, DSCP EF marking
  congested    — 100ms RTT + jitter, traffic classification
  flaky        — 15% loss + reorder + corruption, chain collapse + failover
  satellite    — 600ms RTT, jumbo frame + header compression hints
  recovery     — normal restored, rollback to baseline
  asymmetric   — directional loss (upstream clean, downstream 20% loss)
  burst        — bursty traffic with 50ms inter-burst idle, adaptive coalescing
  multi_impair — loss + jitter + corruption simultaneously, layered mitigation
"""
import subprocess, sys
from pathlib import Path; _R = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(_R)) if str(_R) not in sys.path else None
from experiments.base import BaseExperiment
from prometheus_client import Gauge, Counter

current_impairment  = Gauge("p4harness_exp3_impairment_level", "Current impairment scenario index")
adaptation_proposed = Counter("p4harness_exp3_adaptations_total", "P4 adaptations proposed", ["condition", "outcome"])

CLEAR_CMD = ["sudo", "tc", "qdisc", "del", "dev", "lo", "root"]

IMPAIRMENT_CMDS = {
    "perfect":      [],
    "lossy":        ["sudo", "tc", "qdisc", "add", "dev", "lo", "root", "netem", "loss", "5%", "delay", "10ms"],
    "congested":    ["sudo", "tc", "qdisc", "add", "dev", "lo", "root", "netem", "delay", "100ms", "jitter", "20ms", "loss", "2%"],
    "flaky":        ["sudo", "tc", "qdisc", "add", "dev", "lo", "root", "netem", "loss", "15%", "reorder", "5%", "corrupt", "1%"],
    "satellite":    ["sudo", "tc", "qdisc", "add", "dev", "lo", "root", "netem", "delay", "600ms", "jitter", "50ms", "loss", "1%"],
    "recovery":     [],
    "asymmetric":   ["sudo", "tc", "qdisc", "add", "dev", "lo", "root", "netem", "loss", "20%"],
    "burst":        ["sudo", "tc", "qdisc", "add", "dev", "lo", "root", "netem", "delay", "5ms", "loss", "0%"],
    "multi_impair": ["sudo", "tc", "qdisc", "add", "dev", "lo", "root", "netem", "loss", "8%", "delay", "50ms", "jitter", "15ms", "corrupt", "2%"],
}

class ImpairmentAdaptationExperiment(BaseExperiment):
    name = "exp3_impairment_adaptation"
    description = "LLM proposes P4 adaptations to degrading network conditions autonomously"

    def scenarios(self):
        return [
            {
                "scenario": "perfect", "condition_idx": 0,
                "task": "Network telemetry: 0ms RTT, 0% loss. All SFC chains nominal. Output the current sfc_forward table and forward/drop actions unchanged — confirming the baseline is optimal for normal conditions.",
                "extra": {"expected_action": "no_change"},
            },
            {
                "scenario": "lossy", "condition_idx": 1,
                "task": "Network telemetry ALERT: 5% packet loss, 10ms RTT. SFC chain SPI=100 experiencing retransmissions. Propose P4 changes: add DSCP EF marking (0x2E) for SFC NSH traffic to prioritize it, and add drop counters per port.",
                "extra": {"expected_action": "dscp_marking"},
            },
            {
                "scenario": "congested", "condition_idx": 2,
                "task": "Network telemetry CRITICAL: 100ms RTT, 20ms jitter, 2% loss. Buffer bloat detected. Implement P4 traffic classification: NSH/SFC control packets get DSCP CS7 (high priority), bulk traffic gets DSCP CS0. Add per-SPI counters.",
                "extra": {"expected_action": "traffic_classification"},
            },
            {
                "scenario": "flaky", "condition_idx": 3,
                "task": "Network telemetry SEVERE: 15% loss, 5% reorder, 1% corruption. Propose aggressive P4 changes: 1) Collapse SI=255+SI=254 into a single forwarding action 2) Add NSH integrity check at parser, drop corrupted packets 3) Implement fast failover between ports.",
                "extra": {"expected_action": "chain_collapse"},
            },
            {
                "scenario": "satellite", "condition_idx": 4,
                "task": "Network telemetry: 600ms RTT, 50ms jitter, 1% loss. High-latency satellite uplink. Optimize P4 for high latency: add jumbo frame support (9000B MTU), implement header compression DSCP hints, add coalescing register for table updates.",
                "extra": {"expected_action": "latency_optimization"},
            },
            {
                "scenario": "recovery", "condition_idx": 5,
                "task": "Network telemetry: returned to normal. 0ms RTT, 0% loss. Roll back impairment-specific P4 changes. Restore optimized baseline for normal LAN operation. Keep any universally beneficial improvements.",
                "extra": {"expected_action": "rollback_to_baseline"},
            },
            {
                "scenario": "asymmetric", "condition_idx": 6,
                "task": (
                    "Network telemetry ALERT: Asymmetric impairment detected. "
                    "Upstream (ingress port 0→1): 0ms RTT, 0% loss — clean. "
                    "Downstream (ingress port 1→0): 20% packet loss, likely a faulty NIC buffer. "
                    "Propose directional P4 changes: "
                    "1. Add a per-direction drop counter using two registers "
                    "   (reg_drop_upstream, reg_drop_downstream) indexed by a single slot. "
                    "2. For downstream traffic (ingress port=1), add DSCP CS6 marking "
                    "   to trigger upstream retransmission priority. "
                    "3. For upstream traffic (ingress port=0), no changes — keep clean path clean. "
                    "Use standard_metadata.ingress_port to distinguish directions."
                ),
                "extra": {"expected_action": "directional_mitigation", "lossy_direction": "downstream"},
            },
            {
                "scenario": "burst", "condition_idx": 7,
                "task": (
                    "Network telemetry: Bursty traffic pattern detected. "
                    "Traffic arrives in 10ms bursts followed by 50ms idle periods. "
                    "Peak burst rate exceeds 10 Gbps; average is only 1.2 Gbps. "
                    "Propose P4 changes to handle this pattern: "
                    "1. Add a 32-bit register 'burst_detector' tracking packets in the last window. "
                    "2. If burst_detector > 10000 (burst threshold), mark packets with DSCP AF21 "
                    "   to trigger weighted queuing downstream. "
                    "3. Implement a decay mechanism: apply a right-shift by 1 to burst_detector "
                    "   on each packet arrival to simulate exponential decay. "
                    "Keep all existing SFC tables unchanged."
                ),
                "extra": {"expected_action": "burst_detection_register", "threshold": 10000},
            },
            {
                "scenario": "multi_impair", "condition_idx": 8,
                "task": (
                    "Network telemetry CRITICAL: Multiple simultaneous impairments. "
                    "8% packet loss + 50ms RTT + 15ms jitter + 2% bit corruption detected. "
                    "This is likely a failing fiber segment. Implement layered mitigation: "
                    "1. Drop corrupted packets early: add an IPv4 checksum validity check "
                    "   at the parser; if invalid, set a metadata flag 'corrupt_pkt=1'. "
                    "2. In ingress control, if corrupt_pkt=1, drop immediately (before SFC tables). "
                    "3. For non-corrupt SFC traffic, add DSCP EF marking for all NSH packets "
                    "   to prioritize through the congested link. "
                    "4. Add a direct counter 'corruption_stats' on the drop action to track "
                    "   how many corrupt packets were caught per table entry. "
                    "Maintain all existing SFC forwarding logic for clean packets."
                ),
                "extra": {
                    "expected_action": "layered_mitigation",
                    "loss_pct": 8, "rtt_ms": 50, "jitter_ms": 15, "corrupt_pct": 2,
                },
            },
        ]

    def setup_scenario(self, scenario):
        cond = scenario["scenario"]
        subprocess.run(CLEAR_CMD, capture_output=True)
        cmd = IMPAIRMENT_CMDS.get(cond, [])
        if cmd:
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode == 0:
                print(f"  [netem] Applied impairment: {cond}")
        current_impairment.set(scenario.get("condition_idx", 0))

    def teardown_scenario(self, scenario):
        subprocess.run(CLEAR_CMD, capture_output=True)
        current_impairment.set(0)

    def run_all(self, pipeline=None, save: bool = True):
        records = super().run_all(pipeline=pipeline, save=save)
        for rec in records:
            outcome = "approved" if rec.approved else "blocked"
            adaptation_proposed.labels(condition=rec.scenario, outcome=outcome).inc()
        return records

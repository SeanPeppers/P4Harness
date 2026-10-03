"""
Experiment 1 — Autonomous Link Failure Recovery
================================================
Tests whether the LLM pipeline can detect a port failure reported in the task
and autonomously rewrite the SFC forwarding table to restore service — with no
human in the loop.

Scenarios (6 increasing complexity):
  S1: Single hop failure     — reroute one SI entry
  S2: Multi-hop failure      — reroute entire SPI path
  S3: Cascade failure        — two simultaneous port failures, find alternate path
  S4: Priority rerouting     — mixed critical/best-effort SPIs, preserve QoS ordering
  S5: Path restoration       — failed port recovers, decide restore vs. stay on backup
  S6: ECMP degradation       — 3-way ECMP loses one leg, rebalance to 2-way
"""
import sys
from pathlib import Path; _R = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(_R)) if str(_R) not in sys.path else None
from experiments.base import BaseExperiment


class LinkFailureExperiment(BaseExperiment):
    name = "exp1_link_failure"
    description = "Autonomous SFC rerouting under simulated port failures"

    def scenarios(self) -> list[dict]:
        return [
            {
                "scenario": "S1_single_hop_failure",
                "task": (
                    "ALERT: Port 1 on the SFC switch is DOWN (carrier lost). "
                    "The current sfc_forward table routes SPI=100, SI=255 to port 1. "
                    "Reroute SPI=100, SI=255 to port 2 as a backup path. "
                    "Ensure the NSH header is preserved and SI is decremented correctly. "
                    "Do not affect any other SPI/SI entries."
                ),
                "extra": {"failed_port": 1, "backup_port": 2, "spi": 100, "si": 255},
            },
            {
                "scenario": "S2_multi_hop_failure",
                "task": (
                    "ALERT: Port 0 is DOWN — this breaks the entire SPI=100 chain "
                    "(SI=255 ingress is on port 0). "
                    "Reroute the full SPI=100 chain (SI=255, SI=254, SI=253) "
                    "to use port 2 as the new ingress. Maintain correct SI decrement "
                    "at each hop. Update the sfc_forward table entries accordingly."
                ),
                "extra": {"failed_port": 0, "backup_port": 2, "spi": 100, "affected_sis": [255, 254, 253]},
            },
            {
                "scenario": "S3_cascade_failure",
                "task": (
                    "CRITICAL: Ports 0 AND 1 are both DOWN simultaneously. "
                    "This affects SPI=100 (all hops) and SPI=200 (SI=255). "
                    "Only port 2 and port 3 are available. "
                    "Redesign the sfc_forward table to maintain SFC continuity "
                    "for both SPI=100 and SPI=200 using only ports 2 and 3. "
                    "Add a default drop action for any SPI with no valid path."
                ),
                "extra": {"failed_ports": [0, 1], "available_ports": [2, 3], "affected_spis": [100, 200]},
            },
            {
                "scenario": "S4_priority_rerouting",
                "task": (
                    "ALERT: Port 1 is DOWN. Two SPIs are affected: "
                    "SPI=100 carries emergency/voice traffic (DSCP EF, must be restored first), "
                    "SPI=300 carries bulk backup traffic (DSCP CS0, best-effort). "
                    "Port 2 has limited bandwidth — it can carry SPI=100 at full rate or SPI=300 "
                    "at full rate, but not both without congestion. "
                    "Reroute SPI=100 (SI=255, SI=254) to port 2 unconditionally. "
                    "Reroute SPI=300 (SI=255) to port 2 with a meter that caps it at 500 Kbps "
                    "so voice traffic is protected. "
                    "Add DSCP-aware metering actions to the sfc_forward table."
                ),
                "extra": {
                    "failed_port": 1, "backup_port": 2,
                    "critical_spi": 100, "besteffort_spi": 300,
                    "cap_kbps": 500,
                },
            },
            {
                "scenario": "S5_path_restoration",
                "task": (
                    "UPDATE: Port 0, which failed 10 minutes ago, has now recovered (carrier up). "
                    "The current sfc_forward table was modified during the outage to route "
                    "SPI=100 (all SIs) through the backup port 2. "
                    "Evaluate whether to restore the original port 0 path or stay on port 2. "
                    "Decision criteria: if port 0 flapped more than 3 times in the last hour "
                    "(it did — 4 flaps logged), prefer stability and stay on port 2. "
                    "Add a register 'port_flap_count' (32-bit, indexed by port number) initialized "
                    "to track this. Keep sfc_forward on port 2 and increment the register for port 0 flap tracking."
                ),
                "extra": {
                    "recovered_port": 0, "current_backup": 2, "spi": 100,
                    "flap_count": 4, "flap_threshold": 3,
                },
            },
            {
                "scenario": "S6_ecmp_degradation",
                "task": (
                    "ALERT: SPI=100 is currently load-balanced 3-way across ports 0, 1, 2 "
                    "using a hash on (src_ip XOR dst_ip) mod 3. Port 1 has just gone DOWN. "
                    "Rewrite the ECMP logic to 2-way: hash mod 2 selecting between port 0 and port 2. "
                    "The action must correctly handle the mapping: "
                    "hash=0 → port 0, hash=1 → port 2. "
                    "Existing non-ECMP SPI entries (SPI=200, SPI=300) must be left unchanged. "
                    "Use a register or metadata field for the hash result."
                ),
                "extra": {
                    "failed_port": 1, "remaining_ports": [0, 2],
                    "spi": 100, "old_width": 3, "new_width": 2,
                },
            },
        ]

"""
Experiment 2 — Dynamic VNF Insertion
=====================================
Tests whether the LLM can progressively insert new Virtual Network Functions
into a live SFC chain without breaking existing traffic flows.

Each scenario CHAINS onto the previous — the approved P4 output from S1 is the
input to S2, and so on. This tests whether the models can maintain coherence
across sequential autonomous edits.

Scenarios (7 VNF insertions of increasing complexity):
  S1: Insert a simple firewall VNF  (new table + action)
  S2: Insert a load balancer VNF    (ECMP-style port selection)
  S3: Insert a rate limiter VNF     (meter + conditional drop)
  S4: Reorder chain + verify end-to-end SI sequence is still correct
  S5: Insert INT telemetry VNF      (in-band telemetry header insertion)
  S6: Insert NAT VNF                (src IP rewriting at SI boundary)
  S7: Remove a VNF cleanly          (excise load balancer, rechain remaining)
"""
import sys
from pathlib import Path; _R = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(_R)) if str(_R) not in sys.path else None
from experiments.base import BaseExperiment
from prometheus_client import Gauge

vnf_chain_depth = Gauge("p4harness_exp2_vnf_chain_depth", "Number of VNFs in the live chain")


class VNFInsertionExperiment(BaseExperiment):
    name = "exp2_vnf_insertion"
    description = "Progressive VNF insertion into a live SFC chain (chained edits)"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._current_p4 = self.baseline_p4
        self._chain_depth = 0

    def scenarios(self) -> list[dict]:
        return [
            {
                "scenario": "S1_firewall_vnf",
                "task": (
                    "Insert a stateless firewall VNF into SPI=100 at SI=200. "
                    "The firewall should: DROP packets from src_ip 10.0.0.0/8, "
                    "PASS all other IPv4 traffic. "
                    "Add this as a new P4 table 'firewall_acl' applied before sfc_forward. "
                    "Do not modify existing tables."
                ),
                "extra": {"vnf": "firewall", "si": 200, "spi": 100},
            },
            {
                "scenario": "S2_load_balancer_vnf",
                "task": (
                    "Insert a load balancer VNF at SI=190 in SPI=100 (after the firewall at SI=200). "
                    "Implement 2-way ECMP: hash on (src_ip XOR dst_ip) mod 2 to select port 0 or port 1. "
                    "Add a table 'lb_select' with a hash-based action. "
                    "Keep the firewall_acl table from the previous version intact."
                ),
                "extra": {"vnf": "load_balancer", "si": 190, "spi": 100, "backends": [0, 1]},
            },
            {
                "scenario": "S3_rate_limiter_vnf",
                "task": (
                    "Insert a token-bucket rate limiter VNF at SI=180 in SPI=100 (after LB at SI=190). "
                    "Use a register<bit<32>>(1) named 'token_bucket' to track remaining tokens. "
                    "In a new table 'rate_limit' with action 'check_rate': read register index 0, "
                    "if tokens > 0 decrement by 1 and forward; if tokens == 0 apply NoAction (drop intent signaled via metadata). "
                    "Do NOT use meter, DirectMeter, or MeterType — use only register read/write. "
                    "Keep all existing VNFs (firewall at SI=200, LB at SI=190) intact."
                ),
                "extra": {"vnf": "rate_limiter", "si": 180, "spi": 100, "rate_mbps": 1},
            },
            {
                "scenario": "S4_chain_integrity_check",
                "task": (
                    "The SFC chain for SPI=100 now has 3 VNFs: "
                    "firewall (SI=200) → load_balancer (SI=190) → rate_limiter (SI=180) → egress (SI=170). "
                    "Verify the complete P4 program is correct: "
                    "1. SI decrements correctly at each hop "
                    "2. All three VNF tables are applied in order in the ingress control block "
                    "3. No table dependency cycles "
                    "4. NSH header is valid throughout "
                    "Fix any issues found. Output the final complete P4 program."
                ),
                "extra": {"vnf": "integrity_check", "chain": ["firewall", "lb", "rate_limiter"]},
            },
            {
                "scenario": "S5_int_telemetry_vnf",
                "task": (
                    "Insert an In-Band Network Telemetry (INT) VNF at SI=175 in the SPI=100 chain "
                    "(after the rate limiter at SI=180, before egress at SI=170). "
                    "The INT VNF must: "
                    "1. Add a custom metadata header 'int_data_t' with fields: "
                    "   ingress_port (9-bit), queue_depth (19-bit), timestamp (32-bit). "
                    "2. Read ingress_port from standard_metadata.ingress_port, "
                    "   queue_depth from standard_metadata.deq_qdepth, "
                    "   timestamp from standard_metadata.ingress_global_timestamp. "
                    "3. Apply this via a new table 'int_collect' with a single 'insert_int' action. "
                    "4. Do not emit the int_data header to the wire — write values to "
                    "   registers indexed by ingress_port for collector polling. "
                    "Keep all previous VNFs (firewall at SI=200, LB at SI=190, rate limiter at SI=180) intact."
                ),
                "extra": {
                    "vnf": "int_telemetry", "si": 175, "spi": 100,
                    "fields": ["ingress_port", "queue_depth", "timestamp"],
                },
            },
            {
                "scenario": "S6_nat_vnf",
                "task": (
                    "Insert a source NAT VNF at SI=165 in SPI=100 "
                    "(after INT at SI=175, before egress at SI=170). "
                    "The NAT VNF rewrites the inner IPv4 src address for packets leaving the SFC domain: "
                    "Any packet with src_ip in 10.0.0.0/8 must be rewritten to 203.0.113.1 (public IP). "
                    "Implement as a new table 'snat_table' matching on ipv4.srcAddr with LPM. "
                    "The action 'do_snat' takes a parameter 'new_src_ip' and writes it to ipv4.srcAddr. "
                    "Recalculate ipv4 checksum after rewrite using update_checksum. "
                    "Keep all previous VNFs intact."
                ),
                "extra": {
                    "vnf": "nat", "si": 165, "spi": 100,
                    "src_prefix": "10.0.0.0/8", "public_ip": "203.0.113.1",
                },
            },
            {
                "scenario": "S7_vnf_removal",
                "task": (
                    "The load balancer VNF at SI=190 in SPI=100 is being decommissioned. "
                    "Remove it cleanly from the P4 program: "
                    "1. Delete the 'lb_select' table and its associated action. "
                    "2. Remove the lb_select.apply() call from the ingress control block. "
                    "3. Close the SI gap: the chain was "
                    "   firewall(SI=200) → lb(SI=190) → rate_limiter(SI=180). "
                    "   After removal it should be firewall(SI=200) → rate_limiter(SI=185) → ... "
                    "   Renumber rate_limiter to SI=185 in the sfc_forward table. "
                    "4. Keep firewall, rate_limiter, INT (SI=175), NAT (SI=165) tables intact. "
                    "Output a valid, compilable P4 program with no orphaned table references."
                ),
                "extra": {
                    "vnf": "removal", "removed_vnf": "load_balancer", "removed_si": 190,
                    "renumber": {"rate_limiter": 185},
                },
            },
        ]

    def run_all(self, pipeline=None, save: bool = True):
        """Override to chain approved P4 output between scenarios."""
        print(f"\n{'='*60}")
        print(f"  Experiment: {self.name}")
        print(f"  {self.description}")
        print(f"{'='*60}")

        current_p4 = self.baseline_p4

        for i, scenario in enumerate(self.scenarios(), 1):
            print(f"\n[{i}/{len(self.scenarios())}] {scenario['scenario']}")

            rec = self.run_one(scenario, p4_code=current_p4, pipeline=pipeline)
            self.records.append(rec)
            self._print_result(rec)

            if rec.approved and rec.p4_code:
                self._chain_depth += 1
                vnf_chain_depth.set(self._chain_depth)
                current_p4 = rec.p4_code
            else:
                print(f"  [chain] VNF insertion blocked — chain stays at depth {self._chain_depth}")

        if save:
            self._save_results()
        return self.records

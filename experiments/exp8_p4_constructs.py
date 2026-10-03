"""
Experiment 8 — Advanced P4 Constructs
=======================================
Tests the pipeline's ability to correctly generate P4 programs that use
constructs beyond basic table+action patterns. These are the scenarios most
likely to trip up LLMs that learned from limited P4 corpora.

Adds 5 scenarios covering:
  S1: Action selector / WCMP  — weighted cost multi-path using ActionProfile
  S2: Direct counters         — per-entry packet/byte accounting on sfc_forward
  S3: Multi-table pipeline    — 3 chained tables with metadata passing between them
  S4: Register array          — sliding-window congestion detection using registers
  S5: Header stack / tunneling — NSH-over-VXLAN, parser must handle outer+inner headers
"""
import sys
from pathlib import Path; _R = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(_R)) if str(_R) not in sys.path else None
from experiments.base import BaseExperiment


class P4ConstructsExperiment(BaseExperiment):
    name = "exp8_p4_constructs"
    description = "Advanced P4 constructs: action selectors, counters, registers, header stacks"
    # S1-S5 are independent constructs, not a narrative — each must start from the
    # pristine baseline. Chaining would let S1's accumulated edits decide S5.
    chain_p4 = False

    def scenarios(self) -> list[dict]:
        return [
            {
                "scenario": "S1_action_selector_wcmp",
                "task": (
                    "Replace the simple sfc_forward table with a Weighted-Cost Multi-Path (WCMP) "
                    "implementation using a P4 ActionProfile and ActionSelector. "
                    "Requirements: "
                    "1. Define an ActionProfile 'sfc_action_profile' with size=256. "
                    "2. Define an ActionSelector 'sfc_selector' backed by sfc_action_profile, "
                    "   using a CRC32 hash over (nsh.spi, nsh.si, ipv4.srcAddr, ipv4.dstAddr). "
                    "3. Redefine sfc_forward to use 'implementation = sfc_selector'. "
                    "4. Define two actions: forward_port0 (egress_port=0, weight=2) and "
                    "   forward_port2 (egress_port=2, weight=1) — 2:1 WCMP ratio for SPI=100. "
                    "5. The table should still match on nsh.spi and nsh.si (exact match). "
                    "Output the complete P4 program including the header and metadata definitions."
                ),
                "extra": {
                    "construct": "action_selector",
                    "spi": 100, "ports": [0, 2], "weights": [2, 1],
                    "hash_fields": ["nsh.spi", "nsh.si", "ipv4.srcAddr", "ipv4.dstAddr"],
                },
            },
            {
                "scenario": "S2_direct_counters",
                "task": (
                    "Add per-entry packet and byte accounting to the sfc_forward table "
                    "using P4 DirectCounter primitives. "
                    "Requirements: "
                    "1. Define a DirectCounter 'sfc_pkt_bytes' of type PACKETS_AND_BYTES "
                    "   attached to sfc_forward (use @name annotation). "
                    "2. In each action that handles a matched packet (forward, decrement_si), "
                    "   add a count() call on sfc_pkt_bytes. "
                    "3. Add a second DirectCounter 'sfc_drop_counter' of type PACKETS "
                    "   attached to a new 'sfc_drop_table' that fires for unmatched SPI/SI pairs "
                    "   (table miss → drop + count). "
                    "4. Expose both counters via standard P4Runtime read operations "
                    "   (no additional code needed — just ensure proper @id annotations). "
                    "Output the complete updated P4 program."
                ),
                "extra": {
                    "construct": "direct_counter",
                    "counters": ["sfc_pkt_bytes", "sfc_drop_counter"],
                    "types": ["PACKETS_AND_BYTES", "PACKETS"],
                },
            },
            {
                "scenario": "S3_multi_table_pipeline",
                "task": (
                    "Design a 3-stage P4 pipeline for SFC processing, where each stage "
                    "communicates with the next via a custom metadata struct. "
                    "Stage 1 — classify_traffic: Match on ipv4.diffserv (DSCP). "
                    "  - DSCP EF (0x2E) → set metadata.traffic_class = 1 (priority) "
                    "  - DSCP CS0 (0x00) → set metadata.traffic_class = 0 (best-effort) "
                    "  - default         → set metadata.traffic_class = 0 "
                    "Stage 2 — sfc_forward: Match on (nsh.spi, nsh.si). "
                    "  - Use metadata.traffic_class to choose egress port: "
                    "    class=1 → always egress port 0 (priority path), "
                    "    class=0 → egress port 1 (normal path). "
                    "  - Decrement nsh.si by 1. "
                    "Stage 3 — egress_meter: Match on metadata.traffic_class. "
                    "  - class=0 traffic: apply a meter 'be_meter', drop if RED. "
                    "  - class=1 traffic: bypass meter, always forward. "
                    "Define metadata struct 'sfc_meta_t' with field traffic_class (2-bit). "
                    "Apply all 3 tables in order in the ingress control block. "
                    "Output the complete P4 program."
                ),
                "extra": {
                    "construct": "multi_table_metadata",
                    "stages": ["classify_traffic", "sfc_forward", "egress_meter"],
                    "metadata_fields": ["traffic_class"],
                },
            },
            {
                "scenario": "S4_register_array_congestion",
                "task": (
                    "Implement a sliding-window packet-rate estimator using P4 register arrays "
                    "to detect per-SPI congestion in the data plane. "
                    "Requirements: "
                    "1. Define a register array 'pkt_count' of type bit<32> with size=256 "
                    "   (one slot per SPI, indexed by nsh.spi). "
                    "2. In a new table 'congestion_detect', apply an action 'update_and_check': "
                    "   a. Read pkt_count[nsh.spi] into a local variable. "
                    "   b. Increment it by 1 and write back. "
                    "   c. If the count exceeds 10000, set metadata.congested = 1. "
                    "3. Define a second register 'decay_timer' of type bit<48> with size=256. "
                    "   Read standard_metadata.ingress_global_timestamp. "
                    "   If (timestamp - decay_timer[nsh.spi]) > 1000000 (1 second in microseconds), "
                    "   reset pkt_count[nsh.spi] = 0 and update decay_timer[nsh.spi] = timestamp. "
                    "4. In sfc_forward, if metadata.congested = 1, mark packet DSCP AF21 "
                    "   (congestion signaling). "
                    "Apply congestion_detect before sfc_forward in the control block."
                ),
                "extra": {
                    "construct": "register_array",
                    "registers": ["pkt_count", "decay_timer"],
                    "threshold": 10000, "decay_us": 1000000,
                },
            },
            {
                "scenario": "S5_header_stack_vxlan_nsh",
                "task": (
                    "Extend the P4 parser to handle NSH-over-VXLAN encapsulation, "
                    "where the outer UDP/VXLAN header must be parsed before the NSH header. "
                    "Requirements: "
                    "1. Define header types for: ethernet_t, ipv4_t (already exists), "
                    "   udp_t (srcPort 16-bit, dstPort 16-bit, length 16-bit, checksum 16-bit), "
                    "   vxlan_t (flags 8-bit, reserved 24-bit, vni 24-bit, reserved2 8-bit), "
                    "   nsh_t (version 2-bit, oam 1-bit, context 1-bit, ttl 6-bit, length 6-bit, "
                    "          reserved 4-bit, next_proto 8-bit, spi 24-bit, si 8-bit). "
                    "2. Update the parser: ethernet → ipv4 → udp (if proto=17) → "
                    "   vxlan (if dstPort=4789) → inner ethernet → inner ipv4 → nsh "
                    "   (if inner etherType=0x894F). "
                    "3. In ingress, strip the outer VXLAN encap: "
                    "   set vxlan.setInvalid(), outer_ethernet.setInvalid(), outer_ipv4.setInvalid(), "
                    "   outer_udp.setInvalid() before applying sfc_forward. "
                    "4. The sfc_forward table matches on nsh.spi and nsh.si as before. "
                    "5. In deparser, emit only the inner headers (inner_ethernet, inner_ipv4, nsh). "
                    "Output the complete P4 program with parser, ingress, egress, and deparser."
                ),
                "extra": {
                    "construct": "header_stack",
                    "encap": "VXLAN-over-UDP",
                    "inner_proto": "NSH",
                    "vxlan_port": 4789,
                    "nsh_ethertype": "0x894F",
                },
            },
        ]

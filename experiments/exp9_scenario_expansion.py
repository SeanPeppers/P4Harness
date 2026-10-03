"""
Experiment 9 — Scenario Expansion (Breadth Coverage)
=====================================================
Adds 15 functional scenarios covering P4/SDN capability areas that exp1
(link failure), exp2 (VNF insertion), exp3 (impairment adaptation) and exp8
(advanced P4 constructs) never exercise. Together with those 27 scenarios this
brings the functional suite to n=42, addressing the "statistically thin"
review comment.

Every scenario is grounded in the paper's application domain — a video-streaming
service function chain carrying RTP/H.264 over NSH.

Scenarios (15 distinct capability areas):
  S1:  Multicast / packet replication engine  — mcast_grp fan-out of a video stream
  S2:  Clone-to-CPU (CloneType.I2E)           — punt chain-end packets to the controller
  S3:  P4Runtime digest()                     — new-flow learning notification
  S4:  IPv6 dual-stack parsing                — NSH next_proto 0x1/0x2 demux
  S5:  TTL / NSH hop-limit expiry             — underflow-safe decrement + drop counters
  S6:  Parser error handling                  — verify(), custom error, reject, parser_error
  S7:  Recirculate multi-pass processing      — two-pass re-chaining with a loop guard
  S8:  MPLS label push / swap / pop           — transport-label ops over a header stack
  S9:  802.1Q VLAN tag push / pop             — per-SPI VLAN + 802.1p priority
  S10: Meter with 3-color marking             — DirectMeter trTCM GREEN/YELLOW/RED
  S11: Match-kind variety                     — lpm + ternary + range + exact in one key
  S12: ARP responder / L2 address resolution  — data-plane ARP reply generation
  S13: Checksum verify + UDP recompute        — verify_checksum + update_checksum_with_payload
  S14: Non-byte-aligned bit fields            — RTP header parsing and loss detection
  S15: Idle timeout / entry aging             — support_timeout flow cache in front of sfc_forward

Field names follow p4/sfc_baseline.p4 exactly (snake_case: src_addr, dst_addr,
ether_type, hdr_checksum, diffserv).
"""
import sys
from pathlib import Path; _R = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(_R)) if str(_R) not in sys.path else None
from experiments.base import BaseExperiment


class ScenarioExpansionExperiment(BaseExperiment):
    name = "exp9_scenario_expansion"
    description = "Breadth expansion: multicast, cloning, digests, IPv6, MPLS, VLAN, meters, checksums, aging"
    # S1-S15 are independent capability probes, not a narrative — each must start
    # from the pristine baseline. Chained, S15 would inherit multicast + clone +
    # digest + IPv6 + MPLS + VLAN + meters stacked into one program and fail for
    # reasons unrelated to idle timeouts.
    chain_p4 = False

    def scenarios(self) -> list[dict]:
        return [
            {
                "scenario": "S1_multicast_video_replication",
                "task": (
                    "The SPI=100 video chain must fan a single 4K stream out to three "
                    "downstream transcoder VNFs simultaneously. Implement this with the "
                    "v1model Packet Replication Engine instead of unicast forwarding. "
                    "Requirements: "
                    "1. Add a table 'video_multicast' with key { meta.spi: exact; } and a "
                    "   single action 'set_mcast_group(bit<16> gid)' that writes "
                    "   std_meta.mcast_grp = gid. size = 256, default_action = drop(). "
                    "2. Entry: meta.spi = 100 -> set_mcast_group(1). Multicast group 1 is "
                    "   provisioned by the control plane with replicas on ports 1, 2 and 3, "
                    "   carrying egress_rid values 1, 2 and 3 respectively. "
                    "3. When mcast_grp is set, the program must NOT also write "
                    "   std_meta.egress_spec — setting both is a v1model error. Apply "
                    "   video_multicast only for packets whose NSH header is valid; all "
                    "   non-multicast SPIs continue to use sfc_forward unchanged. "
                    "4. In the egress control, use std_meta.egress_rid to rewrite the "
                    "   destination MAC per replica: rid=1 -> 00:00:00:00:01:01, "
                    "   rid=2 -> 00:00:00:00:01:02, rid=3 -> 00:00:00:00:01:03. "
                    "5. Split-horizon: in egress, if std_meta.egress_port == "
                    "   std_meta.ingress_port, call mark_to_drop(std_meta) so the source "
                    "   port never receives its own replica. "
                    "6. Decrement nsh.si exactly once, in ingress, before replication — not "
                    "   per replica. "
                    "Output the complete P4 program including parser, ingress, egress and deparser."
                ),
                "extra": {
                    "construct": "multicast_replication",
                    "table": "video_multicast",
                    "spi": 100, "mcast_grp": 1,
                    "replica_ports": [1, 2, 3], "egress_rids": [1, 2, 3],
                    "split_horizon": True,
                },
            },
            {
                "scenario": "S2_clone_to_cpu_chain_end",
                "task": (
                    "When a video packet reaches the end of its service chain (nsh.si == 0) "
                    "or misses sfc_forward, the controller must receive a copy for "
                    "diagnosis while the original packet is still handled normally. "
                    "Implement ingress-to-egress cloning to a CPU port. "
                    "Requirements: "
                    "1. Define '#define CPU_PORT 255' and a header "
                    "   cpu_header_t { bit<24> spi; bit<8> si; bit<9> ingress_port; "
                    "   bit<7> reason_code; } (48 bits total, byte-aligned). Add it to "
                    "   headers_t as 'cpu'. "
                    "2. Add metadata fields meta.clone_reason (bit<7>) and "
                    "   meta.orig_ingress_port (bit<9>), both annotated with "
                    "   @field_list(1) so they survive the clone. "
                    "3. In ingress: if hdr.nsh.si == 0 set meta.clone_reason = 1; "
                    "   if sfc_forward.apply().hit == false set meta.clone_reason = 2. "
                    "   In either case set meta.orig_ingress_port = std_meta.ingress_port "
                    "   and call clone_preserving_field_list(CloneType.I2E, 100, 1). "
                    "   Clone session 100 is programmed by the control plane to egress "
                    "   on CPU_PORT. "
                    "4. In egress: if std_meta.instance_type == 1 "
                    "   (PKT_INSTANCE_TYPE_INGRESS_CLONE), call hdr.cpu.setValid(), fill "
                    "   spi/si from the NSH header, ingress_port and reason_code from the "
                    "   preserved metadata, and set hdr.ethernet.ether_type = 0x9000. "
                    "   For all other instance types leave hdr.cpu invalid. "
                    "5. In the deparser emit hdr.ethernet, then hdr.cpu, then hdr.nsh, "
                    "   then hdr.ipv4 — the CPU header must sit immediately after Ethernet. "
                    "6. The original (non-clone) packet must be forwarded or dropped exactly "
                    "   as it was before this change; cloning must not alter its path. "
                    "Output the complete P4 program."
                ),
                "extra": {
                    "construct": "clone_to_cpu",
                    "clone_type": "CloneType.I2E",
                    "session_id": 100, "cpu_port": 255,
                    "field_list_id": 1,
                    "reason_codes": {"chain_end": 1, "table_miss": 2},
                    "cpu_ether_type": "0x9000",
                },
            },
            {
                "scenario": "S3_p4runtime_digest_flow_learn",
                "task": (
                    "The controller needs to learn about new video flows entering the SFC "
                    "domain without a full packet punt. Implement P4Runtime digest "
                    "notification with data-plane deduplication. "
                    "Requirements: "
                    "1. Define 'struct flow_learn_t { bit<48> src_mac; bit<32> src_addr; "
                    "   bit<24> spi; bit<9> ingress_port; }'. "
                    "2. Define a register 'seen_flows' of type bit<1> with size 4096. "
                    "3. Compute a 12-bit index with the hash() extern: "
                    "   hash(meta.flow_idx, HashAlgorithm.crc16, (bit<16>)0, "
                    "   { hdr.ethernet.src_addr, hdr.ipv4.src_addr, meta.spi }, (bit<32>)4096); "
                    "   meta.flow_idx is bit<16>. "
                    "4. Read seen_flows[meta.flow_idx]. If the bit is 0: write 1 back and "
                    "   call digest<flow_learn_t>(1, { hdr.ethernet.src_addr, "
                    "   hdr.ipv4.src_addr, meta.spi, std_meta.ingress_port }). "
                    "   If the bit is already 1, emit no digest — this bounds digest rate "
                    "   to one message per flow per index. "
                    "5. Perform the learn step in the ingress control BEFORE sfc_forward, "
                    "   and only for packets where hdr.ipv4.isValid() && hdr.nsh.isValid(). "
                    "6. Forwarding behaviour must be unchanged: learning must never drop, "
                    "   reorder or re-route a packet. "
                    "Note that entry aging of seen_flows is a control-plane concern and "
                    "requires no P4 code. Output the complete P4 program."
                ),
                "extra": {
                    "construct": "digest",
                    "digest_struct": "flow_learn_t",
                    "receiver_id": 1,
                    "dedup_register": "seen_flows",
                    "register_size": 4096,
                    "hash_algorithm": "crc16",
                    "digest_fields": ["src_mac", "src_addr", "spi", "ingress_port"],
                },
            },
            {
                "scenario": "S4_ipv6_dual_stack_parsing",
                "task": (
                    "Video subscribers are migrating to IPv6. Extend the program to a "
                    "dual-stack SFC data plane that parses and forwards both IPv4 and IPv6 "
                    "payloads under the same NSH chain. "
                    "Requirements: "
                    "1. Define header ipv6_t { bit<4> version; bit<8> traffic_class; "
                    "   bit<20> flow_label; bit<16> payload_len; bit<8> next_hdr; "
                    "   bit<8> hop_limit; bit<128> src_addr; bit<128> dst_addr; } and add "
                    "   it to headers_t as 'ipv6'. "
                    "2. Parser: from the start state, ether_type 0x0800 -> parse_ipv4, "
                    "   0x86DD -> parse_ipv6, 0x894F -> parse_nsh. From parse_nsh, branch "
                    "   on hdr.nsh.next_proto per RFC 8300: 0x01 -> parse_ipv4, "
                    "   0x02 -> parse_ipv6, default -> accept. "
                    "3. Add metadata field meta.is_ipv6 (bit<1>), set to 1 in parse_ipv6 "
                    "   and 0 in parse_ipv4. "
                    "4. sfc_forward keeps its exact key on (meta.spi, meta.si) and is "
                    "   family-agnostic. Add a second table 'dscp_mark' with key "
                    "   { meta.is_ipv6: exact; } and two actions: mark_v4() writing "
                    "   hdr.ipv4.diffserv = 0xB8 (DSCP EF) guarded by hdr.ipv4.isValid(), "
                    "   and mark_v6() writing hdr.ipv6.traffic_class = 0xB8 guarded by "
                    "   hdr.ipv6.isValid(). "
                    "5. The IPv4 checksum update in MyComputeChecksum must stay conditioned "
                    "   on hdr.ipv4.isValid(); IPv6 has no header checksum and must not be "
                    "   given one. "
                    "6. Deparser emits ethernet, nsh, ipv4, ipv6 in that order — invalid "
                    "   headers are skipped automatically, so exactly one L3 header is emitted. "
                    "Output the complete P4 program."
                ),
                "extra": {
                    "construct": "ipv6_dual_stack",
                    "ether_types": {"ipv4": "0x0800", "ipv6": "0x86DD", "nsh": "0x894F"},
                    "nsh_next_proto": {"ipv4": 1, "ipv6": 2},
                    "metadata_fields": ["is_ipv6"],
                    "dscp_ef": "0xB8",
                },
            },
            {
                "scenario": "S5_ttl_hoplimit_expiry",
                "task": (
                    "A forwarding loop in the video chain went undetected because neither "
                    "the IPv4 TTL nor the NSH TTL is being decremented. Add correct, "
                    "underflow-safe hop-limit handling. "
                    "Requirements: "
                    "1. Add two counters: counter(1, CounterType.packets) 'ipv4_ttl_expired' "
                    "   and counter(1, CounterType.packets) 'nsh_ttl_expired'. "
                    "2. In ingress, BEFORE any decrement, test expiry: "
                    "   if (hdr.ipv4.isValid() && hdr.ipv4.ttl <= 1) -> "
                    "   ipv4_ttl_expired.count(0); mark_to_drop(std_meta); exit. "
                    "   if (hdr.nsh.isValid() && hdr.nsh.ttl == 0) -> "
                    "   nsh_ttl_expired.count(0); mark_to_drop(std_meta); exit. "
                    "   Testing before decrementing is required: hdr.nsh.ttl is a 6-bit "
                    "   field and 0 - 1 wraps to 63, silently creating an immortal packet. "
                    "3. Only after both checks pass: hdr.ipv4.ttl = hdr.ipv4.ttl - 1 and "
                    "   hdr.nsh.ttl = hdr.nsh.ttl - 1. NSH TTL is decremented once per SFF "
                    "   hop, independently of the SI decrement. "
                    "4. Keep the existing nsh.si decrement and sfc_forward.apply() ordering "
                    "   intact, after the TTL logic. "
                    "5. The IPv4 header checksum must be recomputed after the TTL change — "
                    "   verify MyComputeChecksum lists hdr.ipv4.ttl in its field list. "
                    "6. Do not generate ICMP Time Exceeded; expiry is a silent counted drop. "
                    "Output the complete P4 program."
                ),
                "extra": {
                    "construct": "ttl_expiry",
                    "fields": ["ipv4.ttl", "nsh.ttl"],
                    "nsh_ttl_width": 6,
                    "counters": ["ipv4_ttl_expired", "nsh_ttl_expired"],
                    "check_before_decrement": True,
                },
            },
            {
                "scenario": "S6_parser_error_reject_handling",
                "task": (
                    "Malformed NSH packets from an upstream peer are being forwarded "
                    "instead of dropped. Add explicit parser validation, custom error "
                    "types and an ingress error handler. "
                    "Requirements: "
                    "1. Declare 'error { NshVersionUnsupported, NshLengthInvalid, "
                    "   NshProtoUnsupported }' at file scope. "
                    "2. In parse_nsh, immediately after pkt.extract(hdr.nsh): "
                    "   verify(hdr.nsh.ver == 0, error.NshVersionUnsupported); "
                    "   verify(hdr.nsh.len >= 2 && hdr.nsh.len <= 63, error.NshLengthInvalid); "
                    "   A failed verify() aborts parsing and transitions to reject. "
                    "3. Replace the default branch of the next_proto select with an explicit "
                    "   'default: reject;' so unknown payload protocols are rejected rather "
                    "   than silently accepted. "
                    "4. Add a table 'parser_error_handler' with key "
                    "   { std_meta.parser_error: exact; } and actions "
                    "   { drop_and_count; NoAction; }, size = 16, "
                    "   default_action = drop_and_count(). Attach a "
                    "   DirectCounter(CounterType.packets) 'parse_err_stats' to it so each "
                    "   error type is counted separately. "
                    "5. In the ingress apply block, the FIRST statement must be: "
                    "   if (std_meta.parser_error != error.NoError) { "
                    "   parser_error_handler.apply(); exit; } — no SFC table may be reached "
                    "   by a malformed packet. "
                    "6. Note that error.PacketTooShort and error.HeaderTooShort are raised "
                    "   automatically by extract(); the handler table must cover them too "
                    "   via its default action. "
                    "Output the complete P4 program."
                ),
                "extra": {
                    "construct": "parser_error",
                    "errors": ["NshVersionUnsupported", "NshLengthInvalid", "NshProtoUnsupported"],
                    "table": "parser_error_handler",
                    "counter": "parse_err_stats",
                    "checks_first_in_ingress": True,
                },
            },
            {
                "scenario": "S7_recirculate_two_pass_rechain",
                "task": (
                    "A video packet finishing chain SPI=100 must be immediately re-injected "
                    "into a second chain SPI=200 (transcode-then-archive). A single ingress "
                    "pass cannot re-run sfc_forward, so use recirculation for a second pass. "
                    "Requirements: "
                    "1. Add metadata fields meta.pass_number (bit<2>) and meta.next_spi "
                    "   (bit<24>), both annotated @field_list(2) so they survive recirculation. "
                    "2. Pass 1 (meta.pass_number == 0): run the existing sfc_forward logic. "
                    "   If hdr.nsh.spi == 100 && hdr.nsh.si == 0 (chain complete), set "
                    "   meta.next_spi = 200, set meta.pass_number = 1, and mark the packet "
                    "   for recirculation. "
                    "3. Recirculate from the EGRESS control with "
                    "   recirculate_preserving_field_list(2) — v1model does not allow "
                    "   recirculation from ingress. "
                    "4. Pass 2 (meta.pass_number == 1): rewrite the NSH header for the new "
                    "   chain — hdr.nsh.spi = meta.next_spi, hdr.nsh.si = 255, "
                    "   hdr.nsh.ttl = 63 — then apply sfc_forward normally. "
                    "5. LOOP GUARD (mandatory): if meta.pass_number >= 2 on entry to ingress, "
                    "   mark_to_drop(std_meta) and exit. Increment a "
                    "   counter(1, CounterType.packets) 'recirc_loop_guard' when this fires. "
                    "   The packet must be recirculated at most once. "
                    "6. Packets that are not at a chain boundary must never be recirculated. "
                    "Output the complete P4 program including the egress control."
                ),
                "extra": {
                    "construct": "recirculate",
                    "field_list_id": 2,
                    "metadata_fields": ["pass_number", "next_spi"],
                    "src_spi": 100, "dst_spi": 200,
                    "max_passes": 2,
                    "loop_guard_counter": "recirc_loop_guard",
                },
            },
            {
                "scenario": "S8_mpls_label_push_swap_pop",
                "task": (
                    "The SFC domain is being carried over an MPLS transport core. Implement "
                    "MPLS label push, swap and pop on top of the existing NSH forwarding. "
                    "Requirements: "
                    "1. Define header mpls_t { bit<20> label; bit<3> tc; bit<1> bos; "
                    "   bit<8> ttl; } and declare a header stack 'mpls_t[3] mpls' in headers_t. "
                    "2. Parser: ether_type 0x8847 -> parse_mpls. parse_mpls extracts "
                    "   hdr.mpls.next and selects on the just-extracted bos bit: "
                    "   0 -> parse_mpls (another label follows), 1 -> parse_mpls_payload. "
                    "   parse_mpls_payload peeks the first nibble of the payload: 4 -> "
                    "   parse_ipv4, otherwise parse_nsh. "
                    "3. Define three actions: "
                    "   mpls_push(bit<20> label, bit<3> tc) — hdr.mpls[0].setValid(), set "
                    "   label/tc, bos = 1 if no label was previously valid else 0, "
                    "   ttl = hdr.ipv4.ttl, and set hdr.ethernet.ether_type = 0x8847. "
                    "   mpls_swap(bit<20> label) — overwrite hdr.mpls[0].label and "
                    "   decrement hdr.mpls[0].ttl by 1. "
                    "   mpls_pop() — hdr.mpls[0].setInvalid() and restore "
                    "   hdr.ethernet.ether_type to 0x894F when hdr.nsh.isValid(), else 0x0800. "
                    "4. Add a table 'mpls_ops' with key { hdr.mpls[0].label: exact; "
                    "   meta.spi: exact; } actions { mpls_push; mpls_swap; mpls_pop; "
                    "   NoAction; }, size = 1024, default_action = NoAction(). "
                    "   Entry semantics: SPI=100 entering the core -> mpls_push(1001, 5); "
                    "   label 1001 in transit -> mpls_swap(1002); label 1002 at the egress "
                    "   PE -> mpls_pop(). "
                    "5. Apply mpls_ops before sfc_forward; sfc_forward's key is unchanged. "
                    "6. Deparser must emit hdr.ethernet, then hdr.mpls (the whole stack), "
                    "   then hdr.nsh, then hdr.ipv4. "
                    "Output the complete P4 program."
                ),
                "extra": {
                    "construct": "mpls",
                    "ether_type": "0x8847",
                    "stack_depth": 3,
                    "actions": ["mpls_push", "mpls_swap", "mpls_pop"],
                    "labels": {"push": 1001, "swap": 1002},
                    "tc": 5,
                },
            },
            {
                "scenario": "S9_vlan_tag_push_pop",
                "task": (
                    "The SFF is being attached to an 802.1Q trunk. Video and bulk chains "
                    "must ride separate VLANs with distinct 802.1p priorities, and must be "
                    "untagged again on the access port. "
                    "Requirements: "
                    "1. Define header vlan_t { bit<3> pcp; bit<1> dei; bit<12> vid; "
                    "   bit<16> ether_type; } and add it to headers_t as 'vlan'. "
                    "2. Parser: from start, ether_type 0x8100 -> parse_vlan. parse_vlan "
                    "   extracts hdr.vlan and then selects on hdr.vlan.ether_type: "
                    "   0x894F -> parse_nsh, 0x0800 -> parse_ipv4, default -> accept. "
                    "3. Action push_vlan(bit<12> vid, bit<3> pcp): hdr.vlan.setValid(); "
                    "   set vid, pcp, dei = 0; hdr.vlan.ether_type = "
                    "   hdr.ethernet.ether_type; hdr.ethernet.ether_type = 0x8100. "
                    "   The inner ether_type must be moved, not duplicated. "
                    "4. Action pop_vlan(): hdr.ethernet.ether_type = hdr.vlan.ether_type; "
                    "   hdr.vlan.setInvalid(). "
                    "5. Add a table 'vlan_ops' with key { meta.spi: exact; "
                    "   std_meta.ingress_port: exact; } actions { push_vlan; pop_vlan; "
                    "   NoAction; }, size = 512, default_action = NoAction(). Entries: "
                    "   (spi=100, any trunk ingress) -> push_vlan(100, 5)   [video, 802.1p CoS 5]; "
                    "   (spi=300, any trunk ingress) -> push_vlan(300, 1)   [bulk, CoS 1]; "
                    "   (spi=100, ingress_port=3)    -> pop_vlan()          [untagged access port]. "
                    "6. Apply vlan_ops after sfc_forward so the egress port is already "
                    "   resolved. Deparser emits ethernet, vlan, nsh, ipv4 in that order. "
                    "Output the complete P4 program."
                ),
                "extra": {
                    "construct": "vlan_8021q",
                    "tpid": "0x8100",
                    "actions": ["push_vlan", "pop_vlan"],
                    "vlans": {"video_spi_100": {"vid": 100, "pcp": 5},
                              "bulk_spi_300": {"vid": 300, "pcp": 1}},
                    "access_port": 3,
                },
            },
            {
                "scenario": "S10_meter_three_color_marking",
                "task": (
                    "Cap the SPI=100 4K video chain with a two-rate three-color marker so "
                    "that conforming traffic passes untouched, excess traffic is demoted "
                    "rather than dropped, and violating traffic is dropped. Use the P4 "
                    "meter primitive directly — not a register-based token bucket. "
                    "Requirements: "
                    "1. Declare 'direct_meter<bit<2>>(MeterType.bytes) video_rate_meter;'. "
                    "2. Add metadata field meta.meter_color (bit<2>). "
                    "3. Add a table 'video_meter' with key { meta.spi: exact; }, actions "
                    "   { mark_color; NoAction; }, meters = video_rate_meter, size = 256, "
                    "   default_action = NoAction(). Action mark_color() calls "
                    "   video_rate_meter.read(meta.meter_color). "
                    "4. Interpret the color per RFC 2698 trTCM: 0 = GREEN, 1 = YELLOW, "
                    "   2 = RED. After video_meter.apply(): "
                    "   GREEN  -> leave hdr.ipv4.diffserv unchanged; "
                    "   YELLOW -> hdr.ipv4.diffserv = 0x30 (DSCP AF12, drop-eligible); "
                    "   RED    -> mark_to_drop(std_meta) and count into a "
                    "             counter(1, CounterType.packets) 'meter_red_drops', then exit. "
                    "5. Meter parameters are set by the control plane and must be documented "
                    "   in a comment on the meter declaration: CIR = 5 Mbps, "
                    "   CBS = 100000 bytes, PIR = 8 Mbps, PBS = 200000 bytes. "
                    "6. Apply video_meter AFTER sfc_forward so only packets with a resolved "
                    "   SFC path are metered, and guard the diffserv write with "
                    "   hdr.ipv4.isValid(). "
                    "Output the complete P4 program."
                ),
                "extra": {
                    "construct": "direct_meter",
                    "meter": "video_rate_meter",
                    "meter_type": "MeterType.bytes",
                    "colors": {"GREEN": 0, "YELLOW": 1, "RED": 2},
                    "cir_mbps": 5, "pir_mbps": 8,
                    "cbs_bytes": 100000, "pbs_bytes": 200000,
                    "yellow_dscp": "0x30",
                    "spi": 100,
                },
            },
            {
                "scenario": "S11_ternary_range_lpm_acl",
                "task": (
                    "Add a subscriber ACL in front of the video chain that needs four "
                    "different match kinds in a single table key — this exercises the "
                    "compiler's key-packing and P4Runtime priority semantics. "
                    "Requirements: "
                    "1. Define header udp_t { bit<16> src_port; bit<16> dst_port; "
                    "   bit<16> length; bit<16> checksum; } and parse it from parse_ipv4 "
                    "   when hdr.ipv4.protocol == 17. "
                    "2. Add a table 'video_acl' with the mixed key: "
                    "   key = { hdr.ipv4.src_addr: lpm; "
                    "           hdr.ipv4.dst_addr: ternary; "
                    "           hdr.udp.dst_port: range; "
                    "           meta.spi: exact; } "
                    "   actions = { permit; permit_and_mark; acl_drop; } "
                    "   size = 512; default_action = acl_drop(). "
                    "3. Actions: permit() is NoAction-equivalent; "
                    "   permit_and_mark(bit<8> dscp) writes hdr.ipv4.diffserv = dscp; "
                    "   acl_drop() calls mark_to_drop(std_meta) and counts into a "
                    "   DirectCounter(CounterType.packets) 'acl_drop_stats'. "
                    "4. Intended entries, expressed as a comment block above the table: "
                    "   (a) src 10.0.0.0/8, dst 192.168.0.0 mask 255.255.0.0, "
                    "       dst_port 16384..32767 (RTP), spi 100 -> permit_and_mark(0xB8) "
                    "       [priority 100]; "
                    "   (b) src 10.0.0.0/8, dst any (mask 0.0.0.0), dst_port 0..65535, "
                    "       spi 100 -> permit [priority 50]; "
                    "   (c) everything else -> acl_drop via default_action. "
                    "5. Explain in a comment why entries (a) and (b) require explicit, "
                    "   distinct P4Runtime priority values: a key containing ternary or "
                    "   range fields permits overlapping entries, so match order is not "
                    "   implied by the entry set. "
                    "6. Apply video_acl before sfc_forward, guarded by "
                    "   hdr.ipv4.isValid() && hdr.udp.isValid(). Deparser must emit hdr.udp "
                    "   after hdr.ipv4. "
                    "Output the complete P4 program."
                ),
                "extra": {
                    "construct": "match_kind_variety",
                    "table": "video_acl",
                    "match_kinds": ["lpm", "ternary", "range", "exact"],
                    "rtp_port_range": [16384, 32767],
                    "priorities": [100, 50],
                    "counter": "acl_drop_stats",
                },
            },
            {
                "scenario": "S12_arp_responder_l2",
                "task": (
                    "The SFF must answer ARP requests for the service-chain gateway "
                    "addresses itself, so subscriber set-top boxes resolve them without a "
                    "controller round trip. Implement a data-plane ARP responder. "
                    "Requirements: "
                    "1. Define header arp_t { bit<16> htype; bit<16> ptype; bit<8> hlen; "
                    "   bit<8> plen; bit<16> oper; bit<48> sha; bit<32> spa; bit<48> tha; "
                    "   bit<32> tpa; } and add it to headers_t as 'arp'. "
                    "2. Parser: ether_type 0x0806 -> parse_arp. In parse_arp, after "
                    "   extraction, transition accept only for Ethernet/IPv4 ARP: "
                    "   select (hdr.arp.htype, hdr.arp.ptype, hdr.arp.oper) { "
                    "   (1, 0x0800, 1): accept;  default: reject; } — only requests "
                    "   (oper == 1) are handled. "
                    "3. Action send_arp_reply(bit<48> gw_mac): use LOCAL TEMPORARIES to "
                    "   avoid clobbering fields mid-swap — "
                    "   bit<48> req_sha = hdr.arp.sha; bit<32> req_spa = hdr.arp.spa; "
                    "   bit<32> req_tpa = hdr.arp.tpa; then "
                    "   hdr.ethernet.dst_addr = req_sha; hdr.ethernet.src_addr = gw_mac; "
                    "   hdr.arp.oper = 2; hdr.arp.sha = gw_mac; hdr.arp.spa = req_tpa; "
                    "   hdr.arp.tha = req_sha; hdr.arp.tpa = req_spa; "
                    "   std_meta.egress_spec = std_meta.ingress_port. "
                    "4. Add a table 'arp_responder' with key { hdr.arp.tpa: exact; }, "
                    "   actions { send_arp_reply; drop; }, size = 64, "
                    "   default_action = drop() — the switch must not answer for addresses "
                    "   it does not own. Entry: tpa = 10.0.0.1 -> "
                    "   send_arp_reply(0x000000000001). "
                    "5. In the ingress apply block, handle ARP first and then exit: "
                    "   if (hdr.arp.isValid()) { arp_responder.apply(); exit; } — ARP "
                    "   packets must never enter the NSH/SFC path. "
                    "6. Deparser emits hdr.ethernet, hdr.arp, hdr.nsh, hdr.ipv4. "
                    "Output the complete P4 program."
                ),
                "extra": {
                    "construct": "arp_responder",
                    "ether_type": "0x0806",
                    "table": "arp_responder",
                    "gateway_ip": "10.0.0.1",
                    "gateway_mac": "00:00:00:00:00:01",
                    "handled_oper": 1,
                    "reply_oper": 2,
                },
            },
            {
                "scenario": "S13_checksum_verify_and_udp_recompute",
                "task": (
                    "Header rewriting inside the chain is producing packets that receivers "
                    "silently discard, because only the IPv4 checksum is maintained. "
                    "Implement full ingress verification and correct egress recomputation "
                    "for both IPv4 and UDP. "
                    "Requirements: "
                    "1. Implement MyVerifyChecksum (currently empty): "
                    "   verify_checksum(hdr.ipv4.isValid(), { hdr.ipv4.version, "
                    "   hdr.ipv4.ihl, hdr.ipv4.diffserv, hdr.ipv4.total_len, "
                    "   hdr.ipv4.identification, hdr.ipv4.flags, hdr.ipv4.frag_offset, "
                    "   hdr.ipv4.ttl, hdr.ipv4.protocol, hdr.ipv4.src_addr, "
                    "   hdr.ipv4.dst_addr }, hdr.ipv4.hdr_checksum, HashAlgorithm.csum16); "
                    "2. In the ingress apply block, before any table: if "
                    "   (std_meta.checksum_error == 1) { mark_to_drop(std_meta); "
                    "   ipv4_csum_err.count(0); exit; } where ipv4_csum_err is a "
                    "   counter(1, CounterType.packets). "
                    "3. Define header udp_t { bit<16> src_port; bit<16> dst_port; "
                    "   bit<16> length; bit<16> checksum; } parsed from parse_ipv4 when "
                    "   hdr.ipv4.protocol == 17. "
                    "4. Add a table 'chain_snat' with key { meta.spi: exact; }, action "
                    "   'rewrite_src(bit<32> new_src)' that writes hdr.ipv4.src_addr and "
                    "   also sets hdr.ipv4.diffserv = 0xB8. Entry: spi = 100 -> "
                    "   rewrite_src(203.0.113.10). Both mutated fields feed the checksum, "
                    "   so both must be written before MyComputeChecksum runs. "
                    "5. In MyComputeChecksum, keep the existing IPv4 update_checksum and "
                    "   ADD a UDP recompute using update_checksum_with_payload over the "
                    "   IPv4 pseudo-header: "
                    "   update_checksum_with_payload(hdr.udp.isValid(), "
                    "   { hdr.ipv4.src_addr, hdr.ipv4.dst_addr, (bit<8>)0, "
                    "     hdr.ipv4.protocol, hdr.udp.length, hdr.udp.src_port, "
                    "     hdr.udp.dst_port, hdr.udp.length }, hdr.udp.checksum, "
                    "   HashAlgorithm.csum16); "
                    "   update_checksum_with_payload (not update_checksum) is required "
                    "   because the UDP checksum covers the payload bytes. "
                    "6. Do not zero hdr.udp.checksum as a shortcut. Deparser emits "
                    "   ethernet, nsh, ipv4, udp. "
                    "Output the complete P4 program."
                ),
                "extra": {
                    "construct": "checksum_verify_update",
                    "verify_control": "MyVerifyChecksum",
                    "error_counter": "ipv4_csum_err",
                    "udp_primitive": "update_checksum_with_payload",
                    "algorithm": "HashAlgorithm.csum16",
                    "snat_ip": "203.0.113.10",
                    "spi": 100,
                },
            },
            {
                "scenario": "S14_rtp_bitfield_parsing",
                "task": (
                    "The chain must become RTP-aware so keyframes can be prioritised and "
                    "loss can be measured in the data plane. The RTP header is a dense, "
                    "non-byte-aligned bit-field layout — parse it exactly. "
                    "Requirements: "
                    "1. Define header rtp_t { bit<2> version; bit<1> padding; "
                    "   bit<1> extension; bit<4> csrc_count; bit<1> marker; "
                    "   bit<7> payload_type; bit<16> seq_num; bit<32> timestamp; "
                    "   bit<32> ssrc; } — note the marker/payload_type byte splits 1+7 and "
                    "   the first byte splits 2+1+1+4; the total is 96 bits (12 bytes). "
                    "2. Define udp_t (src_port, dst_port, length, checksum — all bit<16>) "
                    "   and parse it when hdr.ipv4.protocol == 17. From parse_udp, "
                    "   transition to parse_rtp when hdr.udp.dst_port == 5004, else accept. "
                    "3. In parse_rtp, after extraction, verify(hdr.rtp.version == 2, "
                    "   error.RtpVersionInvalid) with 'error { RtpVersionInvalid }' declared "
                    "   at file scope. "
                    "4. Add metadata fields meta.is_keyframe (bit<1>) and meta.rtp_idx "
                    "   (bit<16>). Set meta.is_keyframe = hdr.rtp.marker in ingress. "
                    "5. Loss detection: declare register<bit<16>>(1024) 'last_seq'. "
                    "   Compute meta.rtp_idx with hash(meta.rtp_idx, HashAlgorithm.crc16, "
                    "   (bit<16>)0, { hdr.rtp.ssrc }, (bit<32>)1024). Read "
                    "   last_seq[meta.rtp_idx] into a local; if the stored value is non-zero "
                    "   and hdr.rtp.seq_num != stored + 1, increment "
                    "   register<bit<32>>(1024) 'rtp_gap_count' at the same index. Then "
                    "   write hdr.rtp.seq_num back to last_seq[meta.rtp_idx]. "
                    "   Do not special-case the 16-bit sequence wrap; stored + 1 wraps "
                    "   correctly in bit<16> arithmetic. "
                    "6. If meta.is_keyframe == 1 (RTP marker set, i.e. last packet of a "
                    "   frame) and payload_type == 96 (dynamic H.264), set "
                    "   hdr.ipv4.diffserv = 0xB8 (DSCP EF). "
                    "7. Deparser emits ethernet, nsh, ipv4, udp, rtp. "
                    "Output the complete P4 program."
                ),
                "extra": {
                    "construct": "bitfield_packing",
                    "header": "rtp_t",
                    "header_bits": 96,
                    "unaligned_fields": ["version:2", "padding:1", "extension:1",
                                         "csrc_count:4", "marker:1", "payload_type:7"],
                    "rtp_udp_port": 5004,
                    "payload_type_h264": 96,
                    "registers": ["last_seq", "rtp_gap_count"],
                },
            },
            {
                "scenario": "S15_idle_timeout_flow_cache",
                "task": (
                    "Per-flow entries installed for video sessions are never removed and "
                    "the table is filling up. Add an aging flow cache in front of "
                    "sfc_forward using P4 table idle-timeout support. "
                    "Requirements: "
                    "1. Define udp_t (src_port, dst_port, length, checksum — all bit<16>) "
                    "   and parse it when hdr.ipv4.protocol == 17. "
                    "2. Add a table 'sfc_flow_cache' with key "
                    "   { hdr.ipv4.src_addr: exact; hdr.ipv4.dst_addr: exact; "
                    "     hdr.udp.src_port: exact; hdr.udp.dst_port: exact; } "
                    "   actions = { set_cached_path; NoAction; } "
                    "   size = 8192; default_action = NoAction(); "
                    "   and the table property 'support_timeout = true;' — this is what "
                    "   makes the target emit P4Runtime IdleTimeoutNotification messages "
                    "   and is the only P4-side change aging requires. "
                    "3. Action set_cached_path(bit<9> port) writes std_meta.egress_spec = "
                    "   port and sets meta.cache_hit = 1 (new bit<1> metadata field). "
                    "4. Control flow in ingress: apply sfc_flow_cache first. If "
                    "   meta.cache_hit == 1, skip sfc_forward entirely (the cached decision "
                    "   stands). Otherwise fall through to the existing "
                    "   decrement_si() + sfc_forward.apply() path. The NSH SI decrement must "
                    "   still happen exactly once on the cache-hit path too. "
                    "5. Add counter(2, CounterType.packets) 'cache_stats': count index 0 on "
                    "   a hit and index 1 on a miss. "
                    "6. Document in a comment above the table that the idle timeout is "
                    "   30 seconds, configured per entry by the control plane via the "
                    "   P4Runtime TableEntry.idle_timeout_ns field (30000000000), and that "
                    "   entry deletion on notification is a control-plane action — the P4 "
                    "   program must not attempt to delete entries. "
                    "Output the complete P4 program."
                ),
                "extra": {
                    "construct": "idle_timeout",
                    "table": "sfc_flow_cache",
                    "table_property": "support_timeout",
                    "size": 8192,
                    "idle_timeout_ns": 30000000000,
                    "counter": "cache_stats",
                    "key_fields": ["ipv4.src_addr", "ipv4.dst_addr",
                                   "udp.src_port", "udp.dst_port"],
                },
            },
        ]

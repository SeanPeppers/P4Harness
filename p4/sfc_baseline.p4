/* Baseline SFC P4_16 program — stub for p4harness agentic editing. */

#include <core.p4>
#include <v1model.p4>

/* NSH (Network Service Header) — RFC 8300 */
header nsh_t {
    bit<2>  ver;
    bit<1>  oam;
    bit<1>  unused;
    bit<6>  ttl;
    bit<6>  len;
    bit<4>  unused2;
    bit<4>  md_type;
    bit<8>  next_proto;
    bit<24> spi;      // Service Path Identifier
    bit<8>  si;       // Service Index
}

header ethernet_t {
    bit<48> dst_addr;
    bit<48> src_addr;
    bit<16> ether_type;
}

header ipv4_t {
    bit<4>  version;
    bit<4>  ihl;
    bit<8>  diffserv;
    bit<16> total_len;
    bit<16> identification;
    bit<3>  flags;
    bit<13> frag_offset;
    bit<8>  ttl;
    bit<8>  protocol;
    bit<16> hdr_checksum;
    bit<32> src_addr;
    bit<32> dst_addr;
}

struct headers_t {
    ethernet_t ethernet;
    nsh_t      nsh;
    ipv4_t     ipv4;
}

struct metadata_t {
    bit<24> spi;
    bit<8>  si;
}

/* Parser */
parser MyParser(packet_in pkt, out headers_t hdr,
                inout metadata_t meta, inout standard_metadata_t std_meta) {
    state start {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {
            0x894F: parse_nsh;
            0x0800: parse_ipv4;
            default: accept;
        }
    }
    state parse_nsh {
        pkt.extract(hdr.nsh);
        meta.spi = hdr.nsh.spi;
        meta.si  = hdr.nsh.si;
        transition parse_ipv4;
    }
    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition accept;
    }
}

/* Checksum verification */
control MyVerifyChecksum(inout headers_t hdr, inout metadata_t meta) {
    apply {}
}

/* Ingress */
control MyIngress(inout headers_t hdr, inout metadata_t meta,
                  inout standard_metadata_t std_meta) {

    action drop() { mark_to_drop(std_meta); }

    action forward(bit<9> port) { std_meta.egress_spec = port; }

    action decrement_si() {
        hdr.nsh.si = hdr.nsh.si - 1;
    }

    table sfc_forward {
        key = {
            meta.spi: exact;
            meta.si:  exact;
        }
        actions = { forward; drop; }
        size = 1024;
        default_action = drop();
    }

    apply {
        if (hdr.nsh.isValid()) {
            decrement_si();
            sfc_forward.apply();
        } else {
            drop();
        }
    }
}

/* Egress — placeholder */
control MyEgress(inout headers_t hdr, inout metadata_t meta,
                 inout standard_metadata_t std_meta) {
    apply {}
}

/* Checksum computation */
control MyComputeChecksum(inout headers_t hdr, inout metadata_t meta) {
    apply {
        update_checksum(hdr.ipv4.isValid(),
            { hdr.ipv4.version, hdr.ipv4.ihl, hdr.ipv4.diffserv,
              hdr.ipv4.total_len, hdr.ipv4.identification,
              hdr.ipv4.flags, hdr.ipv4.frag_offset, hdr.ipv4.ttl,
              hdr.ipv4.protocol, hdr.ipv4.src_addr, hdr.ipv4.dst_addr },
            hdr.ipv4.hdr_checksum, HashAlgorithm.csum16);
    }
}

/* Deparser */
control MyDeparser(packet_out pkt, in headers_t hdr) {
    apply {
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.nsh);
        pkt.emit(hdr.ipv4);
    }
}

V1Switch(MyParser(), MyVerifyChecksum(), MyIngress(), MyEgress(),
         MyComputeChecksum(), MyDeparser()) main;

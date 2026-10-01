#!/usr/bin/env python3
"""Lab 02-3 · NCCL bench: run the playbook's two-Spark NCCL test and read algbw, busbw and the verdict.

Spark A launches `all_gather_perf` on both Sparks with mpirun — the exact command from
the NCCL playbook (two nodes, direct). The lab parses the nccl-tests table, explains
algbw vs busbw with your numbers, and compares the result with the pass mark the
playbook's own cluster script uses (21.875 GB/s ≈ 175 Gb/s for a direct link).

Every run also tests the parsers for real on NVIDIA's published RDMA sample
(ib_write_bw on two Sparks, from the playbook's benchmarking guide).

Options (from a shell):  --op all_reduce   run all_reduce_perf instead (same build; course addition)
                         --rdma            also run the raw RDMA test (ib_write_bw, needs `perftest`)

Run: .venv/bin/python week25/02_two_sparks_nccl/labs/lab03_nccl_bench.py
"""
import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, bar, check, note, result, sh, step, table, warn  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--op", choices=["all_gather", "all_reduce"], default="all_gather")
ap.add_argument("--rdma", action="store_true")
args = ap.parse_args()

MGMT = os.environ.get("MGMT_IFNAME", "enP7s7")            # the playbook's name; Wi-Fi: MGMT_IFNAME=wlP9s9
LINK_GBS = 200 / 8                                         # 200 Gb/s line rate = 25 GB/s
PASS_DIRECT_GBS = 21.875                                   # spark_cluster_setup.py: MIN_NCCL_TEST_BW  (175 Gbps)
PASS_RING_GBS = 10                                         # spark_cluster_setup.py: MIN_NCCL_TEST_BW_RING (80 Gbps)

# ── REFERENCE: the playbook's RDMA sample (Connect Two Sparks → performance_benchmarking_guide.md) ──
REF_RDMA_1 = """ #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]    MsgRate[Mpps]
 65536      882805           0.00               92.57                0.176554
 65536      882802           0.00               92.57                0.176554
 65536      882791           0.00               92.57                0.176554
 65536      882791           0.00               92.56                0.176552
 65536      882821           0.00               92.57                0.176555"""
REF_RDMA_2 = """ #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]    MsgRate[Mpps]
 65536      927940           0.00               97.28                0.185548
 65536      927790           0.00               97.28                0.185549
 65536      927766           0.00               97.28                0.185550
 65536      927754           0.00               97.28                0.185545
 65536      927804           0.00               97.29                0.185557
 65536      927807           0.00               97.28                0.185554"""
REF_RDMA_TOTAL = 189.85                                    # "Total throughput = 92.57 + 97.28 = 189.85 Gbps"

# ── EXAMPLE: the nccl-tests table shape. The playbook prints no sample output, so these numbers are
#    illustrative (chosen so algbw = 2 × busbw is easy to see), not a measurement. ──
EX_NCCL = """# nThread 1 nGpus 1 minBytes 17179869184 maxBytes 17179869184 step: 2(factor) warmup iters: 1 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid   4242 on    spark-a device  0 [000f:01:00] NVIDIA GB10
#  Rank  1 Group  0 Pid   4243 on    spark-b device  0 [000f:01:00] NVIDIA GB10
#
#                                                              out-of-place                       in-place
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
 17179869184    4294967296     float    none      -1   390452   44.00   22.00      0   390210   44.03   22.01      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 22.005
#"""

BUS_FACTOR = {"all_reduce": lambda n: 2 * (n - 1) / n, "all_gather": lambda n: (n - 1) / n,
              "reduce_scatter": lambda n: (n - 1) / n, "broadcast": lambda n: 1.0}


def parse_nccl(text: str) -> dict:
    """nccl-tests output → {'rows': [{size, time_us, algbw, busbw, ...}], 'avg_busbw': float|None}.

    Columns are found from the header line (the one with 'algbw' and 'busbw'), so the parser survives
    small format changes. 'Avg bus bandwidth' is matched with the same regex NVIDIA's cluster script uses.
    """
    head, rows, avg = None, [], None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#") and "algbw" in s and "busbw" in s:
            head = s.lstrip("#").split()
            continue
        m = re.match(r"# Avg bus bandwidth\s*:\s*([0-9.]+)", s)
        if m:
            avg = float(m.group(1))
            continue
        if head and s and s[0].isdigit():
            cells = s.split()
            if len(cells) != len(head):
                continue
            i_oop = head.index("algbw")
            i_ip = len(head) - 1 - head[::-1].index("algbw")
            rows.append({"size": int(cells[0]), "type": cells[2], "redop": cells[3],
                         "time_us": float(cells[i_oop - 1]), "algbw": float(cells[i_oop]),
                         "busbw": float(cells[i_oop + 1]), "busbw_ip": float(cells[i_ip + 1])})
    return {"rows": rows, "avg_busbw": avg}


def parse_ib_write_bw(text: str) -> float:
    """Mean of the 'BW average[Gb/sec]' column in perftest output (Gb/s)."""
    vals, on = [], False
    for line in text.splitlines():
        if "BW average" in line:
            on = True
            continue
        cells = line.split()
        if on and len(cells) == 5 and cells[0].isdigit():
            vals.append(float(cells[3]))
    return sum(vals) / len(vals) if vals else 0.0


def ipv4(text: str) -> str:
    m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", text)
    return m.group(1) if m else ""


banner(f"Lab 02-3 · NCCL {args.op} across two Sparks", "the playbook's test · algbw vs busbw · pass or not")

step(1, "is NCCL built on both Sparks? (NCCL playbook, Steps 2–3)")
exe = f"{args.op}_perf"
built = {}
for which in ("a", "b"):
    r = sh(f"ls ~/nccl/build/lib/libnccl.so.2 ~/nccl-tests/build/{exe}", which, timeout=30,
           example=f"/home/nvidia/nccl/build/lib/libnccl.so.2\n/home/nvidia/nccl-tests/build/{exe}")
    built[which] = r.ok and "No such file" not in r.out
    live = r.source == "live"
if live and not all(built.values()):
    warn("NCCL is not built on every Spark. The playbook's quick start builds both (it asks for each sudo password), "
         "so run it in the ⌨ terminal on Spark A:")
    print('$ curl -fsSL "https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/'
          'playbook-nccl/assets/setup.sh" -o setup.sh')
    print("$ bash setup.sh <NODE_2_IP>")

step(2, f"management IPs — mpirun reaches the other Spark over {MGMT} (set MGMT_IFNAME for Wi-Fi)")
ips = {}
for which, ex in (("a", "10.0.0.21"), ("b", "10.0.0.22")):
    r = sh(f"ip -4 -o addr show {MGMT}", which, timeout=30,
           example=f"2: {MGMT}    inet {ex}/24 brd 10.0.0.255 scope global dynamic noprefixroute {MGMT}")
    ips[which] = ipv4(r.out)
print(f"◆ Node 1 = {ips['a'] or '?'} (launcher) · Node 2 = {ips['b'] or '?'}")

step(3, f"run {exe} on both Sparks (16 GB buffer, as in the playbook's second command)")
cmd = f"""export CUDA_HOME="/usr/local/cuda"
export MPI_HOME="/usr/lib/aarch64-linux-gnu/openmpi"
export NCCL_HOME="$HOME/nccl/build/"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64/:$MPI_HOME/lib:$LD_LIBRARY_PATH"
mpirun -np 2 -H {ips['a'] or '<NODE_1_IP>'}:1,{ips['b'] or '<NODE_2_IP>'}:1 \\
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \\
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \\
  -x UCX_NET_DEVICES={MGMT} \\
  -x NCCL_SOCKET_IFNAME={MGMT} \\
  -x OMPI_MCA_btl_tcp_if_include={MGMT} \\
  $HOME/nccl-tests/build/{exe} -b 16G -e 16G -f 2"""
if live and not (all(built.values()) and all(ips.values())):
    warn("skipping the run: NCCL is missing or a management IP was not found (see above).")
    out, src = "", "skipped"
else:
    ex = EX_NCCL if args.op == "all_gather" else EX_NCCL.replace("none      -1", " sum      -1").replace(
        "4294967296", "4294967296").replace("44.00   22.00", "22.00   22.00").replace(
        "44.03   22.01", "22.01   22.01").replace("390452", "780903").replace("390210", "780496")
    r = sh(cmd, "a", example=ex, timeout=600)
    out, src = r.out, r.source

step(4, "read the table")
res = parse_nccl(out)
n = 2
for row in res["rows"]:
    table([[f"{row['size'] / 2**30:.0f} GiB", f"{row['time_us'] / 1000:.1f} ms", f"{row['algbw']:.2f} GB/s",
            f"{row['busbw']:.2f} GB/s", f"{row['busbw'] * 8:.0f} Gb/s"]],
          ["size", "time", "algbw", "busbw", "busbw × 8"])
    f = BUS_FACTOR[args.op](n)
    print(f"◆ algbw = size ÷ time = {row['size'] / 1e9:.2f} GB ÷ {row['time_us'] / 1e6:.3f} s = {row['algbw']:.2f} GB/s")
    print(f"◆ busbw = algbw × {'(n−1)/n' if args.op == 'all_gather' else '2(n−1)/n'} = {row['algbw']:.2f} × {f:g} "
          f"= {row['algbw'] * f:.2f} GB/s  (nccl-tests printed {row['busbw']:.2f})")
avg = res["avg_busbw"] if res["avg_busbw"] is not None else (
    sum(r_["busbw"] for r_ in res["rows"]) / len(res["rows"]) if res["rows"] else 0.0)
if res["rows"]:
    print(f"│ busbw     {bar(avg, LINK_GBS)} {avg:6.2f} GB/s")
    print(f"│ pass mark {bar(PASS_DIRECT_GBS, LINK_GBS)} {PASS_DIRECT_GBS:6.3f} GB/s  (playbook script, direct link)")
    print(f"│ line rate {bar(LINK_GBS, LINK_GBS)} {LINK_GBS:6.2f} GB/s  (200 Gb/s ÷ 8)")
    passed = avg >= PASS_DIRECT_GBS
    if src == "live":
        check(passed, f"Avg bus bandwidth {avg:.2f} GB/s ≥ {PASS_DIRECT_GBS} GB/s — the link performs as NVIDIA expects",
              f"Avg bus bandwidth {avg:.2f} GB/s is below {PASS_DIRECT_GBS} GB/s — stop other GPU jobs and rerun; "
              "then check lab01 (both logical interfaces addressed?)")
    else:
        print(f"◈ on this EXAMPLE text the verdict would be {'pass' if passed else 'fail'} — not your link")
else:
    warn("no nccl-tests table in the output — look at the mpirun errors above (Troubleshooting).")

step(5, "parser self-test on the playbook's published RDMA sample (REFERENCE, runs everywhere)")
bw1, bw2 = parse_ib_write_bw(REF_RDMA_1), parse_ib_write_bw(REF_RDMA_2)
table([["client 1 (rocep1s0f0)", f"{bw1:.2f} Gb/s"], ["client 2 (roceP2p1s0f0)", f"{bw2:.2f} Gb/s"],
       ["total", f"{bw1 + bw2:.2f} Gb/s = {(bw1 + bw2) / 8:.2f} GB/s"]], ["RDMA write test", "BW average"])
check(abs((bw1 + bw2) - REF_RDMA_TOTAL) < 0.02,
      f"parser total {bw1 + bw2:.2f} Gb/s matches the playbook's stated {REF_RDMA_TOTAL} Gbps",
      f"parser total {bw1 + bw2:.2f} Gb/s differs from the playbook's {REF_RDMA_TOTAL} — parser bug")
note("One QSFP port is two logical interfaces; the playbook runs one RDMA test on each and adds them. "
     "That is why both need an IP address to reach ~190 Gb/s.")

if args.rdma:
    step(6, "raw RDMA test on your link (perftest's ib_write_bw; server on A, clients on B)")
    pairs = re.findall(r"^(\S+) port \d+ ==> (\S+) \(Up\)", sh("ibdev2netdev", "a", timeout=30).out,
                       re.M) if src == "live" else []
    if not pairs:
        warn("needs two live Sparks with Up interfaces — skipped (the commands are in the tutorial).")
    else:
        srv = " ".join(f"nohup ib_write_bw -d {d} -i 1 -p {12000 + i} -F --report_gbits "
                         f"> ~/w25/logs/ib_srv_{i}.log 2>&1 &" for i, (d, _) in enumerate(pairs))
        sh(f"mkdir -p ~/w25/logs ; {srv} sleep 2", "a", timeout=30)
        peers = [ipv4(sh(f"ip -4 -o addr show {nd}", "a", timeout=30, quiet=True).out) for _, nd in pairs]
        ib_b = dict((nd, d) for d, nd in re.findall(r"^(\S+) port \d+ ==> (\S+) \(Up\)",
                                                    sh("ibdev2netdev", "b", timeout=30, quiet=True).out, re.M))
        cl = " ; ".join(f"(ib_write_bw -d {ib_b.get(nd, d)} -i 1 -p {12000 + i} -F --report_gbits {peers[i]} "
                        f"> /tmp/ib_cl_{i}.txt 2>&1 &)" for i, (d, nd) in enumerate(pairs))
        r = sh(f"{cl} ; sleep 20 ; cat " + " ".join(f"/tmp/ib_cl_{i}.txt" for i in range(len(pairs))), "b",
               timeout=120)
        per = [parse_ib_write_bw(chunk) for chunk in r.out.split("RDMA_Write BW Test")[1:]]
        total = sum(per)
        check(total >= 184, f"RDMA total {total:.2f} Gb/s — at or above NVIDIA Sync's 184 Gb/s lower bound",
              f"RDMA total {total:.2f} Gb/s — below NVIDIA Sync's 184 Gb/s lower bound")

print()
if src == "live" and res["rows"]:
    result(f"measured on your link: busbw {avg:.2f} GB/s ({avg * 8:.0f} Gb/s). Save this number — Modules 05 and 11 use it.")
else:
    result("DRY: the NCCL table is an EXAMPLE shape (the playbook prints none); the RDMA check used NVIDIA's real sample. "
           "Connect both Sparks to measure yours.")

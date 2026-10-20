#!/usr/bin/env python3
"""Exercise 02 · reference solution — read the link: which ports are Up, what busbw means, and did the test pass?

Fill in the three TODOs, save, then run:
    .venv/bin/python week25/02_two_sparks_nccl/exercises/ex02_read_the_link.py

The checker is free and offline. It feeds your functions the real `ibdev2netdev` outputs
printed in three NVIDIA playbooks, then known bus-bandwidth factors, then the pass marks
NVIDIA's own cluster script uses. Stuck? Compare with exercises/solutions/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from sparkkit import banner, check  # noqa: E402

# ── TODO 1 ── return the netdev names that are Up, in the order they appear.
#   A line looks like:  rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)
def up_interfaces(ibdev2netdev_text: str) -> list[str]:
    out = []
    for line in ibdev2netdev_text.splitlines():
        parts = line.split()
        if len(parts) == 6 and parts[3] == "==>" and parts[5] == "(Up)":
            out.append(parts[4])
    return out


# ── TODO 2 ── the factor nccl-tests multiplies algbw by to get busbw, for n ranks.
#   all_reduce: 2(n-1)/n · all_gather and reduce_scatter: (n-1)/n · broadcast: 1
def bus_factor(op: str, n: int) -> float:
    if op == "all_reduce":
        return 2 * (n - 1) / n
    if op in ("all_gather", "reduce_scatter"):
        return (n - 1) / n
    return 1.0


# ── TODO 3 ── verdict for an "Avg bus bandwidth" in GB/s: return "pass" or "low".
#   NVIDIA's spark_cluster_setup.py: 21.875 GB/s (175 Gbps) for direct or switch, 10 GB/s (80 Gbps) for a ring.
def verdict(busbw_gbs: float, topology: str) -> str:
    need = 10.0 if topology == "ring" else 21.875
    return "pass" if busbw_gbs >= need else "low"


# ─────────────────────────── checker — no need to edit below ────────────────
TWO_SPARKS = """roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Down)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)
rocep1s0f0 port 1 ==> enp1s0f0np0 (Down)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)"""                    # Connect Two Sparks, Step 2
NCCL = """rocep1s0f0 port 1 ==> enp1s0f0np0 (Up)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Down)
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Up)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Down)"""              # NCCL, two nodes, Step 4
RING = """nvidia@node-1:~$ ibdev2netdev
rocep1s0f0 port 1 ==> enp1s0f0np0 (Up)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Up)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)"""               # Connect Three Sparks, Step 2


def close(a, b) -> bool:
    return isinstance(a, (int, float)) and abs(a - b) < 1e-9


def main() -> None:
    banner("Exercise 02 · read the link", "offline checker · free · no Spark needed", status=False)
    ok = True
    ok &= check(up_interfaces(TWO_SPARKS) == ["enP2p1s0f1np1", "enp1s0f1np1"]
                and up_interfaces(NCCL) == ["enp1s0f0np0", "enP2p1s0f0np0"] and len(up_interfaces(RING) or []) == 4,
                "up_interfaces: two-Spark sample → port 1 (2 netdevs) · NCCL sample → port 0 · ring sample → all 4",
                f"TODO 1: up_interfaces(two-Spark sample) should be ['enP2p1s0f1np1', 'enp1s0f1np1'] "
                f"(got {up_interfaces(TWO_SPARKS)!r})")
    ok &= check(close(bus_factor("all_reduce", 2), 1.0) and close(bus_factor("all_gather", 2), 0.5)
                and close(bus_factor("all_reduce", 4), 1.5) and close(bus_factor("broadcast", 3), 1.0),
                "bus_factor: all_reduce n=2 → 1 · all_gather n=2 → 0.5 · all_reduce n=4 → 1.5 · broadcast → 1",
                f"TODO 2: bus_factor('all_gather', 2) should be 0.5 (got {bus_factor('all_gather', 2)!r})")
    ok &= check([verdict(x, t) for x, t in ((22.0, "direct"), (21.0, "direct"), (12.0, "ring"), (9.5, "ring"),
                                            (21.875, "switch"))] == ["pass", "low", "pass", "low", "pass"],
                "verdict: 22 direct → pass · 21 direct → low · 12 ring → pass · 9.5 ring → low · 21.875 switch → pass",
                f"TODO 3: verdict(21.0, 'direct') should be 'low' (got {verdict(21.0, 'direct')!r})")
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)

    print("\n▣ your functions, applied to three practice runs (EXAMPLE numbers, not measurements)")
    for op, n, topo, algbw in [("all_gather", 2, "direct", 44.6), ("all_reduce", 2, "direct", 19.0),
                               ("all_gather", 3, "ring", 16.2)]:
        bus = algbw * bus_factor(op, n)
        print(f"│ {op:10s} n={n} {topo:6s} algbw {algbw:5.1f} GB/s → busbw {bus:5.2f} GB/s "
              f"({bus * 8:5.1f} Gb/s) → {verdict(bus, topo)}")
    print("\n═ busbw, not algbw, is the number to compare with the 200 Gb/s cable. Why? See the hint in the tutorial.")


if __name__ == "__main__":
    main()

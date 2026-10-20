#!/usr/bin/env python3
"""Lab 02-1 · Link check: is the QSFP cable up, addressed, and usable on BOTH Sparks?

Read-only. On Spark A (SPARK_HOST) and Spark B (SPARK_HOST2) it runs the checks the
Connect Two Sparks playbook uses: which ConnectX-7 interfaces are Up (`ibdev2netdev`),
their IPv4 address and MTU (`ip addr show`), the negotiated speed, and the username.
Then it pings Spark B across the cable and tries passwordless SSH over the link.

In DRY mode the interface list and Spark A's address are the playbook's REFERENCE
output; everything the playbook shows no output for is a labelled EXAMPLE.
Nothing here changes either machine.

Run: .venv/bin/python week25/02_two_sparks_nccl/labs/lab01_link_check.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, check, note, result, sh, step, table, warn  # noqa: E402

# ── playbook text (Connect Two Sparks, Steps 2 and 4) ─────────────────────────
REF_IBDEV = """roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Down)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)
rocep1s0f0 port 1 ==> enp1s0f0np0 (Down)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)"""

REF_IPADDR_A = """    4: enp1s0f1np1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
        link/ether 3c:6d:66:cc:b3:b7 brd ff:ff:ff:ff:ff:ff
        inet 192.168.100.10/24 brd 192.168.100.255 scope global noprefixroute enp1s0f1np1
          valid_lft forever preferred_lft forever"""

# Multi Sparks Through a Switch, Step 3.1 (after the speed fix) — the value every link should show.
REF_SPEED = "\tSpeed: 200000Mb/s"

# ── illustrative shapes for commands the playbooks show no output for ────────
def _ex_addr(dev: str, cidr: str, mac: str) -> str:
    return (REF_IPADDR_A.replace("enp1s0f1np1", dev).replace("192.168.100.10/24", cidr)
            .replace("192.168.100.255", cidr.rsplit(".", 1)[0] + ".255").replace("3c:6d:66:cc:b3:b7", mac))


# The playbook's netplan (Step 3): enp1s0f1np1 → 192.168.100.x, enP2p1s0f1np1 → 192.168.101.x; A = .10, B = .11
EX_ADDR = {("a", "enP2p1s0f1np1"): _ex_addr("enP2p1s0f1np1", "192.168.101.10/24", "3c:6d:66:cc:b3:b8"),
           ("b", "enp1s0f1np1"): _ex_addr("enp1s0f1np1", "192.168.100.11/24", "3c:6d:66:aa:bb:cc"),
           ("b", "enP2p1s0f1np1"): _ex_addr("enP2p1s0f1np1", "192.168.101.11/24", "3c:6d:66:aa:bb:cd")}
EX_WHOAMI = "nvidia"
EX_PING = """PING 192.168.100.11 (192.168.100.11) from 192.168.100.10 enp1s0f1np1: 56(84) bytes of data.
64 bytes from 192.168.100.11: icmp_seq=1 ttl=64 time=0.3 ms
--- 192.168.100.11 ping statistics ---
3 packets transmitted, 3 received, 0% packet loss"""
EX_SSH = "spark-b"


def up_interfaces(text: str) -> list[tuple[str, str]]:
    """ibdev2netdev lines → [(rdma_device, netdev)] for the ports that are Up."""
    return re.findall(r"^(\S+) port \d+ ==> (\S+) \(Up\)", text, re.M)


def ipv4_and_mtu(text: str) -> tuple[str, int]:
    ip = re.search(r"inet (\d+\.\d+\.\d+\.\d+/\d+)", text)
    mtu = re.search(r"\bmtu (\d+)", text)
    return (ip.group(1) if ip else ""), (int(mtu.group(1)) if mtu else 0)


def verdict(cond, good: str, bad: str, live: bool) -> bool:
    """✓/✕ on a live run. On DRY text it only says what the check would conclude — it is not your cable."""
    if live:
        return check(cond, good, bad)
    print(("◈ on this text: " + good) if cond else ("◈ on this text: ✕ " + bad))
    return bool(cond)


def port_of(netdev: str) -> str:
    """enp1s0f1np1 / enP2p1s0f1np1 → 'port 1'. Each physical QSFP port shows up as two netdevs."""
    m = re.search(r"f(\d)np\d$", netdev)
    return f"port {m.group(1)}" if m else "?"


banner("Lab 02-1 · link check on both Sparks",
       "which ConnectX-7 interfaces are Up, addressed, and reachable across the cable")

nodes = {}
for which in ("a", "b"):
    name = f"Spark {which.upper()}"
    step(1 if which == "a" else 2, f"{name}: interfaces, address, MTU, speed, user")
    r = sh("ibdev2netdev", which, reference=REF_IBDEV, timeout=30)
    ups = up_interfaces(r.out)
    if not ups:
        warn(f"{name}: no ConnectX-7 interface is Up. Reseat the QSFP cable (same port on both Sparks), "
             "reboot, and re-run ibdev2netdev.")
    info = {"src": r.source, "ups": ups, "addrs": {}, "user": ""}
    for rdma, dev in sorted(ups, key=lambda u: u[1].startswith("enP")):      # enp1s0… first, like the playbook
        if which == "a" and dev == "enp1s0f1np1":
            kw = {"reference": REF_IPADDR_A}
        else:
            kw = {"example": EX_ADDR.get((which, dev), _ex_addr(dev, "", "00:00:00:00:00:00"))}
        ra = sh(f"ip addr show {dev}", which, timeout=30, **kw)
        ip, mtu = ipv4_and_mtu(ra.out)
        rs = sh(f"ethtool {dev} 2>/dev/null | grep Speed || cat /sys/class/net/{dev}/speed", which,
                reference=REF_SPEED, timeout=30)
        sp = re.search(r"(\d{4,})", rs.out)
        info["addrs"][dev] = {"rdma": rdma, "ip": ip, "mtu": mtu, "speed": int(sp.group(1)) if sp else 0}
    info["user"] = sh("whoami", which, example=EX_WHOAMI, timeout=30).out.strip()
    nodes[which] = info

live = all(n["src"] == "live" for n in nodes.values())
chk = lambda c, g, b: verdict(c, g, b, live)          # noqa: E731

step(3, "compare the two ends")
rows = []
for which, info in nodes.items():
    for dev, a in info["addrs"].items():
        rows.append([f"Spark {which.upper()}", a["rdma"], dev, port_of(dev), a["ip"] or "—", a["mtu"] or "—",
                     f"{a['speed'] // 1000} Gb/s" if a["speed"] else "—", info["user"] or "—", info["src"]])
table(rows, ["node", "RDMA device", "netdev", "QSFP", "IPv4", "MTU", "speed", "user", "iface list"])

subnet = lambda cidr: ".".join(cidr.split("/")[0].split(".")[:3]) if cidr else ""  # noqa: E731
A, B = nodes["a"]["addrs"], nodes["b"]["addrs"]
a_dev = next(iter(A), "")
good = True
good &= chk(bool(A) and sorted(A) == sorted(B) and len({port_of(d) for d in A}) == 1,
            f"same physical port Up on both ends ({port_of(a_dev)}: {' + '.join(A)})",
            "the Up ports differ between the Sparks — cable the same port on both (Connect Two Sparks, Step 2)")
for dev in A:
    ia, ib = A[dev].get("ip", ""), B.get(dev, {}).get("ip", "")
    good &= chk(bool(ia and ib) and subnet(ia) == subnet(ib) and ia != ib,
                f"{dev}: both ends on one subnet ({ia} ↔ {ib})",
                f"{dev}: an end has no IPv4 address, or the ends are on different subnets — lab02 writes the netplan")
subs = [subnet(v["ip"]) for v in A.values() if v["ip"]]
good &= chk(len(subs) == len(set(subs)) == len(A),
            "each logical interface has its own subnet (" + ", ".join(s_ + ".x" for s_ in subs) + ")",
            "two interfaces share a subnet or one has none — the playbooks require distinct subnets")
mtus = {v["mtu"] for n in (A, B) for v in n.values()}
good &= chk(len(mtus) == 1 and 0 not in mtus, f"MTU is the same on every link interface ({mtus.pop() if len(mtus) == 1 else '?'})",
            "MTU differs between the ends — the Spark playbooks keep the default on both")
speeds = {v["speed"] for n in (A, B) for v in n.values()}
good &= chk(speeds == {200000}, "every link interface negotiated 200000 Mb/s",
            "a link is below 200 Gb/s — check the cable (QSFP112 DAC, Ethernet mode) or the switch port speed")
good &= chk(nodes["a"]["user"] and nodes["a"]["user"] == nodes["b"]["user"],
            f"same username on both Sparks ({nodes['a']['user']}) — mpirun and the helper scripts need it",
            "usernames differ — create the same user on both (Connect Two Sparks, Step 1)")
b = B.get(a_dev, {})

step(4, "across the cable: ping, then passwordless SSH from Spark A to Spark B")
peer = (b.get("ip") or "192.168.100.11").split("/")[0]
rp = sh(f"ping -c 3 -W 2 -I {a_dev or 'enp1s0f1np1'} {peer}", "a", example=EX_PING, timeout=30)
good &= chk(" 0% packet loss" in rp.out, f"Spark A reaches {peer} over the QSFP link",
            "ping across the link failed — check both IPs, and that each side's Up interface has one")
rs = sh(f"ssh -o BatchMode=yes -o ConnectTimeout=5 {peer} hostname", "a", example=EX_SSH, timeout=30)
good &= chk(rs.ok and bool(rs.out.strip()) and "denied" not in rs.out.lower(),
            f"passwordless SSH A → B over the link works (answered: {rs.out.strip().splitlines()[-1] if rs.out.strip() else '—'})",
            "SSH over the link asks for a password — run discover-sparks or ssh-copy-id (Connect Two Sparks, Step 4)")

print()
if live and good:
    result("the link is up, addressed and reachable. Next: lab03 measures it with NCCL.")
elif live:
    result("fix the ✕ lines above (the fix is printed on each), then run this lab again.")
else:
    note("DRY: interface names, Spark A's address and the speed are REFERENCE text from the playbooks; "
         "the other addresses, ping, ssh and whoami are EXAMPLE shapes. The ◈ lines test that text, not your cable.")
    result("set SPARK_HOST and SPARK_HOST2 (🖥 Spark setup) to check your own two Sparks.")

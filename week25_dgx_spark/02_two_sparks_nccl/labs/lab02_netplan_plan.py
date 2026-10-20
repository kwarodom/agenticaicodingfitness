#!/usr/bin/env python3
"""Lab 02-2 · Netplan plan: write the link's IP configuration for both Sparks — and apply it only if you say so.

From the interfaces that are Up on each Spark, this lab writes the two files the
Connect Two Sparks playbook asks for (Step 3, Option 1: /etc/netplan/40-cx7.yaml),
saves them on your laptop, validates them, and prints the exact commands to
install them and to roll them back.

It changes nothing unless you run it from a shell with SPARK_APPLY=1, and even then
only with passwordless sudo (`sudo -n`, it never asks for a password) and only when
/etc/netplan/40-cx7.yaml does not exist yet. The Lab Runner's ▶ button never sets SPARK_APPLY.

Run: .venv/bin/python week25/02_two_sparks_nccl/labs/lab02_netplan_plan.py
     SPARK_APPLY=1 .venv/bin/python week25/02_two_sparks_nccl/labs/lab02_netplan_plan.py   # apply
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, check, note, put, result, sh, step, table, warn, where  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / ".runs"          # gitignored
APPLY = os.environ.get("SPARK_APPLY") == "1"

REF_IBDEV = """roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Down)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)
rocep1s0f0 port 1 ==> enp1s0f0np0 (Down)
rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)"""

# The playbook's node 1 / node 2 files (Connect Two Sparks, Step 3 Option 1), byte for byte.
REF_NETPLAN = {
    "a": """network:
  version: 2
  ethernets:
    enp1s0f1np1:
      addresses:
        - 192.168.100.10/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - 192.168.101.10/24
      dhcp4: no
""",
    "b": """network:
  version: 2
  ethernets:
    enp1s0f1np1:
      addresses:
        - 192.168.100.11/24
      dhcp4: no
    enP2p1s0f1np1:
      addresses:
        - 192.168.101.11/24
      dhcp4: no
"""}

EX_IP_BR = """lo               UNKNOWN        127.0.0.1/8
enP7s7           UP             10.0.0.21/24
wlP9s9           DOWN
docker0          DOWN           172.17.0.1/16"""
EX_NETPLAN_LS = "50-cloud-init.yaml"
HOST_OCTET = {"a": 10, "b": 11}                # the playbook's node 1 = .10, node 2 = .11


def home_path(which: str, rel: str) -> str:
    """put() target in the Spark's home: '~/…' when we run ON the Spark, plain 'rel' over scp (scp is home-relative)."""
    return f"~/{rel}" if where(which) == "local" else rel


def up_interfaces(text: str) -> list[str]:
    devs = re.findall(r"^\S+ port \d+ ==> (\S+) \(Up\)", text, re.M)
    return sorted(devs, key=lambda d: (d.startswith("enP"), d))       # enp1s0… first, as in the playbook


def netplan_for(which: str, devs: list[str]) -> str:
    """One /24 per logical interface (192.168.100.x, .101.x, …) — distinct subnets, as the playbooks require."""
    lines = ["network:", "  version: 2", "  ethernets:"]
    for i, dev in enumerate(devs):
        lines += [f"    {dev}:", "      addresses:", f"        - 192.168.{100 + i}.{HOST_OCTET[which]}/24",
                  "      dhcp4: no"]
    return "\n".join(lines) + "\n"


banner("Lab 02-2 · netplan plan for the QSFP link",
       "generate + validate /etc/netplan/40-cx7.yaml for both Sparks · applied only with SPARK_APPLY=1")

plans, sources = {}, {}
for n, which in enumerate(("a", "b"), 1):
    step(n, f"Spark {which.upper()}: which interfaces are Up, and are they already addressed?")
    r = sh("ibdev2netdev", which, reference=REF_IBDEV, timeout=30)
    devs = up_interfaces(r.out)
    sources[which] = r.source
    if not devs:
        warn(f"Spark {which.upper()}: no QSFP interface is Up — is the cable plugged into the same port on both "
             "Sparks? Reseat it (or reboot), then run this lab again. No file is generated for this Spark.")
    br = sh("ip -br -4 address", which, example=EX_IP_BR, timeout=30)
    have = [d for d in devs if re.search(rf"^{re.escape(d)}\s+\S+\s+\d", br.out, re.M)]
    if have:
        warn(f"{', '.join(have)} already {'has' if len(have) == 1 else 'have'} an IPv4 address on Spark "
             f"{which.upper()} — NVIDIA Sync Cluster Assistant or an earlier run configured it. "
             "The playbook says: do not repeat the manual steps.")
    ls = sh("ls /etc/netplan/", which, example=EX_NETPLAN_LS, timeout=30)
    plans[which] = {"devs": devs, "yaml": netplan_for(which, devs), "exists": "40-cx7.yaml" in ls.out,
                    "configured": bool(have)}

step(3, "the two files")
OUT.mkdir(parents=True, exist_ok=True)
for which, p in plans.items():
    if not p["devs"]:
        print(f"── Spark {which.upper()}: no Up interface, no file")
        continue
    path = OUT / f"40-cx7.spark-{which}.yaml"
    path.write_text(p["yaml"], encoding="utf-8")
    p["path"] = path
    print(f"── {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")
    print(p["yaml"].rstrip())
try:
    import yaml                                    # PyYAML is in the repo .venv
    parsed = {w: yaml.safe_load(p["yaml"]) for w, p in plans.items() if p["devs"]}
    ok_yaml = all(d["network"]["version"] == 2 and d["network"]["ethernets"] for d in parsed.values())
    if parsed:
        check(ok_yaml, f"{'both files parse' if len(parsed) == 2 else 'the generated file parses'} as YAML with "
                       "network.version 2 and one entry per Up interface",
              "a generated file does not parse — report this as a bug")
except ImportError:
    warn("PyYAML not installed — skipped the YAML parse check")
if all(s != "live" for s in sources.values()):
    same = all(plans[w]["yaml"] == REF_NETPLAN[w] for w in ("a", "b"))
    check(same, "generated from the playbook's interface list, both files match the playbook's Step 3 files "
                "byte for byte", "generated files differ from the playbook's — report this as a bug")

for which, p in plans.items():                  # copying into ~/w25 is harmless; /etc is only touched in step 5
    p["copied"] = bool(p["devs"]) and sources[which] == "live" and put(p["path"], home_path(which, "w25/40-cx7.yaml"), which)

rows = [[f"Spark {w.upper()}", d, f"192.168.{100 + i}.{HOST_OCTET[w]}/24"]
        for w, p in plans.items() for i, d in enumerate(p["devs"])]
table(rows, ["node", "interface", "address"])

step(4, "install on each Spark — run these in the ⌨ terminal, one Spark at a time")
note("The playbook pipes a heredoc into `sudo tee`; this is the same file, already copied to ~/w25/ in LIVE mode.")
print("$ sudo tee /etc/netplan/40-cx7.yaml > /dev/null < ~/w25/40-cx7.yaml")
print("$ sudo chmod 600 /etc/netplan/40-cx7.yaml")
print("$ sudo netplan apply")
note("Rollback (Connect Two Sparks, Step 6): sudo rm /etc/netplan/40-cx7.yaml && sudo netplan apply")
note("The file names only the ConnectX-7 interfaces, not your management interface. `netplan apply` can still "
     "blip the network for a moment, so keep a way back in (a local console, or a second ssh session).")

step(5, "apply?")
if not APPLY:
    result("not applied — nothing changed. To apply from a shell: SPARK_APPLY=1 .venv/bin/python "
           "week25/02_two_sparks_nccl/labs/lab02_netplan_plan.py (needs passwordless sudo), "
           "or paste the step-4 commands into the ⌨ terminal.")
    sys.exit(0)
for which, p in plans.items():
    tag = f"Spark {which.upper()}"
    if sources[which] != "live":
        warn(f"{tag}: DRY — nothing to apply to.")
        continue
    if not p["devs"]:
        warn(f"{tag}: no QSFP interface is Up — nothing to apply.")
        continue
    if p["exists"] or p["configured"]:
        warn(f"{tag}: /etc/netplan/40-cx7.yaml exists or the link is already addressed — not touching it.")
        continue
    if not p["copied"]:
        warn(f"{tag}: copy to ~/w25/40-cx7.yaml failed — not applied.")
        continue
    r = sh("sudo -n install -m 600 ~/w25/40-cx7.yaml /etc/netplan/40-cx7.yaml && sudo -n netplan apply", which,
           timeout=120)
    if r.ok:
        print(f"✓ {tag}: netplan applied")
    else:
        warn(f"{tag}: sudo needs a password here. The file is at ~/w25/40-cx7.yaml on the Spark — run the "
             "step-4 commands in the ⌨ terminal.")
result("done. Re-run lab01 to check both ends.")

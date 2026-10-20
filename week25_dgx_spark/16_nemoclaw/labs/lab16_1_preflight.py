#!/usr/bin/env python3
"""Lab 16-1 · NemoClaw preflight: is this Spark ready for an always-on sandboxed agent?

Ten read-only checks on your Spark (locally, or over `ssh $SPARK_HOST`), taken from the
NemoClaw playbook's prerequisites and its starter prompt's readiness list: OS, GPU,
Docker, Node.js, memory, disk, non-interactive sudo, an existing NemoClaw/OpenShell
install, the ports NemoClaw uses, and whether a vLLM server already answers on :8000.

Nothing here installs or changes anything. The last step tells you which onboarding
path fits (Express / managed vLLM, or "Existing vLLM").

Run: .venv/bin/python week25/16_nemoclaw/labs/lab16_1_preflight.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, check, note, result, sh, step, table  # noqa: E402

# (title, command, (kind, dry output), pass test, what it means if it fails)
#   kind "ref" = text quoted from the NemoClaw playbook; "ex" = illustrative shape written for this course.
CHECKS = [
    ("Operating system", "head -n 2 /etc/os-release",
     ("ex", 'PRETTY_NAME="Ubuntu 24.04.3 LTS"\nNAME="Ubuntu"'),
     lambda o: "24.04" in o,
     "The playbook expects Ubuntu 24.04 (DGX OS). Update from DGX Dashboard → Settings → Updates."),
    ("GPU visible", "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader",
     ("ex", "NVIDIA GB10, 580.95.05"),
     lambda o: "GB10" in o or "NVIDIA" in o,
     "No NVIDIA GPU. The installer's managed vLLM needs it. Reboot and re-run nvidia-smi."),
    ("Docker 28.x+", "docker info --format '{{.ServerVersion}}'",
     ("ex", "28.3.3"),
     lambda o: bool(re.match(r"\s*(2[89]|[3-9]\d)\.", o)),
     "The playbook wants Docker 28.x+ usable without sudo: sudo usermod -aG docker $USER, then log out and in."),
    ("Node.js", "node --version 2>/dev/null || echo 'node not installed'",
     ("ex", "node not installed"),
     lambda o: "not installed" in o or _node_ok(o),
     "Node.js is older than 22.16. The installer normally upgrades it; see Troubleshooting if it fails."),
    ("Unified memory", "free -g | head -2",
     ("ex", "               total        used        free      shared  buff/cache   available\n"
            "Mem:             119           9         102           0           8         109"),
     lambda o: _mem_avail(o) >= 60,
     "Less than ~60 GiB available. Stop other model servers first: NemoClaw's managed vLLM needs the room."),
    ("Free disk", "df -h ~ | tail -1",
     ("ex", "/dev/nvme0n1p2  3.7T  412G  3.1T  12% /"),
     lambda o: _free_gb(o) >= 200,
     "Under 200 GB free. The playbook warns large Express models can need hundreds of GB."),
    ("Non-interactive sudo",
     "sudo -n true 2>/dev/null && echo 'passwordless sudo: yes' || echo 'passwordless sudo: no — the installer will ask for your password'",
     ("ex", "passwordless sudo: no — the installer will ask for your password"),
     lambda o: True,
     ""),
    ("Existing NemoClaw / OpenShell",
     "command -v nemoclaw openshell 2>/dev/null; ls -d ~/.nemoclaw ~/.config/openshell 2>/dev/null; echo \"(end of list)\"",
     ("ex", "(end of list)"),
     lambda o: True,
     ""),
    ("Ports 8000 · 8080 · 18789 · 18790",
     "ss -ltn 2>/dev/null | grep -E ':(8000|8080|18789|18790)\\b' || echo 'none of 8000 8080 18789 18790 is listening'",
     ("ex", "none of 8000 8080 18789 18790 is listening"),
     lambda o: True,
     ""),
    ("vLLM already on :8000", "curl -s --max-time 3 http://127.0.0.1:8000/v1/models || echo 'no server on :8000'",
     ("ex", "no server on :8000"),
     lambda o: True,
     ""),
]


def _node_ok(out: str) -> bool:
    m = re.search(r"v(\d+)\.(\d+)", out)
    return bool(m) and (int(m.group(1)), int(m.group(2))) >= (22, 16)


def _mem_avail(out: str) -> float:
    for line in out.splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            return float(parts[-1]) if len(parts) >= 7 else float(parts[3])
    return 0.0


def _free_gb(df_line: str) -> float:
    parts = df_line.split()
    if len(parts) < 4:
        return 0.0
    m = re.match(r"([\d.]+)([KMGTP]?)", parts[3])
    if not m:
        return 0.0
    return float(m.group(1)) * {"": 1e-9, "K": 1e-6, "M": 1e-3, "G": 1, "T": 1e3, "P": 1e6}[m.group(2)]


banner("Lab 16-1 · NemoClaw preflight", "ten read-only checks before you run the NemoClaw installer")
note("Playbook: 'Use only a clean environment' — a fresh device or VM with no personal data or real credentials.")

rows, failures, outs = [], [], {}
for i, (title, cmd, (kind, dry_out), test, fix) in enumerate(CHECKS, 1):
    step(i, title)
    r = sh(cmd, timeout=30, **({"reference": dry_out} if kind == "ref" else {"example": dry_out}))
    outs[title] = r
    passed = bool(test(r.out or ""))
    status = ("✓" if passed else "✕") if r.live else "◈ " + r.source
    last = (r.out or "").strip().splitlines()
    rows.append([title, status, last[-1][:50] if last else "—"])
    if r.live and not passed and fix:
        failures.append((title, fix))

print()
table(rows, ["check", "result", "last line of output"])

step(len(CHECKS) + 1, "which onboarding path fits this Spark?")
vllm = outs["vLLM already on :8000"]
busy = outs["Ports 8000 · 8080 · 18789 · 18790"]
if vllm.live and '"data"' in (vllm.out or ""):
    note("A vLLM server already answers on :8000 → onboarding option 'Existing vLLM' (NEMOCLAW_PROVIDER=vllm).")
    note("Make sure it was started with tool calling enabled (Module 05 / the agent-ready recipe): agents need tool_calls.")
elif vllm.live:
    note("Nothing on :8000 → accept Express Install (managed vLLM, maintained model, sandbox 'my-assistant', Balanced policy).")
else:
    note("DRY: with a Spark you would see whether :8000 is free (Express Install) or already serving (Existing vLLM).")
if busy.live and ":8080" in (busy.out or ""):
    note("Port 8080 is taken. The OpenShell gateway uses it — see Troubleshooting ('port 8080 is held by container').")

if failures:
    print()
    for title, fix in failures:
        check(False, "", f"{title}: {fix}")
    result(f"{len(failures)} check(s) need attention before `curl -fsSL https://www.nvidia.com/nemoclaw.sh | bash`.")
elif all(r[1] == "✓" for r in rows):
    result("every check passed — this Spark is ready for the NemoClaw installer (Section 3).")
else:
    result("DRY run: EXAMPLE rows are illustrative shapes, not your Spark. Connect a Spark (🖥 Spark setup) to check yours.")

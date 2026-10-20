#!/usr/bin/env python3
"""Lab 15-1 · Install OpenShell and check the gateway — on the Spark (playbook) and on this laptop.

Part A follows the OpenShell playbook on the Spark, read-only: environment check (Step 1), Docker
access (Step 2), the CLI (Step 3) and the gateway service (Step 4). It never runs the installer for
you — that is a `curl | sh` you run yourself on the Spark (TUTORIAL §2). Part B runs the OpenShell CLI
on THIS laptop from week25/15_openshell_sandbox/.venv-openshell (a pinned PyPI wheel, no installer,
no sudo): version, `openshell status` and `openshell doctor check` — real output, no gateway needed.

Run: .venv/bin/python week25/15_openshell_sandbox/labs/lab15_1_install_and_gateway.py
"""
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, note, ok, result, sh, step, table, warn, where  # noqa: E402

MOD = Path(__file__).resolve().parents[1]
OPENSHELL = Path(os.environ.get("OPENSHELL_BIN", MOD / ".venv-openshell" / "bin" / "openshell"))
ANSI = re.compile(r"\x1b\[[0-9;]*m")
PIN = "openshell==0.0.111"

banner("Lab 15-1 · install OpenShell and check the gateway",
       "Spark: playbook Steps 1–4 (read-only) · laptop: the real CLI from a week25-local venv")

step("A1", "confirm the Spark's environment (playbook Step 1)")
rows = []
for title, cmd, ex in [
        ("OS", "head -n 2 /etc/os-release", 'PRETTY_NAME="Ubuntu 24.04.3 LTS"\nNAME="Ubuntu"'),
        ("GPU", "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader", "NVIDIA GB10, 580.95.05"),
        ("Docker", "docker info --format '{{.ServerVersion}}'", "28.3.3"),
        ("Python ≥ 3.12", "python3 --version", "Python 3.12.3"),
        ("Docker without sudo", "docker ps --format '{{.Names}}' | head -3 && echo ok", "ok")]:
    r = sh(cmd, example=ex, timeout=60)
    rows.append([title, "✓" if r.live and r.ok else ("✕" if r.live else "◈ " + r.source),
                 (r.out.strip().splitlines() or ["—"])[-1][:48]])
table(rows, ["check", "result", "last line"])
note("Docker must also use the NVIDIA runtime (playbook Step 2: sudo nvidia-ctk runtime configure --runtime=docker). "
     "Run that yourself — labs never sudo.")

step("A2", "is the OpenShell CLI installed on the Spark? (playbook Step 3 — you run the installer, not this lab)")
r = sh("command -v openshell && openshell --version || echo 'openshell not installed — see TUTORIAL §2'",
       example="/home/<you>/.local/bin/openshell\nopenshell <version>", timeout=60)

step("A3", "is the gateway running? (playbook Step 4)")
sh("systemctl --user status --no-pager openshell-gateway | head -5; openshell status",
   example="● openshell-gateway.service - OpenShell gateway\n     Active: active (running)\n"
           "Gateway Status\n  Status: Connected", timeout=60)
note("Playbook: `openshell status` should report the gateway as Connected. If not: "
     "systemctl --user start openshell-gateway, then journalctl --user -u openshell-gateway -f.")

step("B1", f"the OpenShell CLI on this laptop ({PIN} in week25/15_openshell_sandbox/.venv-openshell)")
if not OPENSHELL.exists():
    warn("not installed on this laptop. It is optional; to add it (week25-local, no sudo, no installer script):")
    print("$ uv venv -p 3.12 week25/15_openshell_sandbox/.venv-openshell")
    print(f'$ uv pip install --python week25/15_openshell_sandbox/.venv-openshell/bin/python "{PIN}"')
    result("Part A is the playbook; Part B needs the laptop CLI. Everything else in Module 15 works without it.")
    sys.exit(0)
env = {**os.environ, "NO_COLOR": "1", "HOME": str(MOD / ".runs" / "openshell-home")}
(MOD / ".runs" / "openshell-home").mkdir(parents=True, exist_ok=True)


def cli(*args: str, limit: int = 40) -> subprocess.CompletedProcess:
    print("$ openshell " + " ".join(args) + "   [this laptop]")
    p = subprocess.run([str(OPENSHELL), *args], capture_output=True, text=True, timeout=60, env=env)
    lines = [ln for ln in ANSI.sub("", (p.stdout + p.stderr)).splitlines() if ln.strip()]
    print("\n".join(lines[:limit]) + (f"\n  … ({len(lines) - limit} more lines)" if len(lines) > limit else ""))
    return p


cli("--version")
p = cli("--help", limit=22)
cmds = re.findall(r"^\s{2}([a-z-]+):", ANSI.sub("", p.stdout), re.M)
ok(f"{len(cmds)} top-level commands, including " + ", ".join(c for c in cmds if c in
                                                            ("sandbox", "policy", "provider", "inference", "logs", "term", "gateway")))

step("B2", "status and prerequisites on the laptop (no gateway here — the output says so)")
cli("status")
cli("doctor", "check")
note("A gateway needs Docker (plus k3s inside it). This lab never starts one on the laptop: the course runs the "
     "gateway on the Spark, and a laptop CLI can manage it remotely (`openshell gateway add … --remote`, TUTORIAL §2).")

result(f"Spark: {'LIVE checks above' if where() != 'dry' else 'DRY — connect a Spark to run the checks'}. "
       "Laptop: the real CLI answers; lab 15-2 uses its policy parser.")

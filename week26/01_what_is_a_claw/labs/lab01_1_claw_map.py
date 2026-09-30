#!/usr/bin/env python3
"""Lab 01-1 · The claw map: which parts of the stack exist here, and on your Spark?

Read-only. On THIS laptop it runs the Week 26 tools for real (the NAT CLI, the OpenShell policy parser,
Docker, Ollama). On the Spark it asks, with one read-only command, which of nemoclaw / openshell / nat /
docker are on the PATH and which versions they report. In DRY mode the Spark half prints an EXAMPLE
shape instead — clearly labelled, never counted as found.

Run: .venv/bin/python week26/01_what_is_a_claw/labs/lab01_1_claw_map.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from clawkit import (NAT, OPENSHELL, OPENSHELL_PINNED, banner, laptop, laptop_models, note, ok, result,  # noqa: E402
                     sh, step, table, warn)

banner("Lab 01-1 · the claw map", "laptop tools for real · the Spark read-only (or DRY)")

step(1, "this laptop — the tools Modules 03–08 run for real")
rows = []
r = laptop([NAT, "--version"], quiet=True, show="nat --version")
nat_v = next((ln for ln in r.out.splitlines() if ln.startswith("nat")), "")
rows.append(["NAT CLI (NeMo Agent Toolkit)", nat_v or "✕ missing", "Modules 04–08: build, serve, trace, eval"])
r = laptop([OPENSHELL, "--version"], quiet=True, show="openshell --version")
os_v = r.out.strip().splitlines()[-1] if r.ok and r.out.strip() else ""
rows.append(["OpenShell CLI (policy parser)", os_v or "✕ missing", "Modules 03, 07: parse policies offline"])
docker = shutil.which("docker")
if docker:
    r = laptop([docker, "info", "--format", "{{.ServerVersion}}"], quiet=True, timeout=20,
               show="docker info --format '{{.ServerVersion}}'")
    dv = r.out.strip().splitlines()[-1] if r.ok and r.out.strip() else "client only — daemon not running"
else:
    dv = "not installed"
rows.append(["Docker", dv, "Module 05: Phoenix / OTel collector on the laptop (optional)"])
lm = [m for m in laptop_models() if "cloud" not in m]
rows.append(["Ollama (laptop stand-in model)", ", ".join(lm[:3]) or "none", "the LLM behind laptop NAT agents"])
table(rows, ["component", "found", "used for"])
if os_v and OPENSHELL_PINNED not in os_v:
    note(f"the laptop CLI is {os_v}; NemoClaw pins OpenShell {OPENSHELL_PINNED} on the Spark — the laptop one "
         "only PARSES policies, it never manages a sandbox")

step(2, "the Spark — which claw tools are on the PATH? (read-only)")
EXAMPLE_WHICH = """nemoclaw=/home/<you>/.local/bin/nemoclaw
openshell=/home/<you>/.local/bin/openshell
nat=-
docker=/usr/bin/docker
openshell 0.0.116"""
res = sh('for b in nemoclaw openshell nat docker; do printf "%s=" $b; command -v $b || echo -; done; '
         'command -v openshell >/dev/null && openshell --version', example=EXAMPLE_WHICH, timeout=60)
found = {}
for ln in res.out.splitlines():
    if "=" in ln:
        k, v = ln.split("=", 1)
        found[k.strip()] = v.strip() not in ("", "-")

step(3, "the map — one row per part of the stack")
live = res.live
mark = (lambda k: ("✓ on the Spark" if found.get(k) else "✕ not yet") if live else "◈ DRY — not checked")
table([
    ["NemoClaw CLI", "Spark host", mark("nemoclaw"), "02"],
    ["OpenShell gateway + CLI", "Spark host (Docker)", mark("openshell"), "02–03"],
    ["Sandbox + supervisor", "container on the Spark", "see 🪸 Reef → Sandboxes" if live else "◈ DRY", "03"],
    ["Harness (OpenClaw / Hermes / Deep Agents)", "inside the sandbox", "created by `nemoclaw onboard`", "02"],
    ["Inference provider", "Spark host (vLLM / Ollama)", "see 🪸 Reef → Services", "02, 04"],
    ["NAT", "Spark host · sandbox · laptop", (mark("nat") if live else "◈ DRY") + f" · laptop {'✓' if nat_v else '✕'}", "04–06"],
    ["Docker", "Spark host", mark("docker"), "02"],
], ["part", "where it runs", "status", "module"])

if not live:
    warn("DRY: the Spark column is not your machine. Connect a Spark (🖥 Spark setup) and switch to ⚡ Live.")
elif not found.get("nemoclaw"):
    note("No NemoClaw yet — that is expected before Module 02, which installs it with one command.")
else:
    ok("NemoClaw is installed — 🪸 Reef will show your sandboxes and the inference route.")
result("You now know which parts exist. Module 02 fills in the Spark column.")

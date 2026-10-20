#!/usr/bin/env python3
"""Lab 14-1 · Install NeMo Agent Toolkit, register the hotel tools, and validate the workflow.

NAT lives in its own venv (week25/.venv-nat, Python 3.12), separate from the repo .venv that runs
the labs, so this lab drives the `nat` CLI as a subprocess. It:
  1. finds the NAT CLI and prints its version,
  2. installs the course's plugin package (hotel_ops_nat) into that venv if it is not registered yet
     — an editable `uv pip install --no-deps -e`, the same thing `nat workflow create` does,
  3. asks NAT which components it can see (`nat info components`), and
  4. validates the four workflow YAML files with `nat validate` (offline: no LLM is called),
then shows the matching install on the Spark (DRY unless a Spark is connected).

Run: .venv/bin/python week25/14_nat_agents/labs/lab14_1_install_and_register.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import WEEK, banner, note, ok, result, sh, step, table, warn  # noqa: E402

MOD = Path(__file__).resolve().parents[1]                 # week25/14_nat_agents
NAT = Path(os.environ.get("NAT_BIN", WEEK / ".venv-nat" / "bin" / "nat"))
NAT_PY = NAT.parent / "python"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
NOISE = ("AuthlibDeprecationWarning", "compatible before version", "from authlib.jose")


def nat(*args: str, timeout: float = 300) -> subprocess.CompletedProcess:
    """Run the NAT CLI from the module folder, printing the command first."""
    print(f"$ {NAT.relative_to(WEEK.parent) if NAT.is_relative_to(WEEK.parent) else NAT} {' '.join(args)}")
    env = {**os.environ, "PYTHONWARNINGS": "ignore", "NO_COLOR": "1"}
    p = subprocess.run([str(NAT), *args], cwd=MOD, capture_output=True, text=True, timeout=timeout, env=env)
    p.stdout = ANSI.sub("", p.stdout)
    p.stderr = "\n".join(ln for ln in ANSI.sub("", p.stderr).splitlines() if not any(n in ln for n in NOISE))
    return p


banner("Lab 14-1 · install NAT, register the hotel tools, validate the workflow",
       "NeMo Agent Toolkit 1.9 · runs on the laptop · the Spark part is DRY until a Spark is connected")

step(1, "find the NAT CLI")
if not NAT.exists():
    warn(f"no NAT CLI at {NAT}")
    print("→ install it into a week25-local venv (Python 3.11–3.13; this course uses 3.12):")
    print("$ uv venv -p 3.12 week25/.venv-nat")
    print('$ uv pip install --python week25/.venv-nat/bin/python "nvidia-nat[langchain]~=1.9" greenlet "nvidia-nat-mcp~=1.9"')
    result("install NAT first, then run this lab again.")
    sys.exit(0)
p = nat("--version", timeout=120)
ver = (p.stdout.strip() or p.stderr.strip()).splitlines()[-1]
ok(ver)

step(2, "register the course plugin (hotel_ops_nat) — the NAT way: a package with a 'nat.components' entry point")
probe = subprocess.run([str(NAT_PY), "-c", "import importlib.metadata as m; "
                        "print([e.value for e in m.entry_points(group='nat.components') if e.name=='hotel_ops_nat'])"],
                       capture_output=True, text=True, timeout=60)
if "hotel_ops_nat.register" in probe.stdout:
    ok("hotel_ops_nat is already installed in week25/.venv-nat (entry point nat.components → hotel_ops_nat.register)")
else:
    uv = shutil.which("uv")
    cmd = ([uv, "pip", "install", "--python", str(NAT_PY), "--no-deps", "-e", str(MOD / "hotel_ops_nat")] if uv else
           [str(NAT_PY), "-m", "pip", "install", "--no-deps", "-e", str(MOD / "hotel_ops_nat")])
    print("$ " + " ".join(Path(c).name if i == 0 else c.replace(str(WEEK.parent) + "/", "") for i, c in enumerate(cmd)))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print((r.stdout + r.stderr).strip()[-600:])
    if r.returncode != 0:
        warn("install failed — see the output above")
        sys.exit(1)
    ok("installed (editable: edits to hotel_ops_nat/src/ take effect on the next `nat run`)")

step(3, "what can NAT see? `nat info components` (only the kinds this module uses)")
out_json = MOD / ".runs" / "components.json"
out_json.parent.mkdir(exist_ok=True)
p = nat("info", "components", "-o", str(out_json.relative_to(MOD)), timeout=300)
comps = json.loads(out_json.read_text())["results"] if out_json.exists() else []
want = {"function": ("room_temperature", "create_maintenance_ticket", "current_datetime", "react_agent",
                     "tool_calling_agent", "rewoo_agent", "router_agent"),
        "function_group": ("mcp_client",), "llm_provider": ("openai", "nim", "litellm"),
        "front_end": ("console", "fastapi", "mcp"), "tracing": ("file", "otelcollector", "langfuse", "phoenix"),
        "evaluator": ("langsmith", "langsmith_custom", "trajectory")}
rows = [[c["component_type"], c["component_name"], c["package"]] for c in comps
        if c["component_name"] in want.get(c["component_type"], ())]
table(sorted(rows), ["type", "name", "package"])
note(f"{len(comps)} components registered in total. The two hotel tools come from YOUR package, "
     "the agents from nvidia-nat-langchain, mcp_client from nvidia-nat-mcp.")
if not any(r[1] == "room_temperature" for r in rows):
    warn("room_temperature is not registered — rerun step 2")

step(4, "validate the workflow files (offline — NAT parses and type-checks, no model is called)")
vrows = []
for f in ("configs/hotel_agent.yml", "configs/hotel_agent_spark.yml", "configs/hotel_agent_mcp.yml",
          "configs/hotel_eval.yml"):
    p = nat("validate", "--config_file", f, timeout=300)
    good = p.returncode == 0 and "is valid" in (p.stdout + p.stderr)
    wf = re.search(r"Workflow Type: (\S+)", p.stdout)
    vrows.append([f, "✓ valid" if good else "✕ " + (p.stderr.strip().splitlines() or ["?"])[-1][:60],
                  wf.group(1) if wf else "—"])
table(vrows, ["config", "nat validate", "workflow"])

step(5, "the same install on the Spark (aarch64, DGX OS) — one venv under ~/w25")
sh("mkdir -p ~/w25 && cd ~/w25 && python3 -m venv .venv-nat && "
   ".venv-nat/bin/pip install -q 'nvidia-nat[langchain]~=1.9' greenlet 'nvidia-nat-mcp~=1.9' && "
   ".venv-nat/bin/nat --version",
   example="nat, version 1.9.0", timeout=900)
note("Then copy week25/14_nat_agents to the Spark (or git clone the repo there) and install hotel_ops_nat the same "
     "way. NAT itself is pure Python: nothing in it needs the GPU — the model server does.")

result("NAT is installed, your two tools are registered as NAT functions, and all four workflow files validate. "
       "Lab 14-2 runs the agent.")

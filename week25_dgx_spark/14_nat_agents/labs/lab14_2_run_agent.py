#!/usr/bin/env python3
"""Lab 14-2 · Run the hotel agent with `nat run`, then read its trace.

Picks the model server the same way every Week 25 agent lab does: vLLM on the Spark (:8000) if it
answers, else Ollama on the Spark (:11434), else Ollama on THIS laptop as a labelled LAPTOP STAND-IN.
It then runs configs/hotel_agent.yml (or hotel_agent_spark.yml for vLLM) through the NAT CLI, streams
the tool calls as they happen, and prints the trace NAT wrote (general.telemetry.tracing → _type: file).

Pass your own request as arguments:
  .venv/bin/python week25/14_nat_agents/labs/lab14_2_run_agent.py "Room 1510 is cold, what is going on?"

Run: .venv/bin/python week25/14_nat_agents/labs/lab14_2_run_agent.py
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import (LAPTOP_OLLAMA, WEEK, banner, laptop_models, mode, models, note, ok, pick_laptop_model,  # noqa: E402
                      result, step, table, url, warn)

MOD = Path(__file__).resolve().parents[1]
NAT = Path(os.environ.get("NAT_BIN", WEEK / ".venv-nat" / "bin" / "nat"))
ANSI = re.compile(r"\x1b\[[0-9;]*m")
QUESTION = " ".join(sys.argv[1:]) or "Room 808 feels hot. Check it and open a ticket if something is wrong."
TRACE, TICKETS = MOD / ".runs" / "nat_trace.jsonl", MOD / ".runs" / "tickets.jsonl"

# A real run captured on this Mac while the module was written — shown only when no model server answers.
CAPTURED = """→ tool_call room_temperature({"room": "808"})
  ← Room 808: 27.9 °C, setpoint 23.0 °C (+4.9 °C, too warm). HVAC status: fault: fan coil unit not responding.
→ tool_call create_maintenance_ticket({"room": "808", "issue": "HVAC fault: fan coil unit not responding", "priority": "high"})
  ← Created ticket MT-A83AD2 for room 808 (priority high): HVAC fault: fan coil unit not responding
· ANSWER  Room 808 is too warm (27.9 °C vs. 23 °C setpoint) with a fan coil unit fault. Created maintenance ticket MT-A83AD2 (high priority)."""


def pick_llm() -> tuple[str, str, str, str]:
    """(base_url, model, config file, source) — Spark vLLM → Spark Ollama → laptop stand-in → ('', '', '', 'dry')."""
    if mode() == "live":
        for kind, cfgf in (("vllm", "configs/hotel_agent_spark.yml"), ("ollama", "configs/hotel_agent.yml")):
            ids = models(url(kind))
            if ids:
                return url(kind), os.environ.get("HOTEL_LLM_MODEL") or ids[0], cfgf, f"spark-{kind}"
    if laptop_models():                     # the 💻 stand-in follows its own toggle, not DRY
        return LAPTOP_OLLAMA, pick_laptop_model(["nemotron-3-nano", "gemma4:12b"]), "configs/hotel_agent.yml", "laptop"
    return "", "", "configs/hotel_agent.yml", "dry"


def run_streaming(cmd: list[str], env: dict) -> tuple[int, str]:
    """Run `nat run`, echo only the agent's tool calls and final answer (NAT's own log is very chatty)."""
    proc = subprocess.Popen(cmd, cwd=MOD, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            errors="replace", bufsize=1)
    answer, grab, lines = [], None, []
    for raw in proc.stdout:
        line = ANSI.sub("", raw.rstrip("\n"))
        lines.append(line)
        m = re.search(r"'function': \{'name': '(\w+)', 'arguments': '([^']*)'", line)
        if "Tool's input" in line and m:
            print(f"→ tool_call {m.group(1)}({m.group(2)})", flush=True)
        elif "Tool's response" in line:
            grab = "tool"
        elif "Workflow Result:" in line:
            grab = "answer"
        elif grab == "tool" and line.strip():
            print(f"  ← {line.strip()}", flush=True)
            grab = None
        elif grab == "answer":
            if line.startswith("-----"):
                grab = None
            elif line.strip():
                answer.append(line.strip())
        elif " ERROR " in line or line.startswith("Error"):
            print(f"✕ {line.strip()[:240]}")
    proc.wait()
    print("· ANSWER  " + " ".join(answer) if answer else "⚠ no 'Workflow Result' in NAT's output")
    return proc.returncode, "\n".join(lines)


banner("Lab 14-2 · run the hotel agent with nat run", "tool_calling_agent · two fake hotel tools · file trace")
if not NAT.exists():
    warn(f"no NAT CLI at {NAT} — run lab 14-1 first (it prints the install command)")
    sys.exit(0)

step(1, "where does the model run?")
base, model, cfg_file, src = pick_llm()
where = {"spark-vllm": "vLLM on the Spark", "spark-ollama": "Ollama on the Spark",
         "laptop": "Ollama on THIS laptop — LAPTOP STAND-IN, not the Spark", "dry": "nowhere (DRY)"}[src]
print(f"◆ {where} · base_url={base or '—'} · model={model or '—'} · config={cfg_file}")

step(2, f"nat run · “{QUESTION}”")
cmd = [str(NAT), "run", "--config_file", cfg_file, "--input", QUESTION]
print(f"$ cd week25/14_nat_agents && HOTEL_LLM_BASE_URL={base or '<url>'} HOTEL_LLM_MODEL={model or '<model>'} "
      f"../.venv-nat/bin/nat run --config_file {cfg_file} --input \"{QUESTION}\"")
if src == "dry":
    print("◈ EXAMPLE — a LAPTOP STAND-IN run captured on this Mac while the module was written (not your machine):")
    print(CAPTURED)
    result("No model server answered. Start Ollama on the laptop, or connect a Spark with vLLM (Module 05), and rerun.")
    sys.exit(0)
env = {**os.environ, "PYTHONWARNINGS": "ignore", "NO_COLOR": "1", "HOTEL_LLM_BASE_URL": base, "HOTEL_LLM_MODEL": model,
       "HOTEL_TRACE_FILE": str(TRACE), "HOTEL_TICKET_LOG": str(TICKETS)}
TRACE.parent.mkdir(exist_ok=True)
code, log = run_streaming(cmd, env)
if code != 0:
    warn(f"nat run exited {code}. Last lines:\n" + "\n".join(log.splitlines()[-8:]))
    sys.exit(1)
if src == "laptop":
    note("LAPTOP STAND-IN: the agent logic, tool calls and trace are real; the speed is this laptop's, not a Spark's.")

step(3, "the trace NAT wrote — one JSON line per event (general.telemetry.tracing.local_file)")
rows, t0 = [], None
for ln in TRACE.read_text(encoding="utf-8").splitlines() if TRACE.exists() else []:
    p = json.loads(ln)["payload"]
    t0 = t0 or p["event_timestamp"]
    data = p.get("data") or {}
    io = data.get("output") if p["event_type"].endswith("_END") else data.get("input")
    rows.append([f"{p['event_timestamp'] - t0:6.2f}s", p["event_type"], p["name"], str(io or "")[:58]])
table(rows, ["t", "event", "name", "input (START) / output (END)"])
note(f"{len(rows)} events in {TRACE.relative_to(WEEK.parent)}. The gaps between tool events are the LLM calls; "
     "swap `_type: file` for `otelcollector` or `langfuse` (installed) — or `phoenix` (extra nvidia-nat[phoenix]) — to send the same events to a tracing UI.")

step(4, "side effects — the ticket log the create_maintenance_ticket tool appended to")
if TICKETS.exists():
    for ln in TICKETS.read_text(encoding="utf-8").splitlines()[-3:]:
        print(f"│ {ln}")
    ok("ticket ids are hashes of (room, issue): the same fault always gets the same id, so a repeat run is easy to spot")
else:
    note("no ticket this time — the agent judged that nothing was wrong (try room 808)")

result("One YAML file gave you an agent: NAT built the LLM client, wrapped your Python functions as tools, "
       "ran the tool-calling loop and traced every step.")

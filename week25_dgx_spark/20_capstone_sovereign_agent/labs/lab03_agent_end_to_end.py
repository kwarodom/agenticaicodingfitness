#!/usr/bin/env python3
"""Lab 20-3 · End to end: four guest messages through gateway → NAT agent → fine-tuned router → tools.

1. Starts Module 08's LiteLLM proxy with the capstone config: `agent-brain` (Spark A vLLM) and
   `hotel-router` (Spark B vLLM + LoRA), each with a laptop Ollama fallback.
2. Runs the capstone NAT workflow once per scenario (smoke · hot room · restaurant · Thai TV) with the
   gateway key in the environment only.
3. Reads NAT's trace and the ticket log, and applies the acceptance checks from _capstone.SCENARIOS.

Everything is real on a laptop (both aliases fall back to Ollama, labelled LAPTOP STAND-IN). With two
Sparks serving, the same run exercises the real fine-tune. The proxy is always stopped on exit.

Needs: week25/.venv-nat with hotel_ops_nat (Module 14) and hotel_capstone_nat (this folder) installed.
Run: .venv/bin/python week25/20_capstone_sovereign_agent/labs/lab03_agent_end_to_end.py [--only smoke]
"""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _capstone as C  # noqa: E402
from sparkkit import banner, check, note, result, step, table, warn  # noqa: E402

ONLY = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None

banner("Lab 20-3 · the sovereign hotel agent, end to end",
       "gateway → NAT agent → fine-tuned router (as a tool) → room + ticket tools")

step(1, "preflight: NAT and both plugins installed?")
if not C.NAT.exists():
    warn(f"{C.NAT} missing — uv venv -p 3.12 week25/.venv-nat && uv pip install -p week25/.venv-nat/bin/python "
         "'nvidia-nat[langchain]==1.9.0'")
    raise SystemExit(0)
info = subprocess.run([str(C.NAT), "info", "components", "-t", "function"], capture_output=True, text=True,
                      timeout=120).stdout
need = {"route_guest_message": "hotel_capstone_nat (this folder)", "room_temperature": "hotel_ops_nat (Module 14)"}
missing = [f"{k} ← {v}" for k, v in need.items() if k[:14] not in info]
if missing:
    for m in missing:
        warn(f"tool not registered: {m}")
    print("$ uv pip install --python week25/.venv-nat/bin/python --no-deps -e week25/14_nat_agents/hotel_ops_nat "
          "-e week25/20_capstone_sovereign_agent/hotel_capstone_nat")
    raise SystemExit(0)
check(True, "route_guest_message, room_temperature, create_maintenance_ticket are registered in NAT 1.9", "")

step(2, "start the gateway with the capstone config")
cfg_path = C.G.write_config(C.gateway_config(), C.RUNS / "capstone-gateway.yaml")
table([[d["model_name"], d["litellm_params"]["model"], d["litellm_params"]["api_base"]]
       for d in C.gateway_config()["model_list"]], ["alias", "backend", "api_base"])

results = []
with C.G.Proxy(cfg_path) as px:
    env = {**os.environ, "HOTEL_GATEWAY_KEY": px.key, "CAPSTONE_GATEWAY": px.base + "/v1", "CAPSTONE_BRAIN_URL": px.base + "/v1",
           "CAPSTONE_TRACE_FILE": str(C.RUNS / "capstone_trace.jsonl"),
           "CAPSTONE_TICKET_LOG": str(C.RUNS / "capstone_tickets.jsonl")}
    for sc in C.SCENARIOS:
        if ONLY and sc["id"] != ONLY:
            continue
        step(3, f"scenario '{sc['id']}': {sc['message']}")
        for f in ("capstone_trace.jsonl", "capstone_tickets.jsonl"):
            (C.RUNS / f).unlink(missing_ok=True)
        t0 = time.time()
        proc = subprocess.run([str(C.NAT), "run", "--config_file", str(C.AGENT_YML), "--input", sc["message"]],
                              cwd=C.MOD, env=env, capture_output=True, text=True, timeout=600)
        trace = C.read_trace(C.RUNS / "capstone_trace.jsonl")
        calls = C.tool_calls(trace)
        tickets = C.H.read_predictions(C.RUNS / "capstone_tickets.jsonl") if (C.RUNS / "capstone_tickets.jsonl").is_file() else []
        for name, inp, out in calls:
            print(f"→ {name}({inp[:70]})")
            print(f"  ← {out[:150]}")
        answer = C.final_answer(trace)
        if not answer:                                   # the run failed before the agent answered
            errs = [ln.split(" - ", 3)[-1] for ln in (proc.stdout + proc.stderr).splitlines() if "ERROR" in ln]
            answer = "(no answer) " + (errs[-1][:200] if errs else f"nat exited {proc.returncode}")
        print(f"· ANSWER  {answer[:300]}")
        checks = C.judge(sc, calls, tickets)
        for name, ok in checks:
            check(ok, name, name)
        results.append([sc["id"], f"{sum(ok for _, ok in checks)}/{len(checks)}", f"{time.time() - t0:.0f}s",
                        "✓" if all(ok for _, ok in checks) else "✕"])

import json  # noqa: E402
if not ONLY:                                     # a partial run must never look like a full acceptance
    (C.RUNS / "acceptance_agent.json").write_text(json.dumps(
        {"ts": time.strftime("%Y-%m-%d %H:%M"), "stand_in": not C.url("vllm", "b"),
         "scenarios": [{"id": r[0], "passed": r[1], "ok": r[3] == "✓"} for r in results]}, indent=1), encoding="utf-8")

step(4, "acceptance summary")
table(results, ["scenario", "checks passed", "time", ""])
note("LAPTOP STAND-IN unless both Sparks serve: then `hotel-router` is the fine-tune and `agent-brain` is Qwen3.6.")
passed = all(r[3] == "✓" for r in results)
result("ACCEPTED: the agent routes, reads, tickets and stays quiet exactly as specified." if passed else
       "NOT accepted yet — read the ✕ lines: a routing miss is the router's job (retrain, Module 09/13), "
       "a tool-order or ticket miss is the agent's (prompt, Module 14).")

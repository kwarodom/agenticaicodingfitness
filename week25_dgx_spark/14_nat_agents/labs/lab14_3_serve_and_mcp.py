#!/usr/bin/env python3
"""Lab 14-3 · Serve the agent: `nat serve` (REST + OpenAI-compatible) and `nat mcp serve` (tools over MCP).

Part A starts the hotel agent as a FastAPI server on :8400, checks /health, lists the routes NAT
created, and sends ONE OpenAI-style request to /v1/chat/completions — so any OpenAI client (Open WebUI,
LiteLLM, your own code) can talk to the agent as if it were a model. Part B publishes the two hotel
tools as an MCP server on :9901 and calls them with `nat mcp client` — no LLM involved, so Part B runs
even when no model server is up. Both servers are stopped at the end.

Run: .venv/bin/python week25/14_nat_agents/labs/lab14_3_serve_and_mcp.py
"""
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import (LAPTOP_OLLAMA, WEEK, banner, http_json, laptop_models, mode, models, note, ok,  # noqa: E402
                      pick_laptop_model, result, step, table, url, warn)

MOD = Path(__file__).resolve().parents[1]
NAT = Path(os.environ.get("NAT_BIN", WEEK / ".venv-nat" / "bin" / "nat"))
ANSI = re.compile(r"\x1b\[[0-9;]*m")
NOISE = ("AuthlibDeprecationWarning", "compatible before version", "from authlib.jose", " - INFO ", " - WARNING ")
REST_PORT, MCP_PORT = 8400, 9901                     # course choice: :8000 is vLLM's port on the Spark
RUNS = MOD / ".runs"


def pick_llm() -> tuple[str, str, str, str]:
    if mode() == "live":
        for kind, cfgf in (("vllm", "configs/hotel_agent_spark.yml"), ("ollama", "configs/hotel_agent.yml")):
            ids = models(url(kind))
            if ids:
                return url(kind), os.environ.get("HOTEL_LLM_MODEL") or ids[0], cfgf, f"spark-{kind}"
    if laptop_models():                     # the 💻 stand-in follows its own toggle, not DRY
        return LAPTOP_OLLAMA, pick_laptop_model(["nemotron-3-nano", "gemma4:12b"]), "configs/hotel_agent.yml", "laptop"
    return "", "", "configs/hotel_agent.yml", "dry"


def start(args: list[str], log: Path, env: dict) -> subprocess.Popen:
    print("$ ../.venv-nat/bin/nat " + " ".join(args) + "   (background, log → " + str(log.relative_to(MOD)) + ")")
    return subprocess.Popen([str(NAT), *args], cwd=MOD, env=env, stdout=log.open("w"), stderr=subprocess.STDOUT,
                            start_new_session=True)


def stop(proc: subprocess.Popen, what: str) -> None:
    if proc and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        print(f"◆ stopped {what}")


def wait_http(u: str, proc: subprocess.Popen, secs: int = 240) -> bool:
    t0 = time.time()
    while time.time() - t0 < secs and proc.poll() is None:
        try:
            with urlopen(u, timeout=2):                          # noqa: S310 — our own local server
                return True
        except Exception as e:                                    # 404/405/406 still means "listening"
            if getattr(e, "code", None) in (404, 405, 406):
                return True
        time.sleep(1)
    return False


def nat_cli(*args: str) -> str:
    print("$ ../.venv-nat/bin/nat " + " ".join(args))
    p = subprocess.run([str(NAT), *args], cwd=MOD, capture_output=True, text=True, timeout=300, env=ENV)
    out = ANSI.sub("", p.stdout + p.stderr)
    return "\n".join(ln for ln in out.splitlines() if ln.strip() and not any(n in ln for n in NOISE))


banner("Lab 14-3 · serve the agent over REST and its tools over MCP",
       f"nat serve :{REST_PORT} · nat mcp serve :{MCP_PORT} · both stopped at the end")
if not NAT.exists():
    warn(f"no NAT CLI at {NAT} — run lab 14-1 first")
    sys.exit(0)
RUNS.mkdir(exist_ok=True)
base, model, cfg_file, src = pick_llm()
ENV = {**os.environ, "PYTHONWARNINGS": "ignore", "NO_COLOR": "1", "HOTEL_LLM_BASE_URL": base or LAPTOP_OLLAMA,
       "HOTEL_LLM_MODEL": model or "nemotron-3-nano:latest", "HOTEL_TRACE_FILE": str(RUNS / "serve_trace.jsonl"),
       "HOTEL_TICKET_LOG": str(RUNS / "tickets.jsonl")}

rest = mcp = None
try:
    step(1, f"nat serve — the agent as a web service on 127.0.0.1:{REST_PORT}")
    rest = start(["serve", "--config_file", cfg_file, "--host", "127.0.0.1", "--port", str(REST_PORT)],
                 RUNS / "serve.log", ENV)
    if not wait_http(f"http://127.0.0.1:{REST_PORT}/health", rest):
        warn("nat serve did not come up. Last log lines:\n" + "\n".join((RUNS / "serve.log").read_text().splitlines()[-6:]))
        sys.exit(1)
    print("→ GET /health → " + json.dumps(http_json("GET", f"http://127.0.0.1:{REST_PORT}/health", timeout=10)))
    paths = http_json("GET", f"http://127.0.0.1:{REST_PORT}/openapi.json", timeout=10)["paths"]
    keep = ("/generate", "/generate/stream", "/v1/chat/completions", "/chat", "/v1/workflow", "/evaluate/item", "/health")
    table([[p, ", ".join(m.upper() for m in paths[p])] for p in keep if p in paths], ["route NAT created", "methods"])
    note(f"{len(paths)} routes in total — see http://127.0.0.1:{REST_PORT}/docs while it runs.")

    step(2, "call it like a model: POST /v1/chat/completions (one request)")
    if src == "dry":
        print("◈ no model server answers (DRY) — skipping the call. Start Ollama or a Spark endpoint and rerun.")
    else:
        body = {"model": "hotel-agent", "messages": [{"role": "user", "content": "Room 1510 guest says it is cold. Check it."}]}
        print(f"→ POST http://127.0.0.1:{REST_PORT}/v1/chat/completions  {json.dumps(body)}")
        t0 = time.perf_counter()
        req = Request(f"http://127.0.0.1:{REST_PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=600) as r:                     # noqa: S310
            d = json.loads(r.read())
        print("· ANSWER  " + d["choices"][0]["message"]["content"])
        label = "LAPTOP STAND-IN (not Spark numbers)" if src == "laptop" else f"LIVE · {src}"
        print(f"◆ {label} · object={d['object']} · finish_reason={d['choices'][0]['finish_reason']} · "
              f"{time.perf_counter() - t0:.1f}s for the whole agent loop (LLM calls + tool calls)")
        ok("an OpenAI-shaped response: point Open WebUI or LiteLLM (Module 08) at this URL and the agent looks like a model")
    stop(rest, f"nat serve :{REST_PORT}")

    step(3, f"nat mcp serve — publish ONLY the two hotel tools on :{MCP_PORT} (streamable-http, path /mcp)")
    mcp = start(["mcp", "serve", "--config_file", "configs/hotel_agent.yml", "--name", "hotel-ops", "--port", str(MCP_PORT),
                 "--tool_names", "room_temperature", "--tool_names", "create_maintenance_ticket"], RUNS / "mcp.log", ENV)
    if not wait_http(f"http://localhost:{MCP_PORT}/mcp", mcp):
        warn("nat mcp serve did not come up — is nvidia-nat-mcp installed? (lab 14-1 prints the install line)")
        sys.exit(1)

    step(4, "any MCP client can list and call them — here NAT's own client, no LLM involved")
    print(nat_cli("mcp", "client", "tool", "list", "--url", f"http://localhost:{MCP_PORT}/mcp"))
    print(nat_cli("mcp", "client", "tool", "call", "room_temperature", "--url", f"http://localhost:{MCP_PORT}/mcp",
                  "--json-args", '{"room": "808"}'))
    print(nat_cli("mcp", "client", "tool", "call", "create_maintenance_ticket", "--url", f"http://localhost:{MCP_PORT}/mcp",
                  "--json-args", '{"room": "808", "issue": "fan coil unit not responding", "priority": "high"}'))
    note("configs/hotel_agent_mcp.yml is the other side: `function_groups: {_type: mcp_client}` turns every tool this "
         "server lists into an agent tool. Any other MCP client (Claude Desktop, Cursor, an IDE) can connect the same way.")
finally:
    stop(rest, f"nat serve :{REST_PORT}")
    stop(mcp, f"nat mcp serve :{MCP_PORT}")

result("One workflow file, three front ends: `nat run` (console), `nat serve` (REST + OpenAI-compatible) "
       "and `nat mcp serve` (tools for any MCP client).")

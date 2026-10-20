#!/usr/bin/env python3
"""Interactive tutorial GUIDE — a clickable, debuggable web runner for the
HR-onboarding long-running ADK agent (see TUTORIAL.md).

A small CONTROL PLANE that:
  • serves the guide UI (static/guide.html);
  • manages the long-running onboarding service (server.py, port 8077) as a child
    process — start / stop / restart / health / log tail;
  • runs each tutorial step as a subprocess and STREAMS its output to the browser
    live (so slow LLM steps never "time out" in the UI);
  • cleans up / demolishes generated artifacts so you can start over.

Launch it and open the printed URL (auto-picks a free port if 8070 is taken):

    .venv/bin/python week17/hr_onboarding/tutorial_server.py
    # → http://127.0.0.1:8070
"""
from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

PKG = Path(__file__).resolve().parent                 # …/week17/hr_onboarding
ROOT = PKG.parents[1]                                 # …/agenticaicodingfitness
PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable
LOG_DIR = PKG / ".logs"
LOG_DIR.mkdir(exist_ok=True)


def _port_busy(port: int) -> bool:
    """CONNECT-test (not bind) so it also detects a server bound to 0.0.0.0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _pick_free_port(preferred: int, span: int = 40) -> int:
    for p in range(preferred, preferred + span):
        if not _port_busy(p):
            return p
    return preferred


HR_PORT = int(os.environ.get("ONBOARDING_PORT", "8077"))
GUIDE_PORT = int(os.environ.get("ONBOARDING_GUIDE_PORT", "8070"))
HR_BASE = f"http://127.0.0.1:{HR_PORT}"

# ── the snippets the lifecycle steps run ────────────────────────────────────
S_CREATE = f"""import httpx, uuid
sid = "onb-" + uuid.uuid4().hex[:8]
open("/tmp/hr_guide_sid.txt", "w").write(sid)
B = "{HR_BASE}"
r = httpx.post(f"{{B}}/onboard", json={{"session_id": sid}}, timeout=60).json()
print("session:", sid)
print("current_step:", r["current_step"], "(durable session created at START)")"""

S_KICK = f"""import httpx
sid = open("/tmp/hr_guide_sid.txt").read().strip()
B = "{HR_BASE}"
r = httpx.post(f"{{B}}/chat", json={{"session_id": sid,
    "message": "Start onboarding for Jane Doe, email jane@example.com, start date 2026-07-01."}},
    timeout=300).json()
print("agent:", (r.get("reply") or "").strip()[:280])
print("step:", r["current_step"], "| waiting_for:", r.get("paused_waiting_for"),
      "| pending:", r.get("pending_signals"))
print("details:", r.get("new_hire_details"))"""

S_SKIP = f"""import httpx
sid = open("/tmp/hr_guide_sid.txt").read().strip()
B = "{HR_BASE}"
r = httpx.post(f"{{B}}/chat", json={{"session_id": sid,
    "message": "Can we skip the signature and provision IT accounts right now?"}},
    timeout=300).json()
print("agent:", (r.get("reply") or "").strip()[:280])
print("step:", r["current_step"], "(should STILL be WELCOME_SENT — the agent refuses to skip)")"""

S_STATUS = f"""import httpx, json
sid = open("/tmp/hr_guide_sid.txt").read().strip()
r = httpx.get("{HR_BASE}/status/hr_coordinator/" + sid).json()
print(json.dumps(r, indent=2))
print(">>> this read the durable checkpoint with NO LLM call.")"""

S_DOCSIGNED = f"""import httpx
sid = open("/tmp/hr_guide_sid.txt").read().strip()
B = "{HR_BASE}"
r = httpx.post(f"{{B}}/webhooks/document_signed", json={{"session_id": sid}}, timeout=300).json()
print("agent:", (r.get("reply") or "").strip()[:280])
print("step:", r["current_step"], "(coordinator delegated to it_agent → IT_PROVISIONED)")
print("details:", r.get("new_hire_details"))"""

S_TRACK = f"""import httpx
sid = open("/tmp/hr_guide_sid.txt").read().strip()
B = "{HR_BASE}"
r = httpx.post(f"{{B}}/chat", json={{"session_id": sid,
    "message": "The laptop shipped, tracking id 1Z999AA10123456784."}}, timeout=300).json()
print("agent:", (r.get("reply") or "").strip()[:280])
print("step:", r["current_step"], "| pending:", r.get("pending_signals"),
      "(recorded tracking, now paused for delivery)")"""

S_DELIVERED = f"""import httpx
sid = open("/tmp/hr_guide_sid.txt").read().strip()
B = "{HR_BASE}"
r = httpx.post(f"{{B}}/webhooks/hardware_delivered", json={{"session_id": sid}}, timeout=300).json()
print("agent:", (r.get("reply") or "").strip()[:280])
print("step:", r["current_step"], "(day-one schedule sent → onboarding COMPLETE)")"""

S_FINAL = f"""import httpx, json
sid = open("/tmp/hr_guide_sid.txt").read().strip()
r = httpx.get("{HR_BASE}/status/hr_coordinator/" + sid).json()
print(json.dumps(r, indent=2))
print("complete?", r.get("complete"))"""

S_DUR_PAUSE = f"""import httpx, uuid
sid = "onb-" + uuid.uuid4().hex[:8]
open("/tmp/hr_guide_dur.txt", "w").write(sid)
B = "{HR_BASE}"
print("session:", sid)
httpx.post(f"{{B}}/onboard", json={{"session_id": sid}}, timeout=60)
r = httpx.post(f"{{B}}/chat", json={{"session_id": sid,
    "message": "Start onboarding for Sam Lee, email sam@example.com, start date 2026-08-01."}},
    timeout=300).json()
print("agent:", (r.get("reply") or "").strip()[:200])
print("current_step =", r["current_step"], "(paused, persisted to SQLite)")
print(">>> now RESTART the service (next step), then run the resume step.")"""

S_DUR_RESUME = f"""import httpx
sid = open("/tmp/hr_guide_dur.txt").read().strip()
B = "{HR_BASE}"
print("session:", sid)
print("after restart, step =",
      httpx.get(f"{{B}}/status/hr_coordinator/{{sid}}").json()["current_step"], "(survived!)")
r = httpx.post(f"{{B}}/webhooks/document_signed", json={{"session_id": sid}}, timeout=300).json()
print("resumed → step =", r["current_step"], "(woke from a cold restart, delegated to it_agent)")"""

STEPS = [
    # ---- Understand the code ----
    {"id":"s0","group":"Understand the agent","title":"Step 0 — Setup check",
     "desc":"This is a REAL ADK agent — it needs google-adk + a model. Check they import.",
     "kind":"run","cwd":str(ROOT),
     "cmd":[PY,"-c","import google.adk, litellm; print('google-adk', getattr(google.adk,'__version__','present'), '+ litellm ✓')"]},
    {"id":"s1","group":"Understand the agent","title":"Step 1 — The durable state machine",
     "desc":"Behaviour is driven by an explicit current_step in persisted state — NOT by replaying "
            "chat history. Two of the steps are PAUSE gates waiting on an external signal.",
     "kind":"run","cwd":str(PKG),
     "cmd":["bash","-c","grep -n -A 18 'class OnboardingStep' onboarding_steps.py"]},
    {"id":"s2","group":"Understand the agent","title":"Step 2 — Agent, tools & sub-agent",
     "desc":"The coordinator's tools each advance the state machine atomically; it delegates IT "
            "provisioning to a focused it_agent; sessions persist via DatabaseSessionService.",
     "kind":"run","cwd":str(PKG),
     "cmd":["bash","-c","grep -n 'def build_root_agent' -A 16 agent.py"]},
    {"id":"s3","group":"Understand the agent","title":"Step 3 — The resume handler",
     "desc":"A paused onboarding is woken by a webhook that applies a state_delta BEFORE the next "
            "inference, so the model wakes already seeing the new step (can't hallucinate progress).",
     "kind":"run","cwd":str(PKG),
     "cmd":["bash","-c","grep -n 'async def document_signed' -A 10 resume_handler.py"]},

    # ---- Run the long-running service ----
    {"id":"srv","group":"Run the long-running service","title":"Step 4 — Start the onboarding service",
     "desc":"One persistent FastAPI service holding NO state in memory — every session lives in the "
            "durable SQLite store. It can be restarted and resumed days later (port 8077).",
     "kind":"server","action":"start","target":"hr"},

    # ---- Drive one onboarding across 'days' ----
    {"id":"s5","group":"Drive one onboarding","title":"Step 5 — Create an onboarding session",
     "desc":"Create a durable session seeded at START (no LLM call).",
     "kind":"run","cwd":str(PKG),"requires":["hr"],"cmd":[PY,"-c",S_CREATE]},
    {"id":"s6","group":"Drive one onboarding","title":"Step 6 — Send the welcome packet (pause)",
     "desc":"Give the coordinator the new-hire details. It calls send_welcome_packet and PAUSES at "
            "WELCOME_SENT, waiting for the signed contract. (LLM.)",
     "kind":"run","cwd":str(PKG),"requires":["hr"],"cmd":[PY,"-c",S_KICK]},
    {"id":"s7","group":"Drive one onboarding","title":"Step 7 — Try to skip the wait",
     "desc":"Ask it to jump ahead. It must REFUSE — the idle-time safety gate. State stays "
            "WELCOME_SENT. (LLM.)",
     "kind":"run","cwd":str(PKG),"requires":["hr"],"cmd":[PY,"-c",S_SKIP]},
    {"id":"s8","group":"Drive one onboarding","title":"Step 8 — Inspect the durable state",
     "desc":"Read the checkpoint directly. This proves progress lives in the session store, not the "
            "conversation — and needs no model call.",
     "kind":"run","cwd":str(PKG),"requires":["hr"],"cmd":[PY,"-c",S_STATUS]},
    {"id":"s9","group":"Drive one onboarding","title":"Step 9 — Webhook: document signed",
     "desc":"…days pass… the contract is signed. The webhook resumes the agent with a state_delta; "
            "the coordinator delegates to it_agent, which provisions accounts → IT_PROVISIONED. (LLM.)",
     "kind":"run","cwd":str(PKG),"requires":["hr"],"cmd":[PY,"-c",S_DOCSIGNED]},
    {"id":"s10","group":"Drive one onboarding","title":"Step 10 — Provide hardware tracking",
     "desc":"Give the laptop's tracking id. The agent records it and PAUSES again, waiting for "
            "delivery. (LLM.)",
     "kind":"run","cwd":str(PKG),"requires":["hr"],"cmd":[PY,"-c",S_TRACK]},
    {"id":"s11","group":"Drive one onboarding","title":"Step 11 — Webhook: hardware delivered",
     "desc":"…days pass… the laptop arrives. The webhook resumes the agent, which sends the day-one "
            "schedule and COMPLETES the onboarding. (LLM.)",
     "kind":"run","cwd":str(PKG),"requires":["hr"],"cmd":[PY,"-c",S_DELIVERED]},
    {"id":"s12","group":"Drive one onboarding","title":"Step 12 — Final status",
     "desc":"Read the final checkpoint — COMPLETED, with the full new-hire record.",
     "kind":"run","cwd":str(PKG),"requires":["hr"],"cmd":[PY,"-c",S_FINAL]},

    # ---- Durability ----
    {"id":"s13a","group":"Prove durability","title":"Step 13a — Pause a new onboarding",
     "desc":"Create + kick off a NEW onboarding and stop at WELCOME_SENT (the session is now "
            "persisted to SQLite). (LLM.)",
     "kind":"run","cwd":str(PKG),"requires":["hr"],"cmd":[PY,"-c",S_DUR_PAUSE]},
    {"id":"s13restart","group":"Prove durability","title":"Step 13b — Restart the service",
     "desc":"Restart the onboarding service. In-memory state is destroyed; the SQLite session is "
            "not. This simulates the service being scaled to zero over a quiet weekend.",
     "kind":"server","action":"restart","target":"hr"},
    {"id":"s13c","group":"Prove durability","title":"Step 13c — Resume after the restart",
     "desc":"Check the SAME session after the cold restart — it's still at WELCOME_SENT — then fire "
            "the document_signed webhook to prove it resumes from disk. (LLM.)",
     "kind":"run","cwd":str(PKG),"requires":["hr"],"cmd":[PY,"-c",S_DUR_RESUME]},
    {"id":"vis","group":"Prove durability","title":"Open the live visualizer",
     "desc":"A real-time view of the state machine, sub-agent delegation, the new-hire record, and "
            "an event timeline — drive the whole onboarding with buttons. (Start the service first.)",
     "kind":"link","href":HR_BASE + "/"},

    # ---- Cleanup ----
    {"id":"cstop","group":"Cleanup & teardown","title":"Stop the service",
     "desc":"Terminate the onboarding service process.",
     "kind":"server","action":"stop","target":"all"},
    {"id":"cclean","group":"Cleanup & teardown","title":"Clean generated files",
     "desc":"Delete the SQLite session store and __pycache__.",
     "kind":"run","cwd":str(PKG),
     "cmd":["bash","-c","rm -f onboarding_sessions.db; rm -rf __pycache__; echo 'cleaned: sessions db + pycache'"]},
    {"id":"cdemo","group":"Cleanup & teardown","title":"Start over (demolish)",
     "desc":"Stop the service AND delete generated files — a clean slate to run from Step 0.",
     "kind":"demolish"},
]
STEP_BY_ID = {s["id"]: s for s in STEPS}


class Server:
    def __init__(self, name: str, script: str, port: int, env: dict):
        self.name, self.script, self.port, self.env = name, script, port, env
        self.proc: subprocess.Popen | None = None
        self.log = LOG_DIR / f"{name}.log"

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    async def healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=1.5) as c:
                return (await c.get(f"http://127.0.0.1:{self.port}/healthz")).status_code == 200
        except Exception:
            return False

    def start(self) -> None:
        if self.running():
            return
        fh = open(self.log, "w")
        self.proc = subprocess.Popen([PY, str(PKG / self.script)], cwd=str(PKG),
                                     env={**os.environ, **self.env},
                                     stdout=fh, stderr=subprocess.STDOUT)

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def tail(self, n: int = 80) -> str:
        if not self.log.exists():
            return "(no log yet)"
        return "\n".join(self.log.read_text(errors="replace").splitlines()[-n:])


SERVERS = {"hr": Server("hr", "server.py", HR_PORT, {"ONBOARDING_PORT": str(HR_PORT)})}


async def _wait_healthy(srv: Server, timeout: float = 40) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if await srv.healthy():
            return True
        await asyncio.sleep(0.5)
    return False


def _pretty_cmd(cmd) -> str:
    if cmd is None:
        return ""
    if isinstance(cmd, str):
        return cmd
    if len(cmd) >= 3 and cmd[1] == "-c":
        return f"{Path(cmd[0]).name} -c '<python snippet>'"
    if len(cmd) >= 3 and cmd[0] == "bash" and cmd[1] == "-c":
        return cmd[2]
    return " ".join(Path(cmd[0]).name if i == 0 else x for i, x in enumerate(cmd))


app = FastAPI(title="HR onboarding — tutorial guide")
_run_lock = asyncio.Lock()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(PKG / "static" / "guide.html")


@app.get("/api/steps")
async def steps() -> dict:
    def public(s):
        return {k: s[k] for k in ("id", "group", "title", "desc", "kind") if k in s} | \
               {"command": _pretty_cmd(s.get("cmd")), "href": s.get("href"),
                "requires": s.get("requires", []),
                "action": s.get("action"), "target": s.get("target")}
    return {"steps": [public(s) for s in STEPS]}


@app.get("/api/servers")
async def servers() -> dict:
    return {name: {"running": srv.running(), "healthy": await srv.healthy(), "port": srv.port}
            for name, srv in SERVERS.items()}


@app.get("/api/logs/{which}")
async def logs(which: str, n: int = 80) -> dict:
    srv = SERVERS.get(which if which in SERVERS else "hr")
    return {"log": srv.tail(n)}


class ServerAction(BaseModel):
    target: str
    action: str


@app.post("/api/server")
async def server_action(req: ServerAction) -> dict:
    targets = list(SERVERS) if req.target == "all" else [req.target]
    msgs = []
    for t in targets:
        srv = SERVERS[t]
        if req.action in ("stop", "restart"):
            srv.stop(); msgs.append(f"{t}: stopped")
        if req.action in ("start", "restart"):
            srv.start()
            ok = await _wait_healthy(srv)
            msgs.append(f"{t}: {'healthy' if ok else 'started (not healthy yet — check the log)'}")
    return {"messages": msgs, "servers": await servers()}


def _deadline_left(start: float, timeout: float) -> float:
    return start + timeout - time.time()


def _stream_lines(cmd, cwd: str, env: dict, timeout: float):
    async def gen():
        start = time.time()
        merged = {**os.environ, **(env or {})}
        if isinstance(cmd, list):
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=cwd, env=merged,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        else:
            proc = await asyncio.create_subprocess_shell(
                cmd, cwd=cwd, env=merged,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            while True:
                try:
                    line = await asyncio.wait_for(proc.stdout.readline(),
                                                  timeout=max(1, _deadline_left(start, timeout)))
                except asyncio.TimeoutError:
                    proc.kill()
                    yield f"\n⏱️  step exceeded {timeout:.0f}s — killed.\n__EXIT__ 124 {time.time()-start:.1f}\n"
                    return
                if not line:
                    break
                yield line.decode(errors="replace")
            await proc.wait()
            yield f"__EXIT__ {proc.returncode} {time.time()-start:.1f}\n"
        finally:
            if proc.returncode is None:
                proc.kill()
    return gen()


class RunRequest(BaseModel):
    step_id: str


@app.post("/api/run")
async def run_step(req: RunRequest):
    step = STEP_BY_ID.get(req.step_id)
    if step is None or step.get("kind") != "run":
        async def err():
            yield f"step {req.step_id!r} is not runnable\n__EXIT__ 1 0\n"
        return StreamingResponse(err(), media_type="text/plain")

    missing = [r for r in step.get("requires", []) if not await SERVERS[r].healthy()]
    if missing:
        async def need():
            yield (f"⚠️  this step needs the onboarding service running.\n"
                   f"   Start it (Step 4, or the top-bar control), then retry.\n__EXIT__ 1 0\n")
        return StreamingResponse(need(), media_type="text/plain")

    timeout = 360.0 if req.step_id in ("s6", "s7", "s9", "s10", "s11", "s13a", "s13c") else 90.0

    async def body():
        if _run_lock.locked():
            yield "⚠️  another step is already running — wait for it to finish.\n__EXIT__ 1 0\n"
            return
        async with _run_lock:
            yield f"$ (cwd={Path(step['cwd']).name}) {_pretty_cmd(step['cmd'])}\n\n"
            async for chunk in _stream_lines(step["cmd"], step["cwd"], step.get("env", {}), timeout):
                yield chunk
    return StreamingResponse(body(), media_type="text/plain")


@app.post("/api/cleanup")
async def cleanup() -> dict:
    for srv in SERVERS.values():
        srv.stop()
    removed = []
    db = PKG / "onboarding_sessions.db"
    if db.exists():
        db.unlink(); removed.append("onboarding_sessions.db")
    pyc = PKG / "__pycache__"
    if pyc.exists():
        for c in pyc.glob("*"):
            c.unlink()
        pyc.rmdir(); removed.append("__pycache__")
    return {"messages": ["stopped the service", f"removed: {removed or 'nothing'}"],
            "servers": await servers()}


@app.on_event("shutdown")
async def _shutdown():
    for srv in SERVERS.values():
        srv.stop()


if __name__ == "__main__":
    import uvicorn
    port = _pick_free_port(GUIDE_PORT)
    banner = ["", "  🧑‍💼  HR onboarding — interactive tutorial guide"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠  port {GUIDE_PORT} is already in use —",
                   f"         using {port} instead (set ONBOARDING_GUIDE_PORT to choose)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)

#!/usr/bin/env python3
"""Interactive tutorial GUIDE — a clickable, debuggable web runner for the whole
auth.md × ADK walkthrough (TUTORIAL.md, Steps 0–15 + cleanup).

This is a small CONTROL PLANE that:
  • serves the guide UI (static/guide.html);
  • manages the two demo servers (Part A app_server, Part B agent_server) as
    child processes — start / stop / restart / health / log tail;
  • runs each tutorial step as a subprocess and STREAMS its stdout+stderr to the
    browser live (so slow LLM steps never "time out" in the UI);
  • cleans up / demolishes generated artifacts so you can start over.

It depends only on the OFFLINE subset (fastapi/uvicorn/httpx) so it can guide you
even before google-adk is installed. Launch it and open the printed URL:

    .venv/bin/python week17/authmd_adk/tutorial_server.py
    # → http://127.0.0.1:8080
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

PKG = Path(__file__).resolve().parent                 # …/week17/authmd_adk
ROOT = PKG.parents[1]                                 # …/agenticaicodingfitness
PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable
LOG_DIR = PKG / ".logs"
LOG_DIR.mkdir(exist_ok=True)

def _port_busy(port: int) -> bool:
    """True if something is already accepting connections on 127.0.0.1:port.
    Uses a CONNECT test (not a bind test) so it also detects a server bound to
    0.0.0.0 — e.g. another project's `uvicorn --host 0.0.0.0 --port 8080`."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _pick_free_port(preferred: int, span: int = 40) -> int:
    """Return `preferred` if free, else the next free port above it."""
    for p in range(preferred, preferred + span):
        if not _port_busy(p):
            return p
    return preferred


# Ports are env-overridable; the guide auto-skips a busy port at startup.
APP_PORT = int(os.environ.get("AUTHMD_APP_PORT", "8088"))
AGENT_PORT = int(os.environ.get("AUTHMD_AGENT_PORT", "8089"))
GUIDE_PORT = int(os.environ.get("AUTHMD_GUIDE_PORT", "8080"))
OFFLINE_PORT = int(os.environ.get("AUTHMD_OFFLINE_PORT", "8090"))
APP_BASE = f"http://127.0.0.1:{APP_PORT}"
AGENT_BASE = f"http://127.0.0.1:{AGENT_PORT}"

# ── the snippets steps 4–9 run (verified working) ───────────────────────────
S_VERIFIED = """import sys; sys.path.insert(0, ".")
import idjag_provider, httpx
id_jag = idjag_provider.mint_id_jag(subject="energy-ops-bot",
    email="ops@altotech.ai", audience="%s/")
print("ID-JAG (first 40):", id_jag[:40], "...")
r = httpx.post("%s/agent/auth", json={"type":"identity_assertion",
    "id_jag": id_jag, "scopes":["sites.read"]})
print("token response:", r.json())
tok = r.json()["access_token"]
print("GET /sites/site-bkk-01/energy ->",
      httpx.get("%s/sites/site-bkk-01/energy",
                headers={"Authorization": f"Bearer {tok}"}).json())
""" % (APP_BASE, APP_BASE, APP_BASE)

S_CLAIMED = """import httpx
B = "%s"
start = httpx.post(f"{B}/agent/auth", json={"type":"verified_email",
    "email":"facility.manager@altotech.ai", "scopes":["control.write"]}).json()
print("1) started (no credential yet):", start)
otp = httpx.get(f"{B}/_demo/inbox/facility.manager@altotech.ai").json()["otp"]
print("2) human reads OTP from email:", otp)
done = httpx.post(f"{B}/agent/auth/claim/complete",
    json={"claim_token": start["claim_token"], "otp": otp}).json()
print("3) claim complete -> credential issued:", done)
""" % APP_BASE

S_LEASTPRIV = """import sys; sys.path.insert(0, ".")
import idjag_provider, httpx
B = "%s"
jag = idjag_provider.mint_id_jag(subject="x", email="ops@altotech.ai", audience=f"{B}/")
tok = httpx.post(f"{B}/agent/auth", json={"type":"identity_assertion",
    "id_jag": jag, "scopes":["sites.read"]}).json()["access_token"]
r = httpx.post(f"{B}/sites/site-bkk-01/setpoint", json={"setpoint_c":25.5},
               headers={"Authorization": f"Bearer {tok}"})
print("sites.read token on POST /setpoint ->", r.status_code, r.json())
print("=> a read credential can NEVER write (least privilege).")
""" % APP_BASE

S_REVOKE = """import sys; sys.path.insert(0, ".")
from authmd_client import AuthMdClient, AuthGrant
import httpx
B = "%s"
c = AuthMdClient(B)
cred = c.acquire(AuthGrant("altotech_read","agent_verified",["sites.read"],
                           subject="x", email="ops@altotech.ai"))
print("revoke ->", c.revoke(token=cred.token))
print("call after revoke ->",
      httpx.get(f"{B}/sites/site-bkk-01/energy",
                headers={"Authorization": cred.bearer()}).status_code, "(401 expected)")
""" % APP_BASE

S_REMINT = """import sys; sys.path.insert(0, ".")
from authmd_client import AuthMdClient, AuthGrant
grant = AuthGrant(service="altotech_read", flow="agent_verified",
                  scopes=["sites.read"], subject="energy-ops-bot", email="ops@altotech.ai")
print("DURABLE grant (lives in session state, NO token):")
print("   ", grant.as_state())
c = AuthMdClient("%s")
t1 = c.acquire(grant).token   # wake #1
t2 = c.acquire(grant).token   # wake #2, days later
print("token @ wake 1:", t1[:26], "...")
print("token @ wake 2:", t2[:26], "...")
print("different? ->", t1 != t2, " (an expired token is never a problem)")
""" % APP_BASE

S_DISCOVERY = """import httpx, json
B = "%s"
print("# auth.md\\n" + httpx.get(f"{B}/auth.md").text)
print("# PRM (RFC 9728):", json.dumps(httpx.get(f"{B}/.well-known/oauth-protected-resource").json(), indent=2))
am = httpx.get(f"{B}/.well-known/oauth-authorization-server").json()["agent_auth"]
print("# agent_auth block:", json.dumps(am, indent=2))
r = httpx.get(f"{B}/sites/site-bkk-01/energy")
print("# 401 discovery hint:", r.status_code, "->", r.headers.get("WWW-Authenticate"))
""" % APP_BASE

S_AUDIT = f"""import httpx, json
print(json.dumps(httpx.get("{APP_BASE}/_demo/audit").json(), indent=2))"""

S_DUR_PAUSE = f"""import httpx, uuid
sid = "wo-" + uuid.uuid4().hex[:8]
open("/tmp/authmd_guide_sid.txt","w").write(sid)
A = "{AGENT_BASE}"
print("session:", sid)
httpx.post(f"{{A}}/work_order", json={{"session_id": sid}}, timeout=180)
r = httpx.post(f"{{A}}/chat", json={{"session_id": sid,
    "message":"Begin the energy work order for this site."}}, timeout=300).json()
print("agent:", (r.get("reply") or "").strip()[:200])
print("current_step =", r["current_step"], "(paused, persisted to SQLite)")
print(">>> now RESTART the agent server, then run the resume step.")"""

S_DUR_RESUME = f"""import httpx
sid = open("/tmp/authmd_guide_sid.txt").read().strip()
A, B = "{AGENT_BASE}", "{APP_BASE}"
print("session:", sid)
print("after restart, step =",
      httpx.get(f"{{A}}/status/energy_ops/{{sid}}").json()["current_step"], "(survived!)")
httpx.post(f"{{A}}/webhooks/request_approval", json={{"session_id": sid}}, timeout=300)
otp = httpx.get(f"{{B}}/_demo/inbox/facility.manager@altotech.ai").json()["otp"]
print("human OTP:", otp)
r = httpx.post(f"{{A}}/webhooks/approved", json={{"session_id": sid, "otp": otp}}, timeout=300).json()
print("final step =", r["current_step"], "(woke from cold restart, re-minted token, applied)")"""

# ── the step catalog (id, group, title, what it teaches, how to run it) ──────
# kind: "run" (stream a subprocess) · "server" (start/stop) · "link" (info only)
STEPS = [
    # ---- Part 1: the auth.md protocol, offline ----
    {"id":"s0","group":"Part 1 · auth.md protocol (offline)","title":"Step 0 — Setup check",
     "desc":"Verify the offline dependencies import (no model needed for Part 1).",
     "kind":"run","cwd":str(ROOT),
     "cmd":[PY,"-c","import fastapi,uvicorn,httpx,pydantic,jwt,cryptography;print('offline deps ready ✓')"]},
    {"id":"s1","group":"Part 1 · auth.md protocol (offline)","title":"Step 1 — Watch it all run once",
     "desc":"The whole protocol end to end in one offline script: discovery → verified token → "
            "wake-time re-mint → OTP gate → least privilege (403) → apply → revoke (401) → audit.",
     "kind":"run","cwd":str(ROOT),"env":{"AUTHMD_APP_PORT":str(OFFLINE_PORT)},
     "cmd":[PY,"week17/authmd_adk/run_authmd_demo.py"]},
    {"id":"srvA","group":"Part 1 · auth.md protocol (offline)","title":"Start the app server (Part A)",
     "desc":"Steps 2–9 poke the live Energy API. Start it (port 8088). It serves auth.md, the "
            "discovery docs, /agent/auth, and the protected resource.",
     "kind":"server","action":"start","target":"app"},
    {"id":"s2","group":"Part 1 · auth.md protocol (offline)","title":"Step 2 — Discovery + the 401 hint",
     "desc":"How an agent FINDS how to authenticate: auth.md → PRM (RFC 9728) → AS metadata "
            "(agent_auth block). And the 401 on the protected API carries a WWW-Authenticate hint.",
     "kind":"run","cwd":str(PKG),"requires":["app"],"cmd":[PY,"-c",S_DISCOVERY]},
    {"id":"s3","group":"Part 1 · auth.md protocol (offline)","title":"Step 3 — One endpoint, two flows",
     "desc":"POST /agent/auth dispatches on a `type` field: identity_assertion (agent-verified) vs "
            "anonymous / verified_email (user-claimed). Here's the dispatcher in the code.",
     "kind":"run","cwd":str(PKG),
     "cmd":["bash","-c","grep -n -A 11 'async def agent_auth(req' app_server.py"]},
    {"id":"s4","group":"Part 1 · auth.md protocol (offline)","title":"Step 4 — Agent-verified (ID-JAG)",
     "desc":"A trusted provider signs an ID-JAG (a JWT); the app verifies it against the provider's "
            "JWKS and issues a token — no human, and NO refresh token (the agent re-mints instead).",
     "kind":"run","cwd":str(PKG),"requires":["app"],"cmd":[PY,"-c",S_VERIFIED]},
    {"id":"s5","group":"Part 1 · auth.md protocol (offline)","title":"Step 5 — User-claimed (OTP)",
     "desc":"The human-in-the-loop flow: the app emails a one-time code and WITHHOLDS the credential "
            "until the user confirms it. This is the shape of the approval gate in Part 2.",
     "kind":"run","cwd":str(PKG),"requires":["app"],"cmd":[PY,"-c",S_CLAIMED]},
    {"id":"s6","group":"Part 1 · auth.md protocol (offline)","title":"Step 6 — Least privilege (403)",
     "desc":"Scopes are enforced: a sites.read token is REFUSED (403) on a control.write call. This is "
            "what makes per-sub-agent scoping meaningful in Part 2.",
     "kind":"run","cwd":str(PKG),"requires":["app"],"cmd":[PY,"-c",S_LEASTPRIV]},
    {"id":"s7","group":"Part 1 · auth.md protocol (offline)","title":"Step 7 — Revocation (401)",
     "desc":"Mint a token, revoke it, watch the next call fail with 401.",
     "kind":"run","cwd":str(PKG),"requires":["app"],"cmd":[PY,"-c",S_REVOKE]},
    {"id":"s8","group":"Part 1 · auth.md protocol (offline)","title":"Step 8 — Wake-time re-mint ⭐",
     "desc":"THE idea. Store the durable GRANT (no token); call acquire() to get a FRESH token each "
            "wake. Two acquire() calls → two different tokens. An expired token is never a problem.",
     "kind":"run","cwd":str(PKG),"requires":["app"],"cmd":[PY,"-c",S_REMINT]},
    {"id":"s9","group":"Part 1 · auth.md protocol (offline)","title":"Step 9 — The audit trail",
     "desc":"Every state change the app made is recorded: verified_issued, otp_sent, claim_issued, "
            "setpoint_applied, revoked_token.",
     "kind":"run","cwd":str(PKG),"requires":["app"],"cmd":[PY,"-c",S_AUDIT]},

    # ---- Part 2: the long-running ADK agent ----
    {"id":"s10","group":"Part 2 · long-running ADK agent","title":"Step 10 — ADK setup check",
     "desc":"Part 2 runs a real ADK agent and needs google-adk + a model. Check it's installed.",
     "kind":"run","cwd":str(ROOT),
     "cmd":[PY,"-c","import google.adk; print('google-adk', getattr(google.adk,'__version__','present'), '✓')"]},
    {"id":"s11","group":"Part 2 · long-running ADK agent","title":"Step 11 — The state machine",
     "desc":"The agent's behaviour is driven by an explicit durable current_step, plus an auth_grants "
            "block (the grant, not the token). Here's both from work_order.py.",
     "kind":"run","cwd":str(PKG),
     "cmd":["bash","-c","grep -n -A 9 'class WorkOrderStep' work_order.py; echo; grep -n -A 14 'def initial_grants' work_order.py"]},
    {"id":"srvAgent","group":"Part 2 · long-running ADK agent","title":"Step 12 — Start the agent server (Part B)",
     "desc":"Start the long-running ADK service (port 8089). It builds the coordinator + read_agent + "
            "apply_agent and a durable SQLite session store. (Auto-starts the app server too if needed.)",
     "kind":"server","action":"start","target":"agent"},
    {"id":"s13","group":"Part 2 · long-running ADK agent","title":"Step 13 — Drive one work order end to end",
     "desc":"The full lifecycle: analyze (autonomous sites.read) → park at ANALYZED → refuse to skip → "
            "approval emails an OTP → approve → re-mint control.write → apply → COMPLETED. (LLM; ~1 min.)",
     "kind":"run","cwd":str(ROOT),"requires":["app","agent"],
     "cmd":[PY,"week17/authmd_adk/run_full_demo.py"]},
    {"id":"s14","group":"Part 2 · long-running ADK agent","title":"Step 14 — Audit after the run",
     "desc":"Confirm the real auth flows fired over the wire during the agent run.",
     "kind":"run","cwd":str(PKG),"requires":["app"],"cmd":[PY,"-c",S_AUDIT]},
    {"id":"s15a","group":"Part 2 · long-running ADK agent","title":"Step 15a — Durability: pause",
     "desc":"Create + kick off a NEW work order and stop at ANALYZED (do not approve). The session is "
            "now persisted to SQLite. (LLM; ~1 min.)",
     "kind":"run","cwd":str(PKG),"requires":["app","agent"],"cmd":[PY,"-c",S_DUR_PAUSE]},
    {"id":"s15restart","group":"Part 2 · long-running ADK agent","title":"Step 15b — Restart the agent",
     "desc":"Restart the agent server. In-memory state is destroyed; the SQLite session is not. This "
            "simulates the service being scaled to zero over a quiet weekend.",
     "kind":"server","action":"restart","target":"agent"},
    {"id":"s15c","group":"Part 2 · long-running ADK agent","title":"Step 15c — Durability: resume",
     "desc":"Resume the SAME session after the cold restart: it picks up at ANALYZED, the approval "
            "completes, a fresh control.write token is re-minted, and the work order COMPLETES.",
     "kind":"run","cwd":str(PKG),"requires":["app","agent"],"cmd":[PY,"-c",S_DUR_RESUME]},
    {"id":"vis","group":"Part 2 · long-running ADK agent","title":"Open the live visualizer",
     "desc":"A real-time view of the state machine, grants, OTP inbox and event timeline — drive the "
            "agent with buttons. (Opens the agent server's UI; start the agent server first.)",
     "kind":"link","href":AGENT_BASE + "/"},

    # ---- Cleanup ----
    {"id":"cstop","group":"Cleanup & teardown","title":"Stop both servers",
     "desc":"Terminate the app and agent server processes.",
     "kind":"server","action":"stop","target":"all"},
    {"id":"cclean","group":"Cleanup & teardown","title":"Clean generated files",
     "desc":"Delete the SQLite session store, the provider signing key, and __pycache__.",
     "kind":"run","cwd":str(PKG),
     "cmd":["bash","-c","rm -f work_order_sessions.db idjag_signing_key.pem; rm -rf __pycache__; echo 'cleaned: db + signing key + pycache'"]},
    {"id":"cdemo","group":"Cleanup & teardown","title":"Start over (demolish)",
     "desc":"Stop both servers AND delete all generated files — a clean slate to run from Step 0.",
     "kind":"demolish"},
]
STEP_BY_ID = {s["id"]: s for s in STEPS}


# ── managed child processes (the two demo servers) ──────────────────────────
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
        env = {**os.environ, **self.env}
        fh = open(self.log, "w")
        self.proc = subprocess.Popen([PY, str(PKG / self.script)], cwd=str(PKG),
                                     env=env, stdout=fh, stderr=subprocess.STDOUT)

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def tail(self, n: int = 60) -> str:
        if not self.log.exists():
            return "(no log yet)"
        return "\n".join(self.log.read_text(errors="replace").splitlines()[-n:])


SERVERS = {
    "app": Server("app", "app_server.py", APP_PORT, {"AUTHMD_APP_PORT": str(APP_PORT)}),
    "agent": Server("agent", "agent_server.py", AGENT_PORT,
                    {"AUTHMD_AGENT_PORT": str(AGENT_PORT), "AUTHMD_APP_PORT": str(APP_PORT)}),
}


async def _wait_healthy(srv: Server, timeout: float = 40) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if await srv.healthy():
            return True
        await asyncio.sleep(0.5)
    return False


def _pretty_cmd(cmd) -> str:
    """A clean, copyable one-liner for display (collapses inline -c snippets)."""
    if cmd is None:
        return ""
    if isinstance(cmd, str):
        return cmd
    if len(cmd) >= 3 and cmd[1] == "-c":
        return f"{Path(cmd[0]).name} -c '<python snippet>'"
    if len(cmd) >= 3 and cmd[0] == "bash" and cmd[1] == "-c":
        return cmd[2]
    return " ".join(Path(cmd[0]).name if i == 0 else x for i, x in enumerate(cmd))


app = FastAPI(title="auth.md × ADK — tutorial guide")
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
    out = {}
    for name, srv in SERVERS.items():
        out[name] = {"running": srv.running(), "healthy": await srv.healthy(), "port": srv.port}
    return out


@app.get("/api/logs/{which}")
async def logs(which: str, n: int = 80) -> dict:
    srv = SERVERS.get(which)
    return {"log": srv.tail(n) if srv else f"unknown server {which!r}"}


class ServerAction(BaseModel):
    target: str          # app | agent | all
    action: str          # start | stop | restart


@app.post("/api/server")
async def server_action(req: ServerAction) -> dict:
    targets = ["app", "agent"] if req.target == "all" else [req.target]
    msgs = []
    for t in targets:
        srv = SERVERS[t]
        if req.action in ("stop", "restart"):
            srv.stop(); msgs.append(f"{t}: stopped")
        if req.action in ("start", "restart"):
            if t == "agent" and not await SERVERS["app"].healthy():
                SERVERS["app"].start()
                await _wait_healthy(SERVERS["app"]); msgs.append("app: started (dependency)")
            srv.start()
            ok = await _wait_healthy(srv)
            msgs.append(f"{t}: {'healthy' if ok else 'started (not healthy yet — check logs)'}")
    return {"messages": msgs, "servers": await servers()}


def _stream_lines(cmd, cwd: str, env: dict, timeout: float):
    """Async generator: yield stdout+stderr lines, then an __EXIT__ sentinel."""
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
                                                  timeout=max(1, deadline_left(start, timeout)))
                except asyncio.TimeoutError:
                    proc.kill()
                    yield f"\n⏱️  step exceeded {timeout:.0f}s — killed.\n"
                    yield f"__EXIT__ 124 {time.time()-start:.1f}\n"
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


def deadline_left(start: float, timeout: float) -> float:
    return start + timeout - time.time()


class RunRequest(BaseModel):
    step_id: str


@app.post("/api/run")
async def run_step(req: RunRequest):
    step = STEP_BY_ID.get(req.step_id)
    if step is None or step.get("kind") != "run":
        async def err():
            yield f"step {req.step_id!r} is not runnable\n__EXIT__ 1 0\n"
        return StreamingResponse(err(), media_type="text/plain")

    # precheck required servers
    missing = [r for r in step.get("requires", []) if not await SERVERS[r].healthy()]
    if missing:
        async def need():
            names = ", ".join(missing)
            yield (f"⚠️  this step needs the {names} server(s) running.\n"
                   f"   Start them from the steps above (or the top-bar controls), then retry.\n"
                   f"__EXIT__ 1 0\n")
        return StreamingResponse(need(), media_type="text/plain")

    # longer budget for the LLM-backed Part-2 steps
    timeout = 360.0 if req.step_id in ("s13", "s15a", "s15c") else 90.0

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
    for f in ("work_order_sessions.db", "idjag_signing_key.pem"):
        p = PKG / f
        if p.exists():
            p.unlink(); removed.append(f)
    pyc = PKG / "__pycache__"
    if pyc.exists():
        for c in pyc.glob("*"):
            c.unlink()
        pyc.rmdir(); removed.append("__pycache__")
    return {"messages": ["stopped both servers", f"removed: {removed or 'nothing'}"],
            "servers": await servers()}


@app.on_event("shutdown")
async def _shutdown():
    for srv in SERVERS.values():
        srv.stop()


if __name__ == "__main__":
    import uvicorn
    port = _pick_free_port(GUIDE_PORT)
    banner = ["", "  🧭  auth.md × ADK — interactive tutorial guide"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠  port {GUIDE_PORT} is already in use by another process —",
                   f"         using {port} instead (set AUTHMD_GUIDE_PORT to choose)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)

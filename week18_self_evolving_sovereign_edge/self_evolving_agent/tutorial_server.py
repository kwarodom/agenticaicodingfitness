#!/usr/bin/env python3
"""Interactive tutorial GUIDE — a clickable, debuggable web runner for the
Week 18 self-evolving agent (see TUTORIAL.md).

A small CONTROL PLANE that:
  • serves the guide UI (static/guide.html);
  • runs each tutorial checkpoint as a subprocess and STREAMS its output to the
    browser live (the offline checkpoints finish in seconds);
  • manages the live self-evolution visualizer (server.py, port 8088) as a child
    process — start / stop / restart / health / log tail;
  • cleans up generated memory artifacts so you can start from an amnesiac slate.

Launch it and open the printed URL (auto-picks a free port if 8090 is taken):

    .venv/bin/python week18/self_evolving_agent/tutorial_server.py
    # → http://127.0.0.1:8090
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

PKG = Path(__file__).resolve().parent                 # …/week18/self_evolving_agent
ROOT = PKG.parents[1]                                 # …/agenticaicodingfitness
PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable
CP = PKG / "checkpoints"
LOG_DIR = PKG / ".logs"
LOG_DIR.mkdir(exist_ok=True)


def _port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _pick_free_port(preferred: int, span: int = 40) -> int:
    for p in range(preferred, preferred + span):
        if not _port_busy(p):
            return p
    return preferred


VIZ_PORT = int(os.environ.get("SELF_EVOLVING_PORT", "8088"))
GUIDE_PORT = int(os.environ.get("SELF_EVOLVING_GUIDE_PORT", "8090"))
VIZ_BASE = f"http://127.0.0.1:{VIZ_PORT}"


def _cp(name: str) -> list[str]:
    return [PY, str(CP / name)]


STEPS = [
    # ---- Understand the architecture ----
    {"id": "s0", "group": "Understand the architecture", "title": "Step 0 — Setup check",
     "desc": "Confirm the package imports and report whether LIVE mode (a real LLM via "
             "the claude CLI) is available. Every checkpoint runs OFFLINE regardless.",
     "kind": "run", "cwd": str(ROOT),
     "cmd": [PY, "-c",
             "import sys; sys.path.insert(0,'week18'); "
             "from self_evolving_agent import config; "
             "print('package import ✓'); "
             "print('LIVE mode (real LLM) available:', config.sdk_available()); "
             "print('memory dir:', config.MEMORY_DIR)"]},
    {"id": "s1", "group": "Understand the architecture",
     "title": "Step 1 — Tripartite memory (the 3 layers)",
     "desc": "Episodic (SessionDB/SQLite) · Semantic (MEMORY.md/USER.md) · Procedural "
             "(SKILL.md). Here is the episodic schema — the single declarative source of "
             "truth the DB auto-evolves to on boot.",
     "kind": "run", "cwd": str(PKG),
     "cmd": ["bash", "-c", "grep -n -A 22 'SCHEMA_SQL' core/session_db.py | head -28"]},
    {"id": "s2", "group": "Understand the architecture",
     "title": "Step 2 — Context fencing (injection safety)",
     "desc": "Recalled memory is wrapped in an XML fence with an authoritative system "
             "note so a malicious instruction persisted to MEMORY.md cannot hijack the "
             "agent. This is the function that builds the fence.",
     "kind": "run", "cwd": str(PKG),
     "cmd": ["bash", "-c", "grep -n -A 20 'def build_memory_context_block' core/semantic_memory.py"]},
    {"id": "s3", "group": "Understand the architecture",
     "title": "Step 3 — The subconscious loop",
     "desc": "After a run, a background meta-cognitive agent distils the episodic "
             "transcript into semantic facts + a SKILL.md. This is the prompt that drives it.",
     "kind": "run", "cwd": str(PKG),
     "cmd": ["bash", "-c", "grep -n -A 16 'META_COGNITIVE_PROMPT =' core/consolidation.py | head -22"]},

    # ---- Run the checkpoints (offline) ----
    {"id": "c1", "group": "Run the checkpoints (offline · $0)",
     "title": "Checkpoint 1 — Episodic memory (SessionDB)",
     "desc": "Build the real episodic store: WAL mode, jitter backoff under concurrent "
             "writers, declarative schema evolution, and dual FTS5 (latin + CJK) search.",
     "kind": "run", "cwd": str(ROOT), "cmd": _cp("checkpoint1_episodic.py")},
    {"id": "c2", "group": "Run the checkpoints (offline · $0)",
     "title": "Checkpoint 2 — Semantic memory & fencing",
     "desc": "Persist facts to MEMORY.md/USER.md and watch the XML fence neutralise a "
             "prompt-injection attempt that was hiding inside stored memory.",
     "kind": "run", "cwd": str(ROOT), "cmd": _cp("checkpoint2_semantic_fencing.py")},
    {"id": "c3", "group": "Run the checkpoints (offline · $0)",
     "title": "Checkpoint 3 — Procedural memory (SKILL.md)",
     "desc": "Skill matching against an index, context-fenced injection, versioning with "
             "archival, and rollback after a bad refinement.",
     "kind": "run", "cwd": str(ROOT), "cmd": _cp("checkpoint3_skills.py")},
    {"id": "c4", "group": "Run the checkpoints (offline · $0)",
     "title": "Checkpoint 4 — Background consolidation",
     "desc": "Distil a finished transcript into semantic facts + a SKILL.md (the "
             "subconscious loop), and snapshot working memory before context compaction.",
     "kind": "run", "cwd": str(ROOT), "cmd": _cp("checkpoint4_consolidation.py")},
    {"id": "c5", "group": "Run the checkpoints (offline · $0)",
     "title": "Checkpoint 5 — Memory garbage collection",
     "desc": "TTL compression (summarise an old session into MEMORY.md, prune raw "
             "messages) and GDPR-safe surgical erasure with an audit trail.",
     "kind": "run", "cwd": str(ROOT), "cmd": _cp("checkpoint5_gc.py")},
    {"id": "c6", "group": "Run the checkpoints (offline · $0)",
     "title": "Checkpoint 6 — GEPA prompt evolution",
     "desc": "The agent generates its own prompt variants, scores them, and keeps only "
             "the Pareto-optimal front (accuracy / cost / tokens) — no human prompt engineer.",
     "kind": "run", "cwd": str(ROOT), "cmd": _cp("checkpoint6_gepa.py")},

    # ---- Watch it self-evolve ----
    {"id": "c7", "group": "Watch it self-evolve",
     "title": "Checkpoint 7 — Capstone: compound returns",
     "desc": "Assemble all three layers into one agent and run the SAME task 3×, "
             "consolidating between runs. Turns and cost drop each run — the agent wrote "
             "the skill it then reused. (Deterministic offline simulation.)",
     "kind": "run", "cwd": str(ROOT), "cmd": _cp("checkpoint7_self_evolving.py")},
    {"id": "srv", "group": "Watch it self-evolve",
     "title": "Start the live visualizer service",
     "desc": "A long-running service holding ONE agent whose memory lives on disk. If the "
             "claude CLI is signed in, runs are REAL LLM turns; otherwise an offline "
             "simulation (port 8088).",
     "kind": "server", "action": "start", "target": "viz"},
    {"id": "viz", "group": "Watch it self-evolve",
     "title": "Open the live visualizer",
     "desc": "Drive the agent: Run a task → Consolidate (the subconscious loop) → Run the "
             "SAME task again and watch it load the learned skill and finish faster. The "
             "three memory layers update live. (Start the service first.)",
     "kind": "link", "href": VIZ_BASE + "/"},

    # ---- Cleanup ----
    {"id": "cclean", "group": "Cleanup & teardown", "title": "Reset to amnesiac slate",
     "desc": "Wipe MEMORY.md, USER.md, the skill library, and the episodic DB — re-watch "
             "the agent learn from zero.",
     "kind": "run", "cwd": str(ROOT),
     "cmd": [PY, "-c", "import sys; sys.path.insert(0,'week18'); "
             "from self_evolving_agent import config; "
             "print('removed:', config.wipe_memory())"]},
    {"id": "cdemo", "group": "Cleanup & teardown", "title": "Start over (demolish)",
     "desc": "Stop the visualizer service AND wipe all learned memory — a clean slate.",
     "kind": "demolish"},
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
        # Drop ANTHROPIC_API_KEY so the claude CLI uses subscription auth (an
        # unrelated key would be rejected).
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        self.proc = subprocess.Popen([PY, str(PKG / self.script)], cwd=str(PKG),
                                     env={**env, **self.env},
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


SERVERS = {"viz": Server("viz", "server.py", VIZ_PORT, {"SELF_EVOLVING_PORT": str(VIZ_PORT)})}


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


app = FastAPI(title="Self-evolving agent — tutorial guide")
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
    srv = SERVERS.get(which if which in SERVERS else "viz")
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

    timeout = 120.0
    # The checkpoints drop ANTHROPIC_API_KEY so any live sub-call uses CLI auth.
    env = {**step.get("env", {})}

    async def body():
        if _run_lock.locked():
            yield "⚠️  another step is already running — wait for it to finish.\n__EXIT__ 1 0\n"
            return
        async with _run_lock:
            yield f"$ (cwd={Path(step['cwd']).name}) {_pretty_cmd(step['cmd'])}\n\n"
            async for chunk in _stream_lines(step["cmd"], step["cwd"], env, timeout):
                yield chunk
    return StreamingResponse(body(), media_type="text/plain")


@app.post("/api/cleanup")
async def cleanup() -> dict:
    for srv in SERVERS.values():
        srv.stop()
    sys.path.insert(0, str(ROOT / "week18"))
    from self_evolving_agent import config
    removed = config.wipe_memory()
    return {"messages": ["stopped the visualizer", f"wiped memory: {removed}"],
            "servers": await servers()}


@app.on_event("shutdown")
async def _shutdown():
    for srv in SERVERS.values():
        srv.stop()


if __name__ == "__main__":
    import uvicorn
    port = _pick_free_port(GUIDE_PORT)
    banner = ["", "  🧠  Self-evolving agent — interactive tutorial guide"]
    if port != GUIDE_PORT:
        banner += [f"      ⚠  port {GUIDE_PORT} is already in use —",
                   f"         using {port} instead (set SELF_EVOLVING_GUIDE_PORT to choose)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)

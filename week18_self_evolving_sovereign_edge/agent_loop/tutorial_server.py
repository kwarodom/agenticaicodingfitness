#!/usr/bin/env python3
"""Interactive, explainable tutorial for the Claude Agent SDK **agent loop**.

A small control plane that serves a clickable web guide (static/guide.html) and,
for each part of the tutorial, lets you:

  • read the CONCEPT (the why, distilled from the Week 18 PDF);
  • view the exact SOURCE of the demo that part runs;
  • RUN that demo for real — the SDK drives the local `claude` CLI — and watch
    the agent loop narrate itself live (REASON → ACT → OBSERVE → RESULT) as the
    output streams into the browser.

These are REAL agent loops, not mocks. They use your existing Claude Code / CLI
sign-in, so no ANTHROPIC_API_KEY is needed. Each demo is capped on turns and USD,
so a full click-through costs a few cents.

Launch (auto-picks a free port if 8090 is taken):

    .venv/bin/python week18/agent_loop/tutorial_server.py
    # → http://127.0.0.1:8090
"""
from __future__ import annotations

import asyncio
import os
import shutil
import socket
import sys
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

PKG = Path(__file__).resolve().parent                 # …/week18/agent_loop
ROOT = PKG.parents[1]                                 # …/agenticaicodingfitness
PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable
DEMOS = PKG / "demos"

GUIDE_PORT = int(os.environ.get("AGENTLOOP_GUIDE_PORT", "8090"))


def _port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _pick_free_port(preferred: int, span: int = 40) -> int:
    for p in range(preferred, preferred + span):
        if not _port_busy(p):
            return p
    return preferred


# ── the tutorial steps ───────────────────────────────────────────────────────
# Each "run" step maps to a demos/stepNN_*.py file; concept steps just explain.
STEPS = [
    {
        "id": "intro", "group": "Foundations", "kind": "concept",
        "title": "Part 1 · What is an agent loop?", "level": "beginner",
        "desc": "An agent loop is the repeating cycle — REASON → ACT → OBSERVE → "
        "repeat — that turns a one-shot language model into something that pursues "
        "a goal on its own: it can read files, run commands, call your APIs, check "
        "results, and keep going until the job is done.\n\n"
        "Mental model: a smart employee given a task. They don't ask permission for "
        "every small action — they use their tools, check progress, and report back "
        "only when it's truly done. That 'think + do' cycle is the loop.\n\n"
        "Everything in this tutorial is a REAL loop. The demos call the model for "
        "real through the Claude Agent SDK (it drives your local `claude` CLI), so "
        "you watch genuine REASON/ACT/OBSERVE turns — not a recording.",
    },
    {
        "id": "step01", "group": "Foundations", "kind": "run", "demo": "step01_hello_agent.py",
        "title": "Part 2 · Your first agent loop", "level": "beginner",
        "desc": "The smallest real agent: give Claude a goal plus two tools and let "
        "the loop run until the goal is met. It must ACT (use a tool) to discover "
        "what files exist — it can't just guess — then OBSERVE the result and "
        "answer. Watch SESSION → ACT → OBSERVE → RESULT, each ACT being one turn.",
    },
    {
        "id": "step02", "group": "Foundations", "kind": "run", "demo": "step02_turns_messages.py",
        "title": "Part 3 · Turns, messages & results", "level": "beginner",
        "desc": "A turn is one full round trip: Claude emits a tool call, the SDK "
        "runs it, the result is fed back. Only tool-use turns count toward "
        "max_turns. This demo tags every message type as it arrives "
        "(SystemMessage / AssistantMessage / UserMessage / ResultMessage) and shows "
        "what each ResultMessage subtype means and how to react.",
    },
    {
        "id": "step03", "group": "Tools", "kind": "run", "demo": "step03_builtin_tools.py",
        "title": "Part 4a · Built-in tools & permissions", "level": "intermediate",
        "desc": "The SDK ships every tool that powers Claude Code. Three settings "
        "control what runs: allowed_tools (auto-approve), disallowed_tools (never "
        "run), and permission_mode (everything else). Here the loop gets read/search "
        "tools to find a bug but is DENIED Bash/Write/Edit — proving disallowed_tools "
        "always wins. Watch it chain Glob → Grep → Read.",
    },
    {
        "id": "step04", "group": "Tools", "kind": "run", "demo": "step04_custom_tool.py",
        "title": "Part 4b · Defining a custom tool", "level": "intermediate",
        "desc": "Custom tools connect the loop to YOUR systems. Write an async "
        "handler, wrap it with @tool(name, description, schema), bundle it with "
        "create_sdk_mcp_server, and expose it via mcp_servers. The description is how "
        "Claude decides when to call it. This demo exposes a fake CRM and asks the "
        "loop to find the highest churn-risk customer from data it has never seen.",
    },
    {
        "id": "step05", "group": "Control & safety", "kind": "run", "demo": "step05_hooks_safety.py",
        "title": "Part 5 · Hooks — safety & audit", "level": "intermediate",
        "desc": "Hooks are callbacks that fire at points in the loop, in YOUR process "
        "(no tokens, can't be talked past). This demo arms a PreToolUse safety gate "
        "that BLOCKS dangerous commands (rm -rf, DROP TABLE, protected paths) and a "
        "PostToolUse audit log. We ask the loop to do something destructive and watch "
        "the gate deny it while safe work proceeds.",
    },
    {
        "id": "step06", "group": "Continuity", "kind": "run", "demo": "step06_sessions.py",
        "title": "Part 6 · Sessions & 24/7 continuity", "level": "intermediate",
        "desc": "Every run creates a resumable session. This demo runs the loop "
        "twice: run 1 learns a secret codename and saves the session id; run 2 "
        "resumes that session and answers a follow-up using memory from run 1 — "
        "without the codename in its prompt. This is the basis of agents that pause "
        "for days and perpetual monitors that resume on a timer.",
    },
    {
        "id": "step07", "group": "Scaling", "kind": "run", "demo": "step07_multi_agent.py",
        "title": "Part 7 · Multi-agent orchestration", "level": "advanced",
        "desc": "For big or independent work, the main loop spawns subagents — fresh "
        "context, focused prompt, minimal tools — declared with AgentDefinition and "
        "invoked via the Agent tool. Here the orchestrator delegates to a "
        "code-reviewer and a bug-hunter, then synthesises one report. Context "
        "isolation + parallelism + specialisation, without blowing one window.",
    },
    {
        "id": "step08", "group": "Real-world", "kind": "run", "demo": "step08_usecase_triage.py",
        "title": "Part 8 · Use case — support triage", "level": "intermediate",
        "desc": "The payoff: a production-shaped loop that clears a support queue. "
        "Two custom tools (list tickets, send reply) plus a workflow system prompt "
        "let one loop read every ticket, classify L1/L2/L3, and resolve or escalate "
        "with the right priority — including a critical data-loss incident. The work "
        "that ate hours of founder time, in one run.",
    },
    {
        "id": "step09", "group": "Real-world", "kind": "run", "demo": "step09_production.py",
        "title": "Part 9 · Bounded execution & cost", "level": "advanced",
        "desc": "Unconstrained loops run away (too many turns, runaway cost). This "
        "demo sets a deliberately tiny budget (max_turns=3, max_budget_usd=0.02) on a "
        "deep task so the loop hits its ceiling — proving the guardrail fires — then "
        "detects the stop subtype and shows how a real system resumes or splits.",
    },
    {
        "id": "outro", "group": "Real-world", "kind": "concept",
        "title": "Part 10 · Production checklist", "level": "all levels",
        "desc": "Ship-ready agent loop checklist:\n"
        "• Set max_turns (≤50) — prevent infinite loops.\n"
        "• Set max_budget_usd — cap spend per run.\n"
        "• Add a PreToolUse safety hook — block dangerous ops.\n"
        "• Add a PostToolUse audit log — compliance & debugging.\n"
        "• Persist session_id to disk — enable resume after a crash.\n"
        "• Wrap in try/except with retry — handle transient API errors.\n"
        "• Use permission_mode='dontAsk' in CI; 'bypassPermissions' only in "
        "containers.\n"
        "• Use effort=low/medium for routine tasks; cheaper models for subagents.\n"
        "• Add a CLAUDE.md to the project root — rules survive context compaction.\n"
        "• Test with a small max_turns first, then scale.\n\n"
        "Now do the workshop: open the PDF's Part 10 exercises and rebuild these "
        "patterns from scratch.",
    },
]
STEP_BY_ID = {s["id"]: s for s in STEPS}

app = FastAPI(title="Agent Loop — interactive tutorial")
_run_lock = asyncio.Lock()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(PKG / "static" / "guide.html")


@app.get("/api/steps")
async def steps() -> dict:
    def public(s):
        return {k: s.get(k) for k in ("id", "group", "title", "desc", "kind", "level")} | \
               {"demo": s.get("demo")}
    cli = shutil.which("claude") is not None
    return {"steps": [public(s) for s in STEPS], "cli_ready": cli}


@app.get("/api/source/{step_id}")
async def source(step_id: str) -> dict:
    step = STEP_BY_ID.get(step_id)
    if not step or not step.get("demo"):
        return {"source": "(no source for this step)"}
    path = DEMOS / step["demo"]
    if not path.exists():
        return {"source": f"(missing file: {step['demo']})"}
    return {"source": path.read_text(), "filename": step["demo"]}


def _stream_demo(demo: str, timeout: float):
    async def gen():
        start = time.time()
        # PYTHONUNBUFFERED → stream live; ENABLE_TOOL_SEARCH=0 → keep the loop
        # clean (no ToolSearch interstitial; see config.py for why).
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "ENABLE_TOOL_SEARCH": "0"}
        proc = await asyncio.create_subprocess_exec(
            PY, str(DEMOS / demo), cwd=str(PKG), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            while True:
                try:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(),
                        timeout=max(1, start + timeout - time.time()))
                except asyncio.TimeoutError:
                    proc.kill()
                    yield (f"\n⏱  step exceeded {timeout:.0f}s — killed.\n"
                           f"__EXIT__ 124 {time.time()-start:.1f}\n")
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

    if shutil.which("claude") is None:
        async def nocli():
            yield ("⚠  The `claude` CLI was not found on PATH.\n"
                   "   These demos make real SDK calls via the CLI. Install & sign in:\n"
                   "     npm install -g @anthropic-ai/claude-code   then: claude\n"
                   "__EXIT__ 1 0\n")
        return StreamingResponse(nocli(), media_type="text/plain")

    # Multi-agent / triage loops do more turns → allow longer.
    timeout = 360.0 if req.step_id in ("step07", "step08") else 240.0

    async def body():
        if _run_lock.locked():
            yield "⚠  another demo is already running — wait for it to finish.\n__EXIT__ 1 0\n"
            return
        async with _run_lock:
            yield f"$ {Path(PY).name} demos/{step['demo']}\n\n"
            async for chunk in _stream_demo(step["demo"], timeout):
                yield chunk
    return StreamingResponse(body(), media_type="text/plain")


@app.post("/api/cleanup")
async def cleanup() -> dict:
    removed = []
    sb = PKG / ".sandbox"
    if sb.exists():
        shutil.rmtree(sb, ignore_errors=True)
        removed.append(".sandbox/ (scratch files & saved session)")
    pyc_dirs = list(PKG.rglob("__pycache__"))
    for pyc in pyc_dirs:
        shutil.rmtree(pyc, ignore_errors=True)
    if pyc_dirs:
        removed.append(f"{len(pyc_dirs)} __pycache__ folder(s)")
    msg = "Cleaned: " + ", ".join(removed) if removed else "Nothing to clean — already tidy."
    return {"messages": [msg]}


if __name__ == "__main__":
    import uvicorn

    port = _pick_free_port(GUIDE_PORT)
    cli_ok = shutil.which("claude") is not None
    banner = ["", "  🔁  Agent Loop — interactive, explainable tutorial"]
    if not cli_ok:
        banner += ["      ⚠  `claude` CLI not found — demos will show install hints."]
    else:
        banner += ["      ✓ `claude` CLI found — demos run for real (uses your sign-in)."]
    if port != GUIDE_PORT:
        banner += [f"      ⚠ port {GUIDE_PORT} busy — using {port} "
                   "(set AGENTLOOP_GUIDE_PORT to choose)."]
    banner += [f"      open  →  http://127.0.0.1:{port}", ""]
    print("\n".join(banner), flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)

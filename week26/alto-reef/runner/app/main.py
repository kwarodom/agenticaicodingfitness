from __future__ import annotations
import asyncio, os, json, time
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from .world import WORLD
from .models import AssignRequest, AskRequest
from . import mock

TOKEN = os.environ.get("REEF_TOKEN", "")
app = FastAPI(title="Alto Reef Runner", version="0.1.0-m6")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def auth(authorization: str | None = Header(default=None)):
    if TOKEN and authorization != f"Bearer {TOKEN}":
        raise HTTPException(401, "bearer token required")

@app.get("/api/health")
def health(_=Depends(auth)):
    return WORLD.health()

@app.get("/api/sandboxes")
def sandboxes(_=Depends(auth)):
    if WORLD.mode == "real": WORLD.health()
    return list(WORLD.sandboxes.values())

@app.get("/api/profiles")
def profiles(_=Depends(auth)):
    return list(WORLD.profiles.values())

@app.post("/api/sandboxes/{sid}/members")
def add_member(sid: str, body: AssignRequest, _=Depends(auth)):
    if sid not in WORLD.sandboxes or body.profileId not in WORLD.profiles: raise HTTPException(404)
    return WORLD.assign(sid, body.profileId)

@app.delete("/api/sandboxes/{sid}/members/{pid}")
def remove_member(sid: str, pid: str, _=Depends(auth)):
    if sid not in WORLD.sandboxes: raise HTTPException(404)
    return WORLD.unassign(sid, pid)

@app.get("/api/events")
def events(kind: str | None = None, limit: int = 100, _=Depends(auth)):
    ev = [e for e in WORLD.events if kind in (None, "", e.kind)]
    return ev[-limit:]

@app.post("/api/ask")
def ask(body: AskRequest, _=Depends(auth)):
    """Fan a prompt to selected profiles. In mock mode we answer with an explicit fallback line; real dispatch lands in M2."""
    names = [WORLD.profiles[p].name for p in body.profileIds if p in WORLD.profiles] or ["the team"]
    WORLD.emit(kind="chat", actor="You", text=body.text, level="info")
    if WORLD.mode == "mock":
        WORLD.emit(kind="system", actor="System", text=f"MOCK: no LLM reachable; '{body.text[:60]}' would be dispatched to {', '.join(names)} via their sandbox chat endpoints in M2.", level="warn")
    else:
        WORLD.emit(kind="system", actor="System", text="Dispatch not implemented yet (M2). Nothing was sent.", level="warn")
    return {"queued": False, "recipients": names}

@app.get("/api/stream")
async def stream(_=Depends(auth)):
    q: asyncio.Queue = asyncio.Queue(maxsize=200); WORLD.subscribers.add(q)
    async def gen():
        try:
            yield f"event: hello\ndata: {json.dumps({'mode': WORLD.mode, 'at': time.time()})}\n\n"
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"event: reef\ndata: {ev.model_dump_json()}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            WORLD.subscribers.discard(q)
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

async def _mock_ticker():
    while True:
        await asyncio.sleep(9)
        ev = mock.random_event(); WORLD.emit(kind=ev.kind, actor=ev.actor, text=ev.text, level=ev.level)

@app.on_event("startup")
async def _startup():
    if WORLD.mode == "mock": asyncio.create_task(_mock_ticker())

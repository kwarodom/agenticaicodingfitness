"""In-memory world state. Real mode fills it from adapters; mock mode from mock.py.
Persistence (SQLite) arrives in M1; the shape is already the spec's data model."""
from __future__ import annotations
import asyncio, os, shutil, time
from typing import Literal
from . import mock
from .models import Profile, Sandbox, Landmark, Health, Event
from .adapters import openshell, nemoclaw, http_probes

Mode = Literal["mock","real"]

def detect_mode() -> Mode:
    want = os.environ.get("REEF_MODE", "auto")
    has_gateway = shutil.which("openshell") is not None
    if want == "mock" and has_gateway:
        raise SystemExit("REEF_MODE=mock refused: a real openshell binary is present (spec 4.5).")
    if want == "real" or (want == "auto" and has_gateway):
        return "real"
    return "mock"

class World:
    def __init__(self) -> None:
        self.mode: Mode = detect_mode()
        self.profiles: dict[str, Profile] = {}
        self.sandboxes: dict[str, Sandbox] = {}
        self.landmarks: list[Landmark] = []
        self.events: list[Event] = []
        self.subscribers: set[asyncio.Queue] = set()
        self._event_id = 0
        self.health_at = 0.0
        self.gateway: dict = {}; self.inference: dict = {}
        if self.mode == "mock":
            self.profiles = {p.id: p for p in mock.profiles()}
            self.sandboxes = {s.id: s for s in mock.sandboxes()}
            self.landmarks = mock.landmarks()
            self.events = mock.seed_events(); self._event_id = max(e.id for e in self.events)
            self.gateway = {"ok": True, "detail": "mock gateway", "at": time.time()}
            self.inference = {"provider": "local-vllm", "model": mock.MODEL, "at": time.time()}
        else:
            self.refresh_real()

    # ---- real mode -------------------------------------------------------
    def refresh_real(self) -> None:
        st = openshell.status(); inf = openshell.inference_get(); sl = openshell.sandbox_list()
        self.gateway = {"ok": st.ok and st.parsed["connected"], "detail": (st.stdout or st.stderr).strip()[:200], "command": st.command, "at": st.at}
        self.inference = {**(inf.parsed or {}), "ok": inf.ok, "command": inf.command, "at": inf.at}
        seen = set()
        for i, row in enumerate(sl.parsed or []):
            sid = row["name"]; seen.add(sid)
            sb = self.sandboxes.get(sid) or Sandbox(id=sid, name=sid, blueprint="unknown", slot=i)
            sb.status = {"running":"running","ready":"running","error":"error","stopped":"stopped"}.get(row["status"], "unknown")
            sb.modelHandle = self.inference.get("model"); sb.provider = self.inference.get("provider")
            sb.probedAt = sl.at; sb.source = " ".join(sl.command)
            pl = nemoclaw.policy_list(sid)
            if pl.ok: sb.presets = pl.parsed or []
            self.sandboxes[sid] = sb
        for sid in list(self.sandboxes):
            if sid not in seen: self.sandboxes[sid].status = "unknown"
        probes = {"vllm": ("vLLM","reactor","http://localhost:8000/health"), "nat": ("NAT server","nautilus","http://localhost:8001/v1/models"),
                  "phoenix": ("Phoenix","buoy","http://localhost:6006"), "otel": ("OTel collector","buoy","http://localhost:4318")}
        lms = [Landmark(id="gateway", label="OpenShell gateway", kind="lighthouse", ok=self.gateway["ok"], detail=self.gateway["detail"], probedAt=st.at)]
        for k, (label, kind, url) in probes.items():
            r = http_probes.probe(url)
            lms.append(Landmark(id=k, label=label, kind=kind, ok=r["ok"], detail=(f"{r.get('status')} · {r['ms']} ms" if r["ok"] else r.get("error","")), probedAt=r["at"], url=url))
        self.landmarks = lms; self.health_at = time.time()

    # ---- shared ----------------------------------------------------------
    def health(self) -> Health:
        if self.mode == "real" and time.time() - self.health_at > 10: self.refresh_real()
        return Health(mode=self.mode, gateway=self.gateway, inference=self.inference, landmarks=self.landmarks, at=time.time())

    def emit(self, **kw) -> Event:
        self._event_id += 1
        ev = Event(id=self._event_id, at=time.time(), **kw)
        self.events.append(ev); self.events = self.events[-500:]
        for q in list(self.subscribers):
            try: q.put_nowait(ev)
            except asyncio.QueueFull: pass
        return ev

    def assign(self, sandbox_id: str, profile_id: str) -> Sandbox:
        sb = self.sandboxes[sandbox_id]; p = self.profiles[profile_id]
        if p.sandboxId and p.sandboxId in self.sandboxes:
            old = self.sandboxes[p.sandboxId]; old.members = [m for m in old.members if m != profile_id]
        if profile_id not in sb.members: sb.members.append(profile_id)
        p.sandboxId = sandbox_id
        # In real mode this is where `nemoclaw <s> ...` sub-agent provisioning would run (M3). Recorded honestly as UI-only for now.
        self.emit(kind="history", actor="System", sandboxId=sandbox_id, profileId=profile_id,
                  text=f"{p.name} assigned to {sb.name}" + ("" if self.mode=="mock" else " (runner metadata only — sub-agent provisioning lands in M3)"), level="info")
        return sb

    def unassign(self, sandbox_id: str, profile_id: str) -> Sandbox:
        sb = self.sandboxes[sandbox_id]; sb.members = [m for m in sb.members if m != profile_id]
        p = self.profiles.get(profile_id)
        if p: p.sandboxId = None
        self.emit(kind="history", actor="System", sandboxId=sandbox_id, profileId=profile_id, text=f"{p.name if p else profile_id} removed from {sb.name}", level="info")
        return sb

WORLD = World()

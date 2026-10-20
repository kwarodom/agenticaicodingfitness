from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

Harness = Literal["openclaw", "hermes", "deepagents", "nat"]
Archetype = Literal["researcher","analyst","critic","planner","writer","coder","lead","ops_engineer"]
Tier = Literal["restricted","balanced","open","personal"]
SandboxStatus = Literal["running","degraded","error","stopped","unknown"]

class Profile(BaseModel):
    id: str; name: str; harness: Harness; archetype: Archetype
    color: str = "#3fd0c9"
    accessories: dict[str, str] = Field(default_factory=dict)
    skills: list[str] = Field(default_factory=list)
    tier: Tier = "balanced"
    systemPrompt: str = ""
    sandboxId: Optional[str] = None
    persona: str = ""

class Forward(BaseModel):
    remotePort: int; localPort: int

class Sandbox(BaseModel):
    id: str; name: str; blueprint: str
    status: SandboxStatus = "unknown"
    modelHandle: Optional[str] = None
    provider: Optional[str] = None
    presets: list[str] = Field(default_factory=list)
    policyRevision: Optional[int] = None
    forwards: list[Forward] = Field(default_factory=list)
    members: list[str] = Field(default_factory=list)
    lastRunSummary: Optional[str] = None
    lastRunStatus: Optional[str] = None
    slot: int = 0            # position on the reef map
    probedAt: Optional[float] = None
    source: str = "mock"     # "mock" | command that produced it

class Landmark(BaseModel):
    id: str; label: str; kind: str; ok: Optional[bool] = None
    detail: str = ""; probedAt: Optional[float] = None; url: Optional[str] = None

class Health(BaseModel):
    mode: Literal["mock","real"]
    gateway: dict; inference: dict
    landmarks: list[Landmark]
    at: float

class Event(BaseModel):
    id: int; at: float
    kind: Literal["chat","activity","history","system"]
    sandboxId: Optional[str] = None; profileId: Optional[str] = None
    actor: str; text: str
    level: Literal["info","warn","error","ok"] = "info"

class AssignRequest(BaseModel):
    profileId: str

class AskRequest(BaseModel):
    text: str
    profileIds: list[str] = Field(default_factory=list)

"""Mock world for frontend development. Watermarked in the UI. Never used when a real gateway is detected."""
from __future__ import annotations
import time, random, itertools
from .models import Profile, Sandbox, Landmark, Event

MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"

def profiles() -> list[Profile]:
    P = lambda **k: Profile(**k)
    return [
        P(id="clawdia", name="Clawdia", harness="openclaw", archetype="researcher", color="#4aa3ff", skills=["summarize","session-logs","web-research","file-io"], sandboxId="quill-hollow", persona="Curious, cites everything, hates unsourced claims."),
        P(id="shelldon", name="Shelldon", harness="openclaw", archetype="analyst", color="#ff5c8a", skills=["model-usage","mcporter","data-analysis","file-io"], sandboxId="workbench", persona="Precise, data-driven. Won't claim without evidence."),
        P(id="coraline", name="Coraline", harness="openclaw", archetype="critic", color="#f2b64c", skills=["oracle","web-research","fact-check"], sandboxId="the-bridge", persona="Finds the hole in the plan before the plan finds you."),
        P(id="reefus", name="Reefus", harness="openclaw", archetype="planner", color="#3fd0c9", skills=["taskflow","planning","team-coordination"], sandboxId=None, persona="Turns goals into three concrete steps."),
        P(id="pearl", name="Pearl", harness="openclaw", archetype="writer", color="#ff8fd0", skills=["summarize","final-synthesis","file-io"], sandboxId="the-bridge", persona="Writes the last paragraph first."),
        P(id="snips", name="Snips", harness="openclaw", archetype="coder", color="#ff7a45", skills=["code-authoring","code-execution","coding-agent","github"], sandboxId="workbench", persona="Ships small diffs with tests."),
        P(id="captain-claw", name="Captain Claw", harness="hermes", archetype="lead", color="#9b7bff", skills=["delegation","review"], sandboxId="coral-cove", persona="Assigns, reviews, signs off."),
        P(id="alto-ops", name="Alto Ops", harness="nat", archetype="ops_engineer", color="#76b900", skills=["chiller_kpi","bms.read_point","bms.list_alarms"], tier="restricted", sandboxId="alto-ops", persona="Hotel chiller-plant engineer. Read-only by policy."),
        P(id="franck", name="Franck", harness="openclaw", archetype="researcher", color="#5fd6ff", skills=["web-research"], sandboxId="coral-cove", persona="Speaks three languages, cites in all of them."),
    ]

def sandboxes() -> list[Sandbox]:
    S = lambda **k: Sandbox(**k, modelHandle=MODEL, provider="local-vllm", probedAt=time.time(), source="mock")
    return [
        S(id="coral-cove", name="Coral Cove", blueprint="openclaw", status="running", presets=["npm","pypi","huggingface","brew","github","openclaw-pricing"], policyRevision=6, members=["captain-claw","franck"], slot=0, lastRunSummary="Idle", lastRunStatus="done"),
        S(id="the-bridge", name="The Bridge", blueprint="openclaw", status="running", presets=["npm","pypi","huggingface","brave","openclaw-pricing","local-inference"], policyRevision=6, members=["pearl","coraline"], slot=1, lastRunSummary="Run succeeded: 1/1 agents finished.", lastRunStatus="done"),
        S(id="quill-hollow", name="Quill Hollow", blueprint="openclaw", status="degraded", presets=["npm","pypi","openclaw-pricing"], policyRevision=3, members=["clawdia"], slot=2, lastRunSummary="Gateway OK, model check pending", lastRunStatus="running"),
        S(id="workbench", name="Workbench", blueprint="openclaw", status="error", presets=["npm","pypi","huggingface","brew","openclaw-pricing"], policyRevision=5, members=["shelldon","snips"], slot=3, lastRunSummary="Run failed: 0/2 agents succeeded.", lastRunStatus="failed"),
        S(id="alto-ops", name="Alto Ops (NAT)", blueprint="alto-ops-nat", status="running", presets=["local-inference","alto-bms","otel-collector"], policyRevision=2, members=["alto-ops"], slot=4, lastRunSummary="chiller_kpi 6h: kW/RT 0.78 OK", lastRunStatus="done"),
    ]

def landmarks() -> list[Landmark]:
    now = time.time()
    return [
        Landmark(id="gateway", label="OpenShell gateway", kind="lighthouse", ok=True, detail="Connected · 0.0.116 · :8080", probedAt=now),
        Landmark(id="vllm", label="vLLM", kind="reactor", ok=True, detail=f"{MODEL} · :8000", probedAt=now, url="http://localhost:8000/v1/models"),
        Landmark(id="nat", label="NAT server", kind="nautilus", ok=True, detail="nat serve :8001 · mcp :9901", probedAt=now, url="http://localhost:8001/v1/models"),
        Landmark(id="phoenix", label="Phoenix", kind="buoy", ok=False, detail="connection refused :6006", probedAt=now, url="http://localhost:6006"),
        Landmark(id="otel", label="OTel collector", kind="buoy", ok=True, detail=":4318 /v1/traces", probedAt=now, url="http://localhost:4318"),
    ]

_ids = itertools.count(1)
_LINES = [
    ("activity","Reefus","Planning: split the task into 3 steps; assigning Coraline to fact-check.","info"),
    ("activity","OpenShell","deny  /usr/local/bin/openclaw → api.open-meteo.com:443 (no matching endpoint)","warn"),
    ("activity","OpenShell","inspect_for_inference  openclaw → inference.local:443","info"),
    ("chat","Captain Claw","General prompt accepted. Sandbox-assigned claws stay in place.","info"),
    ("chat","ERIF","Coral Cove has the best bubble acoustics for serious lobster planning.","info"),
    ("activity","Alto Ops","chiller_kpi(hours=6) → avg_kw=412.3 avg_rt=528.6 kw_per_rt=0.780 status=OK","ok"),
    ("system","System","Run nemoclaw-snips-workbench-80f7bc61 finished: 0/2 agents succeeded (LLM request timed out).","error"),
    ("activity","OpenShell","allow  python3.12 → otel.alto.local:4318 POST /v1/traces","ok"),
    ("history","System","Policy revision 6 applied to The Bridge (preset add: local-inference).","info"),
]
def seed_events(n: int = 9) -> list[Event]:
    now = time.time(); out = []
    for i, (k, a, t, lvl) in enumerate(_LINES[:n]):
        out.append(Event(id=next(_ids), at=now - (n-i)*47, kind=k, actor=a, text=t, level=lvl))
    return out

def random_event() -> Event:
    k, a, t, lvl = random.choice(_LINES)
    return Event(id=next(_ids), at=time.time(), kind=k, actor=a, text=t, level=lvl)

#!/usr/bin/env python3
"""PART B — the long-running ADK agent that consumes auth.md.

The energy work-order coordinator and its two least-privilege sub-agents, wired
with the same ADK durability primitives as hr_onboarding, plus the auth layer
the slides add:

  • before_tool_callback (AuthInjector)  — slide B3's injection point. Right
    BEFORE a tool hits the protected API, it mints a fresh, scoped, short-lived
    credential from the durable grant and drops it into session state. An expired
    token is never a problem: the agent re-mints on every wake.
  • scoped sub-agents (slide B4)         — read_agent carries ONLY sites.read;
    apply_agent carries ONLY control.write. The coordinator delegates; neither
    sub-agent can do the other's job, and each token is independently revocable.
  • the approval gate == the user-claimed flow (slide B4) — the agent parks at
    ANALYZED until a human confirms an OTP; the resume webhook completes the
    claim and hands the apply_agent its control.write token in one motion.

Tools talk to Part A (app_server.py) over plain HTTP. The auth is the lesson,
not the telemetry — the API returns mock energy numbers.
"""
from __future__ import annotations

import httpx
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

from authmd_client import AuthGrant, AuthMdClient
from config import APP_BASE, APP_NAME, DB_URL, make_model
from work_order import WorkOrderStep, initial_state


# ── before_agent_callback: initialise durable state ─────────────────────────
def initialize_work_order_state(callback_context: CallbackContext) -> None:
    """Ensure all state keys (incl. auth_grants) exist. Idempotent across resumes."""
    state = callback_context.state
    for key, value in initial_state().items():
        if key not in state:
            state[key] = value


# ── before_tool_callback: mint + inject a fresh scoped token (slide B3) ─────
class AuthInjector:
    """Maps each protected tool to the grant + scope it needs, then re-mints a
    fresh credential into state right before the tool runs."""

    # tool name → (grant service key, required scope)
    TOOL_GRANTS = {
        "analyze_site": ("altotech_read", "sites.read"),
        "apply_setpoint": ("altotech_write", "control.write"),
    }

    def __call__(self, tool, args, tool_context: ToolContext):
        spec = self.TOOL_GRANTS.get(getattr(tool, "name", ""))
        if spec is None:
            return None                       # tool needs no credential (e.g. verify)
        service, scope = spec
        state = tool_context.state
        grant_d = state["auth_grants"][service]

        if grant_d["flow"] == "agent_verified":
            # Autonomous wake-time re-mint — present a fresh ID-JAG, get a token.
            client = AuthMdClient(grant_d.get("app_base", APP_BASE))
            try:
                cred = client.acquire(AuthGrant.from_state(service, grant_d),
                                      scopes=[scope])
            finally:
                client.close()
            state["api_token"] = cred.token
            return None

        # user-claimed (control.write): the token is minted by the approval
        # webhook (resume_handler) and placed in state. If it isn't there, the
        # human gate hasn't been passed — refuse rather than act unauthorised.
        if not state.get("api_token"):
            return {"status": "blocked",
                    "message": "No control.write credential — awaiting human "
                               "approval (user-claimed OTP). Cannot apply."}
        return None


_inject = AuthInjector()


# ── read_agent tool — scope: sites.read ──────────────────────────────────────
def analyze_site(tool_context: ToolContext) -> dict:
    """Read the site's energy telemetry and recommend a setpoint, then PAUSE for
    human approval. Uses a freshly-minted sites.read token (injected)."""
    state = tool_context.state
    site = state["site_id"]
    r = httpx.get(f"{APP_BASE}/sites/{site}/energy",
                  headers={"Authorization": f"Bearer {state['api_token']}"},
                  timeout=30)
    r.raise_for_status()
    data = r.json()
    state["analysis"] = data
    state["recommended_setpoint_c"] = data["recommended_setpoint_c"]
    state["current_step"] = WorkOrderStep.ANALYZED
    state["pending_signals"] = ["approval"]
    return {"status": "success",
            "message": (f"Site {site}: now {data['kw_now']} kW at "
                        f"{data['hvac_setpoint_c']}°C. Recommend "
                        f"{data['recommended_setpoint_c']}°C. PAUSED for human "
                        "approval before any control.write.")}


# ── apply_agent tool — scope: control.write ──────────────────────────────────
def apply_setpoint(tool_context: ToolContext) -> dict:
    """Write the approved setpoint via control.write, then COMPLETE. Uses the
    control.write token minted from the user-claimed approval (injected)."""
    state = tool_context.state
    site = state["site_id"]
    setpoint = state["recommended_setpoint_c"]
    r = httpx.post(f"{APP_BASE}/sites/{site}/setpoint", json={"setpoint_c": setpoint},
                   headers={"Authorization": f"Bearer {state['api_token']}"}, timeout=30)
    r.raise_for_status()
    state["applied_setpoint_c"] = setpoint
    state["current_step"] = WorkOrderStep.APPLIED
    state["pending_signals"] = []
    return {"status": "success",
            "message": f"Applied {setpoint}°C to {site} via control.write."}


# ── coordinator tool — no credential needed ──────────────────────────────────
def verify_and_complete(tool_context: ToolContext) -> dict:
    """Confirm the change took and mark the work order COMPLETED."""
    state = tool_context.state
    state["current_step"] = WorkOrderStep.COMPLETED
    state["pending_signals"] = []
    return {"status": "success",
            "message": f"Setpoint {state.get('applied_setpoint_c')}°C verified for "
                       f"{state['site_id']}. Work order complete."}


# ── instructions (TEMPLATES — ADK fills {…} from durable session state) ─────
COORDINATOR_INSTRUCTION = """You are an Energy Work-Order Coordinator for one site.

Current Step: {current_step}
Site: {site_id}
Analysis: {analysis}
Recommended Setpoint (°C): {recommended_setpoint_c}
Pending Signals: {pending_signals}

Follow this state machine EXACTLY. Never skip a step and never claim a later
step happened before its tool ran:

- START: Transfer to the read_agent to analyze the site.
- ANALYZED: You are PAUSED waiting for a human to approve the recommended
  setpoint change. Do NOT call any tool and do NOT apply anything. If asked to
  apply now, politely refuse and explain you are waiting for human approval.
- APPROVED: Transfer to the apply_agent to write the approved setpoint.
- APPLIED: Call verify_and_complete.
- COMPLETED: Confirm the work order is complete.

Stay grounded in the current step shown above."""

READ_INSTRUCTION = """You are a read-only Energy Analyst sub-agent.

Current Step: {current_step}
Site: {site_id}

Your only job: call analyze_site to read telemetry and recommend a setpoint,
then transfer control back to the coordinator. You can ONLY read (sites.read);
you must never attempt to apply a change."""

APPLY_INSTRUCTION = """You are a control sub-agent that writes HVAC setpoints.

Current Step: {current_step}
Site: {site_id}
Recommended Setpoint (°C): {recommended_setpoint_c}

Your only job, and ONLY once the step is APPROVED: call apply_setpoint to write
the approved setpoint, then transfer control back to the coordinator. You hold a
control.write credential only; you cannot read or do anything else."""


def build_read_agent(model) -> Agent:
    return Agent(
        name="read_agent", model=model,
        description="Reads site energy telemetry (scope: sites.read only).",
        instruction=READ_INSTRUCTION,
        tools=[FunctionTool(analyze_site)],
        before_tool_callback=_inject,
    )


def build_apply_agent(model) -> Agent:
    return Agent(
        name="apply_agent", model=model,
        description="Writes HVAC setpoints (scope: control.write only).",
        instruction=APPLY_INSTRUCTION,
        tools=[FunctionTool(apply_setpoint)],
        before_tool_callback=_inject,
    )


def build_root_agent(model=None) -> Agent:
    """The work-order coordinator with its two least-privilege sub-agents."""
    model = model or make_model()
    return Agent(
        name="energy_work_order_coordinator", model=model,
        description="Coordinates a long-running, human-gated energy work order.",
        instruction=COORDINATOR_INSTRUCTION,
        tools=[FunctionTool(verify_and_complete)],
        sub_agents=[build_read_agent(model), build_apply_agent(model)],
        before_agent_callback=initialize_work_order_state,
        before_tool_callback=_inject,
    )


# `adk run` / `adk web` look for a module-level `root_agent`.
root_agent = None


def build_session_service() -> DatabaseSessionService:
    return DatabaseSessionService(db_url=DB_URL)


def build_runner(session_service: DatabaseSessionService | None = None,
                 model=None) -> Runner:
    return Runner(
        app_name=APP_NAME,
        agent=build_root_agent(model),
        session_service=session_service or build_session_service(),
    )

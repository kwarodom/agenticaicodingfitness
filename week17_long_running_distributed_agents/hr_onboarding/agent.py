#!/usr/bin/env python3
"""The HR onboarding agent: a durable state machine wired with Google ADK.

Pieces (straight from the ADK long-running-agents blog, hotel→HR aside):
  • before_agent_callback  — guarantees every state key exists before turn one.
  • instruction            — a TEMPLATE; ADK interpolates {current_step} etc. from
                             durable session state, so the model always sees the
                             exact checkpoint and can't hallucinate progress.
  • tools (ToolContext)    — the ONLY way a transition happens; each writes
                             tool_context.state atomically and the Runner persists
                             the resulting state_delta.
  • sub_agents             — the coordinator delegates IT provisioning to a
                             focused it_agent (narrow prompt + tool set).

This module builds the agents, a DatabaseSessionService, and a Runner. It does
NOT start a server (see server.py) so it can be imported by both the server and
tests.
"""
from __future__ import annotations

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

from config import APP_NAME, DB_URL, make_model
from onboarding_steps import OnboardingStep, initial_state


# ── before_agent_callback: initialise durable state ─────────────────────────
def initialize_onboarding_state(callback_context: CallbackContext) -> None:
    """Ensure all state-machine keys exist. Runs before every turn; `setdefault`
    makes it idempotent across resumes."""
    state = callback_context.state
    for key, value in initial_state().items():
        if key not in state:
            state[key] = value


# ── Coordinator tools (each advances the state machine atomically) ──────────
def send_welcome_packet(name: str, email: str, start_date: str,
                        tool_context: ToolContext) -> dict:
    """Send the welcome packet to a new hire and PAUSE for their signature.

    Args:
        name: New hire's full name.
        email: New hire's email address.
        start_date: ISO start date, e.g. '2026-07-01'.
    """
    state = tool_context.state
    state["new_hire_details"] = {"name": name, "email": email, "start_date": start_date}
    state["current_step"] = OnboardingStep.WELCOME_SENT
    state["pending_signals"] = ["document_signed"]
    return {"status": "success",
            "message": f"Welcome packet sent to {name} ({email}). "
                       "Onboarding paused pending document signatures."}


def check_hardware_delivery(tracking_id: str, tool_context: ToolContext) -> dict:
    """Record the laptop's shipment tracking id and PAUSE until it is delivered.

    Args:
        tracking_id: Carrier tracking number for the new hire's hardware.
    """
    state = tool_context.state
    state.setdefault("new_hire_details", {})["hardware_tracking_id"] = tracking_id
    state["pending_signals"] = ["hardware_delivered"]
    # NOTE: we do NOT advance to HARDWARE_DELIVERED here — that transition is
    # owned by the external delivery webhook, not by the model.
    return {"status": "pending",
            "message": f"Tracking {tracking_id} recorded. Waiting for the "
                       "hardware_delivered signal."}


def send_day_one_schedule(tool_context: ToolContext) -> dict:
    """Send the day-one schedule and COMPLETE the onboarding."""
    state = tool_context.state
    state["current_step"] = OnboardingStep.COMPLETED
    state["pending_signals"] = []
    name = state.get("new_hire_details", {}).get("name", "the new hire")
    return {"status": "success",
            "message": f"Day-one schedule sent to {name}. Onboarding complete."}


# ── it_agent tool ────────────────────────────────────────────────────────────
def provision_software_accounts(username_prefix: str,
                                tool_context: ToolContext) -> dict:
    """Provision corporate software accounts (email, Slack) for the new hire.

    Args:
        username_prefix: Desired corporate username prefix, e.g. 'jdoe'.
    """
    state = tool_context.state
    details = state.setdefault("new_hire_details", {})
    details["corp_username"] = f"{username_prefix}@corp.example.com"
    details["accounts"] = ["email", "slack"]
    state["current_step"] = OnboardingStep.IT_PROVISIONED
    return {"status": "success",
            "message": f"Provisioned email + Slack for {username_prefix}."}


# ── Instructions (TEMPLATES — ADK fills {…} from durable session state) ─────
COORDINATOR_INSTRUCTION = """You are an HR Onboarding Coordinator for a new hire.

Current Step: {current_step}
New Hire Details: {new_hire_details}
Pending Signals: {pending_signals}

Follow this state machine EXACTLY. Never skip a step and never claim a later
step happened before its tool has run:

- START: Ask for the new hire's name, email, and start date if you do not have
  them, then call send_welcome_packet.
- WELCOME_SENT: You are PAUSED waiting for the employee to sign their documents.
  Do NOT call any tool. If asked to move ahead, politely refuse and explain you
  are waiting for the signed contract.
- DOCUMENTS_SIGNED: Delegate IT provisioning by transferring to the it_agent.
- IT_PROVISIONED: Ask for the hardware tracking id, then call
  check_hardware_delivery.
- HARDWARE_DELIVERED: Call send_day_one_schedule.
- COMPLETED: Confirm the onboarding is complete.

Stay grounded in the current step shown above."""

IT_INSTRUCTION = """You are an IT Provisioning agent in an onboarding workflow.

Current Step: {current_step}
New Hire Details: {new_hire_details}

Provision corporate software accounts for the new hire:
1. Derive a sensible corporate username prefix from their name (e.g. 'jdoe' for
   Jane Doe) unless one is provided.
2. Call provision_software_accounts with that prefix.
3. After provisioning, transfer control back to the coordinator."""


def build_it_agent(model) -> Agent:
    return Agent(
        name="it_agent",
        model=model,
        description="Provisions corporate software accounts (email, Slack).",
        instruction=IT_INSTRUCTION,
        tools=[FunctionTool(provision_software_accounts)],
    )


def build_root_agent(model=None) -> Agent:
    """The onboarding coordinator with its IT sub-agent."""
    model = model or make_model()
    return Agent(
        name="hr_onboarding_coordinator",
        model=model,
        description="Coordinates a multi-day employee onboarding workflow.",
        instruction=COORDINATOR_INSTRUCTION,
        tools=[
            FunctionTool(send_welcome_packet),
            FunctionTool(check_hardware_delivery),
            FunctionTool(send_day_one_schedule),
        ],
        sub_agents=[build_it_agent(model)],
        before_agent_callback=initialize_onboarding_state,
    )


# `adk run` / `adk web` look for a module-level `root_agent`.
root_agent = None


def build_session_service() -> DatabaseSessionService:
    """Durable session store — the same object the server and webhooks share."""
    return DatabaseSessionService(db_url=DB_URL)


def build_runner(session_service: DatabaseSessionService | None = None,
                model=None) -> Runner:
    """Wire an agent + durable sessions into a Runner (the agent loop)."""
    return Runner(
        app_name=APP_NAME,
        agent=build_root_agent(model),
        session_service=session_service or build_session_service(),
    )

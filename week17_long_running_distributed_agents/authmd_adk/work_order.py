#!/usr/bin/env python3
"""The energy work-order state machine — durable, with an auth_grants block.

The long-running job: optimise one site's HVAC. The agent READS the site's
telemetry (scope sites.read), then PAUSES for a human to approve the setpoint
change, then APPLIES it (scope control.write). The pause can last days; the
agent must wake with a fresh, least-privilege token each time.

    START ──analyze (read_agent · sites.read)──▶ ANALYZED
                                                    │  (pause: human approval — days)
                                                    │  ◀── webhook /webhooks/approved  (carries the OTP)
                                                    ▼
                                               APPROVED ──apply (apply_agent · control.write)──▶
                                               APPLIED ──verify──▶ COMPLETED

Two ideas from the slides live in the state here:
  • B2 — state['auth_grants'] stores the GRANT (flow + scopes + references),
    never a live token. One grant per service/scope-set, keyed for least
    privilege: the read path is agent-verified; the control path is user-claimed
    (the approval gate IS the claim).
  • durable state machine — behaviour is driven by current_step in persisted
    state, not by replaying chat history.
"""
from __future__ import annotations

from config import APP_BASE


class WorkOrderStep:
    """Explicit, durable workflow state. Driven by persisted `current_step`."""

    START = "START"
    ANALYZED = "ANALYZED"          # paused: waiting for human approval of the change
    APPROVED = "APPROVED"          # set by the /webhooks/approved resume (OTP confirmed)
    APPLIED = "APPLIED"            # setpoint written via control.write
    COMPLETED = "COMPLETED"

    ORDER = [START, ANALYZED, APPROVED, APPLIED, COMPLETED]

    # Steps where the agent is parked waiting on an external signal → refuse to
    # advance on its own (the idle-time safety gate).
    PAUSED_STEPS = {ANALYZED: "approval"}

    @classmethod
    def is_terminal(cls, step: str) -> bool:
        return step == cls.COMPLETED


def initial_grants() -> dict:
    """The durable auth grants (slide B2). NOTE: no tokens — only what the agent
    is allowed to do and how to re-mint. Least privilege per path:

      • altotech_read  → agent-verified, sites.read    (read_agent; autonomous)
      • altotech_write → user-claimed, control.write    (apply_agent; human gate)

    `claim_ref` for the write grant is empty until the approval gate opens.
    """
    return {
        "altotech_read": {
            "flow": "agent_verified", "scopes": ["sites.read"],
            "subject": "energy-ops-bot", "email": "ops@altotech.ai",
            "claim_ref": None, "app_base": APP_BASE,
        },
        "altotech_write": {
            "flow": "user_claimed", "scopes": ["control.write"],
            "subject": None, "email": "facility.manager@altotech.ai",
            "claim_ref": None, "app_base": APP_BASE,
        },
    }


def initial_state(site_id: str = "site-bkk-01") -> dict:
    """State every new work order starts with. Every key the agent instruction
    interpolates MUST exist up front, or ADK's `{var}` templating raises on the
    first turn."""
    return {
        "current_step": WorkOrderStep.START,
        "site_id": site_id,
        "analysis": {},
        "recommended_setpoint_c": None,
        "applied_setpoint_c": None,
        "pending_signals": [],
        "auth_grants": initial_grants(),
    }

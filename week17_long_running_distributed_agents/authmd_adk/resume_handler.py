#!/usr/bin/env python3
"""Event-driven resume — wake a paused work order, and authorise it in one motion.

The work order parks at ANALYZED waiting on a human. Two webhooks drive it:

  • request_approval — fired when the order parks. Starts the user-claimed flow:
    the app emails the facility manager a one-time code and returns a claim_ref,
    which we persist in the durable grant (no token yet — slide B2).

  • approved — fired when the human confirms the OTP. This is slide B4 in one
    move: COMPLETE the claim (mint a fresh control.write token), inject it via
    `state_delta`, and advance the state machine to APPROVED so the apply_agent
    can write the setpoint. Resumption and authorisation happen together.

Both go through Runner.run_async(state_delta=…) — the same durable-resume seam
the ADK blog uses — so the model always wakes already seeing the new state and
can't hallucinate that it got there on its own.
"""
from __future__ import annotations

from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

from authmd_client import AuthMdClient
from config import APP_NAME, APP_BASE
from work_order import WorkOrderStep


class WorkOrderResumeHandler:
    def __init__(self, runner: Runner, user_id: str,
                 session_service: DatabaseSessionService):
        self.runner = runner
        self.user_id = user_id
        self.session_service = session_service

    async def _state(self, session_id: str) -> dict:
        session = await self.session_service.get_session(
            app_name=APP_NAME, user_id=self.user_id, session_id=session_id)
        return dict(session.state) if session else {}

    async def _resume(self, session_id: str, message: str, state_delta: dict) -> str:
        final_text = ""
        async for event in self.runner.run_async(
            user_id=self.user_id, session_id=session_id,
            new_message=types.Content(role="user",
                                      parts=[types.Part.from_text(text=message)]),
            state_delta=state_delta,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if getattr(part, "text", None):
                        final_text = part.text
        return final_text

    # ── start the approval (user-claimed) flow — emails the human an OTP ─────
    async def request_approval(self, session_id: str) -> str:
        """Begin the user-claimed claim for control.write and persist the
        claim_ref in the durable grant. No token is minted yet."""
        state = await self._state(session_id)
        grants = state.get("auth_grants", {})
        write = grants["altotech_write"]
        client = AuthMdClient(write.get("app_base", APP_BASE))
        try:
            claim_ref = client.start_claim(write["email"], write["scopes"])
        finally:
            client.close()
        write["claim_ref"] = claim_ref                 # persist on the durable grant
        return await self._resume(
            session_id,
            message=("An approval request was emailed to the facility manager. "
                     "Stay paused at ANALYZED until they confirm."),
            state_delta={"auth_grants": grants, "pending_signals": ["approval"]},
        )

    # ── human confirmed the OTP → complete the claim + resume to APPROVED ────
    async def approved(self, session_id: str, otp: str) -> str:
        """Webhook: the human approved by confirming the OTP. Complete the claim
        to mint a fresh control.write token, inject it, and advance to APPROVED."""
        state = await self._state(session_id)
        write = state["auth_grants"]["altotech_write"]
        client = AuthMdClient(write.get("app_base", APP_BASE))
        try:
            cred = client.complete_claim(write["claim_ref"], otp)
        finally:
            client.close()
        return await self._resume(
            session_id,
            message="Resume work order: the setpoint change has been approved.",
            # Inject the freshly-minted control.write token alongside the step
            # transition — authorise + resume in one durable write.
            state_delta={
                "current_step": WorkOrderStep.APPROVED,
                "api_token": cred.token,
                "pending_signals": [],
            },
        )

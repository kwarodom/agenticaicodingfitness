#!/usr/bin/env python3
"""Event-driven resume — wake a paused onboarding from an external signal.

A paused onboarding (WELCOME_SENT or IT_PROVISIONED) should NOT be polled. When
the real world produces a signal — the contract is signed, the laptop is
delivered — an external system calls a webhook, and we resume the agent with a
`state_delta` that advances the checkpoint BEFORE the next inference. The model
wakes up already seeing the new step, so it can't hallucinate that it got there
on its own. (ADK blog: Runner.run_async(state_delta=...).)
"""
from __future__ import annotations

from google.adk.runners import Runner
from google.genai import types

from onboarding_steps import OnboardingStep


class OnboardingResumeHandler:
    def __init__(self, runner: Runner, user_id: str):
        self.runner = runner
        self.user_id = user_id

    async def _resume(self, session_id: str, message: str, state_delta: dict) -> str:
        """Apply the state_delta and run one resume turn. Returns the agent's
        final text (for logging / API responses)."""
        final_text = ""
        async for event in self.runner.run_async(
            user_id=self.user_id,
            session_id=session_id,
            new_message=types.Content(
                role="user", parts=[types.Part.from_text(text=message)]
            ),
            state_delta=state_delta,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if getattr(part, "text", None):
                        final_text = part.text
        return final_text

    async def document_signed(self, session_id: str) -> str:
        """Webhook: the new hire signed their contract → DOCUMENTS_SIGNED."""
        return await self._resume(
            session_id,
            message="Resume onboarding: the employee has signed their documents.",
            state_delta={
                "current_step": OnboardingStep.DOCUMENTS_SIGNED,
                "pending_signals": [],
            },
        )

    async def hardware_delivered(self, session_id: str) -> str:
        """Webhook: the laptop was delivered → HARDWARE_DELIVERED."""
        return await self._resume(
            session_id,
            message="Resume onboarding: the new hire's hardware has been delivered.",
            state_delta={
                "current_step": OnboardingStep.HARDWARE_DELIVERED,
                "pending_signals": [],
            },
        )

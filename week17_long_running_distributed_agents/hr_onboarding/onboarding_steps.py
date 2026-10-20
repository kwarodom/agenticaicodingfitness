#!/usr/bin/env python3
"""The HR onboarding state machine.

Onboarding spans days or weeks: a welcome packet goes out, then the workflow
*pauses* waiting on a human (sign the contract) or the physical world (laptop
ships). The agent must never lose its place. We make "its place" an EXPLICIT,
durable state — `current_step` — instead of trying to reconstruct it from a long
conversation history. (ADK blog: "durable state machines".)

    START ──send_welcome_packet──▶ WELCOME_SENT
                                       │  (pause: employee signs contract — days)
                                       │  ◀── webhook /webhooks/document_signed
                                       ▼
                                  DOCUMENTS_SIGNED ──delegate──▶ it_agent
                                       │  provision_software_accounts
                                       ▼
                                  IT_PROVISIONED ──check_hardware_delivery──┐
                                       │  (pause: laptop in transit — days) │
                                       │  ◀── webhook /webhooks/hardware_delivered
                                       ▼
                                  HARDWARE_DELIVERED ──send_day_one_schedule──▶
                                  COMPLETED
"""
from __future__ import annotations


class OnboardingStep:
    """Explicit, durable workflow state. Behaviour is driven by `current_step`
    in persisted session state — NOT by replaying chat history."""

    START = "START"
    WELCOME_SENT = "WELCOME_SENT"            # paused: waiting for signed contract
    DOCUMENTS_SIGNED = "DOCUMENTS_SIGNED"    # set by the document_signed webhook
    IT_PROVISIONED = "IT_PROVISIONED"        # paused: waiting for hardware delivery
    HARDWARE_DELIVERED = "HARDWARE_DELIVERED"  # set by the hardware_delivered webhook
    COMPLETED = "COMPLETED"

    ORDER = [START, WELCOME_SENT, DOCUMENTS_SIGNED, IT_PROVISIONED,
             HARDWARE_DELIVERED, COMPLETED]

    # Steps where the agent is parked waiting on an external signal. At these
    # steps it must refuse to advance on its own (the idle-time safety gate).
    PAUSED_STEPS = {WELCOME_SENT: "document_signed",
                    IT_PROVISIONED: "hardware_delivered"}

    @classmethod
    def is_terminal(cls, step: str) -> bool:
        return step == cls.COMPLETED


def initial_state() -> dict:
    """The state every new onboarding session starts with. All keys the agent
    instruction interpolates MUST exist up front, or ADK's `{var}` templating
    raises on the first turn."""
    return {
        "current_step": OnboardingStep.START,
        "new_hire_details": {},
        "pending_signals": [],
    }

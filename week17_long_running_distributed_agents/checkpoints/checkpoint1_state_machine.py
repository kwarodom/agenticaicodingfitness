#!/usr/bin/env python3
"""CHECKPOINT 1 — A durable state machine for a long-running work-order.

Goal: stop steering a multi-day workflow with conversation history. Instead,
give the agent an EXPLICIT state machine that lives in session state, inject the
current state into its instruction, and make every tool advance the state
*atomically* via ToolContext.state. This is the foundation the ADK blog calls
"durable state machines": the model always sees exact workflow status, so it
can pause for three days and resume without hallucinating intermediate steps.

What this demonstrates (offline, no Gemini needed):
  Part 1 — the tools + state-interpolated instruction + initializer callback.
  Part 2 — the REAL ADK Agent wiring (shown only if google-adk is installed).
  Part 3 — a pure-Python simulation that drives the tools through
           OPEN → DIAGNOSED → AWAITING_PART, proving each transition is atomic.

(Week 17 · long-running agents · ADK blog: "durable state machines")

Run:  python week17/checkpoints/checkpoint1_state_machine.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import config.py
from config import MODEL, WorkOrderStep, banner, have_adk, step


# ════════════════════════════════════════════════════════════════════════════
# Part 1 — State-driven instruction + tools.
#
# The instruction is a TEMPLATE. ADK fills {current_step} etc. from session
# state on every turn (the blog interpolates state straight into the prompt),
# so the model never has to reconstruct "where am I?" from history.
# ════════════════════════════════════════════════════════════════════════════
INSTRUCTION = """You are the Hotel Maintenance Coordinator for a 200-room hotel.

Work-order: {work_order_id}
Current Step: {current_step}
Fault details: {fault_details}
Pending signals: {pending_signals}

Follow this state machine EXACTLY — never skip a step:
1. OPEN            → Ask for the room and symptom, then call 'diagnose_fault'.
2. DIAGNOSED       → If a part is needed, call 'request_part' (this PAUSES the
                     work-order). Otherwise go straight to 'confirm_repair'.
3. AWAITING_PART   → You are PAUSED waiting for a vendor delivery. Do NOT call
                     any tool. Tell the user the work-order is waiting on a part.
4. PART_DELIVERED  → The part arrived. Call 'confirm_repair'.
5. REPAIRED        → Call 'close_work_order'.
6. CLOSED          → Confirm the work-order is complete.

Stay grounded in your current step. Never claim a repair happened before
'confirm_repair' has run."""


def initialize_work_order(callback_context) -> None:
    """`before_agent_callback`: make sure every state key exists before the
    first turn. ADK passes a CallbackContext whose `.state` is the durable
    session dict (blog: initialize_onboarding_state)."""
    state = callback_context.state
    state.setdefault("work_order_id", "WO-1042")
    state.setdefault("current_step", WorkOrderStep.OPEN)
    state.setdefault("fault_details", {})
    state.setdefault("pending_signals", [])


# Each tool mutates state.* and is the ONLY way a transition happens. If the
# container crashes the instant after a tool returns, the new step is already
# persisted — that is what makes the workflow durable.
def diagnose_fault(room: str, symptom: str, part_needed: str, tool_context) -> dict:
    """Record a diagnosis for the work-order and move it to DIAGNOSED.

    Args:
        room: Room id, e.g. 'R305'.
        symptom: What the guest reported, e.g. 'HVAC not cooling'.
        part_needed: SKU of any replacement part, or '' if none.

    Returns:
        A status dict describing the transition.
    """
    state = tool_context.state
    state["fault_details"] = {"room": room, "symptom": symptom, "part_needed": part_needed}
    state["current_step"] = WorkOrderStep.DIAGNOSED
    return {"status": "success", "current_step": state["current_step"],
            "part_needed": part_needed}


def request_part(sku: str, tool_context) -> dict:
    """Order a replacement part from the vendor and PAUSE the work-order at
    AWAITING_PART until a delivery signal arrives.

    Args:
        sku: The part SKU to order, e.g. 'COMP-24K-BTU'.
    """
    state = tool_context.state
    state["fault_details"]["ordered_sku"] = sku
    state["current_step"] = WorkOrderStep.AWAITING_PART
    state["pending_signals"] = ["part_delivered"]   # the signal we wait for
    return {"status": "paused", "current_step": state["current_step"],
            "waiting_for": "part_delivered"}


def confirm_repair(tool_context) -> dict:
    """Mark the physical repair complete and move to REPAIRED."""
    state = tool_context.state
    state["current_step"] = WorkOrderStep.REPAIRED
    return {"status": "success", "current_step": state["current_step"]}


def close_work_order(tool_context) -> dict:
    """Verify and close the work-order (CLOSED)."""
    state = tool_context.state
    state["current_step"] = WorkOrderStep.CLOSED
    state["pending_signals"] = []
    return {"status": "success", "current_step": state["current_step"]}


# ════════════════════════════════════════════════════════════════════════════
# Part 2 — The REAL ADK Agent (construction only; no Gemini call).
# ════════════════════════════════════════════════════════════════════════════
def build_agent():
    """Wire the coordinator the way you would for production. Importable only if
    google-adk is installed; the lesson (Part 3) runs regardless."""
    from google.adk.agents import Agent
    from google.adk.tools import FunctionTool

    return Agent(
        name="hotel_maintenance_coordinator",
        model=MODEL,
        description="Coordinates long-running hotel maintenance work-orders.",
        instruction=INSTRUCTION,
        tools=[
            FunctionTool(diagnose_fault),
            FunctionTool(request_part),
            FunctionTool(confirm_repair),
            FunctionTool(close_work_order),
        ],
        before_agent_callback=initialize_work_order,
    )


# ════════════════════════════════════════════════════════════════════════════
# Part 3 — Offline simulation: drive the tools and watch state advance.
# ════════════════════════════════════════════════════════════════════════════
class FakeToolContext:
    """Stands in for ADK's ToolContext: just exposes a mutable `.state` dict.
    Lets us exercise the tool functions with zero dependencies."""

    def __init__(self, state: dict):
        self.state = state


def simulate():
    state: dict = {}
    initialize_work_order(FakeToolContext(state))
    step(f"initialized → current_step = {state['current_step']}")

    ctx = FakeToolContext(state)
    r = diagnose_fault("R305", "HVAC not cooling", "COMP-24K-BTU", ctx)
    step(f"diagnose_fault  → {r['current_step']} (part_needed={r['part_needed']})")

    r = request_part("COMP-24K-BTU", ctx)
    step(f"request_part    → {r['current_step']} (waiting_for={r['waiting_for']})")

    print()
    step("Work-order is now PAUSED. The vendor ships the compressor; this could")
    step("take days. Checkpoint 2 shows the state surviving a process restart;")
    step("Checkpoint 3 resumes it from a delivery webhook.")
    assert state["current_step"] == WorkOrderStep.AWAITING_PART
    return state


def main():
    banner("CP1 · Durable state machine for a long-running work-order")

    if have_adk():
        agent = build_agent()
        step("Real ADK Agent constructed:")
        step(f"   name  = {agent.name}")
        step(f"   model = {agent.model}")
        step(f"   tools = {[t.name for t in agent.tools]}")
        step("   before_agent_callback = initialize_work_order")
    else:
        step("google-adk not installed — skipping live Agent wiring "
             "(`pip install google-adk` to see it).")
        step("The state-machine lesson below runs regardless.")

    print()
    step("Simulating the workflow by driving the tools directly:")
    simulate()
    print()
    step("KEY IDEA: behaviour is driven by `current_step` in state, never by")
    step("re-reading the chat log. That is what lets the agent pause for days.")


if __name__ == "__main__":
    main()

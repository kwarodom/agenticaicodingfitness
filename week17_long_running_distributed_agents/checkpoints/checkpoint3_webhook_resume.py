#!/usr/bin/env python3
"""CHECKPOINT 3 — Event-driven resume: wake the agent from a webhook.

Goal: don't POLL a paused work-order ("is the part here yet? is it here yet?").
Let the *outside world* wake it. When the vendor's logistics system delivers the
compressor, it POSTs to a webhook; the handler applies a state_delta that flips
the work-order to PART_DELIVERED and resumes the agent at the right checkpoint.

In ADK the resume is `Runner.run_async(..., state_delta={...})`: the delta is
applied to durable session state BEFORE the next inference, so the model wakes
up already seeing the new checkpoint in its prompt — it cannot hallucinate that
the part was always here. We show that real call (guarded), and run an offline
FastAPI TestClient round-trip so you can fire the webhook with no live server.

What this demonstrates:
  1. A FastAPI webhook  POST /webhooks/part_delivered.
  2. A resume handler that applies the state_delta to the durable store (CP2).
  3. The work-order moving AWAITING_PART → PART_DELIVERED, fired in-process.

(Week 17 · long-running agents · ADK blog: Runner.run_async(state_delta=...))

Run:  python week17/checkpoints/checkpoint3_webhook_resume.py
      # or serve it for real:  uvicorn checkpoint3_webhook_resume:app --reload
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # config.py
sys.path.insert(0, str(Path(__file__).resolve().parent))      # checkpoint2_*
from config import WorkOrderStep, banner, step

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError as e:
    print(f"⚠️  Missing dependency: {e.name}")
    print("   pip install fastapi pydantic 'uvicorn[standard]'")
    sys.exit(1)

from checkpoint2_durable_sessions import DurableWorkOrderStore, WO_ID


# ── The webhook payload an external system sends ────────────────────────────
class PartDeliveredPayload(BaseModel):
    session_id: str          # which work-order to wake (blog: user_id+session_id)
    sku: str                 # which part was delivered
    tracking_id: str = ""


# ── The resume handler: apply a state_delta, then (would) resume the agent ──
class WorkOrderResumeHandler:
    """Mirrors the blog's OnboardingResumeHandler. The durable state transition
    is the real, testable effect; the actual agent inference is shown as the
    guarded ADK call it would be in production."""

    def __init__(self, store: DurableWorkOrderStore):
        self.store = store

    def receive_part_delivered(self, session_id: str, sku: str, tracking_id: str) -> dict:
        state = self.store.get_session(session_id)
        if state is None:
            return {"status": "error", "reason": f"no work-order {session_id}"}
        if state["current_step"] != WorkOrderStep.AWAITING_PART:
            return {"status": "ignored", "reason": f"not paused (at {state['current_step']})"}

        # --- the state_delta (atomic checkpoint applied BEFORE next inference) ---
        state_delta = {
            "current_step": WorkOrderStep.PART_DELIVERED,
            "pending_signals": [],
        }
        state.update(state_delta)
        state["fault_details"]["delivered_sku"] = sku
        state["fault_details"]["tracking_id"] = tracking_id
        self.store.save_state(session_id, state)

        self._resume_agent_inference(session_id, state_delta)
        return {"status": "success", "current_step": state["current_step"]}

    @staticmethod
    def _resume_agent_inference(session_id: str, state_delta: dict) -> None:
        """In production this is where ADK re-enters the agent loop. Shown as the
        real call; we don't execute it (no Gemini/credentials in this demo)."""
        step("   would now resume the agent (real ADK call):")
        print(
            '        async for event in runner.run_async(\n'
            f'            user_id="front_desk", session_id="{session_id}",\n'
            '            new_message=types.Content(role="user", parts=[types.Part\n'
            '                .from_text(text="Resume: replacement part delivered.")]),\n'
            f'            state_delta={state_delta},\n'
            '        ): ...   # model wakes already seeing current_step=PART_DELIVERED'
        )


# ── FastAPI app (importable by uvicorn; also driven offline below) ──────────
app = FastAPI(title="Hotel Maintenance — resume webhooks")
_store = DurableWorkOrderStore()
_handler = WorkOrderResumeHandler(_store)


@app.post("/webhooks/part_delivered")
def part_delivered(payload: PartDeliveredPayload) -> dict:
    """Wakes a paused work-order when the vendor delivers its replacement part."""
    return _handler.receive_part_delivered(
        payload.session_id, payload.sku, payload.tracking_id
    )


def main():
    banner("CP3 · Event-driven resume — wake the agent from a delivery webhook")

    # Ensure a paused work-order exists (CP2 creates it; re-seed if needed).
    if _store.get_session(WO_ID) is None:
        step(f"No paused {WO_ID} found — run checkpoint2 first, or seeding one now.")
        _store.create_session(WO_ID, {
            "work_order_id": WO_ID, "current_step": WorkOrderStep.AWAITING_PART,
            "fault_details": {"room": "R305", "part_needed": "COMP-24K-BTU"},
            "pending_signals": ["part_delivered"],
        })

    before = _store.get_session(WO_ID)["current_step"]
    step(f"before webhook: {WO_ID} is at {before}")

    # Fire the webhook the way the vendor's logistics system would — but offline,
    # via FastAPI's TestClient (no server to start).
    from fastapi.testclient import TestClient
    client = TestClient(app)
    step("POST /webhooks/part_delivered  {sku: COMP-24K-BTU, tracking_id: 1Z999}")
    resp = client.post("/webhooks/part_delivered", json={
        "session_id": WO_ID, "sku": "COMP-24K-BTU", "tracking_id": "1Z999",
    })
    print()
    step(f"webhook response: {resp.json()}")

    after = _store.get_session(WO_ID)["current_step"]
    step(f"after webhook:  {WO_ID} is at {after}")
    assert after == WorkOrderStep.PART_DELIVERED
    print()
    step("KEY IDEA: external events drive resumption (push, not poll). The agent")
    step("stays scaled-to-zero until something real happens. Next: delegation —")
    step("first to an in-house sub-agent (CP4), then across a service boundary")
    step("to the vendor's own agent via A2A (CP5).")


if __name__ == "__main__":
    main()

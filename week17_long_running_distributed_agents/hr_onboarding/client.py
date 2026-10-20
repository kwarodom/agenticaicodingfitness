#!/usr/bin/env python3
"""Drive the long-running onboarding service through a full multi-day journey.

Start the server first (see server.py), then run this. It exercises the durable
pause/resume lifecycle end to end and prints the checkpoint after every step, so
you can watch the workflow park at WELCOME_SENT (waiting on a signature) and
IT_PROVISIONED (waiting on hardware), then get woken by webhooks.

    .venv/bin/python week17/hr_onboarding/server.py        # terminal 1
    .venv/bin/python week17/hr_onboarding/client.py        # terminal 2

The webhook calls simulate days passing — in reality they'd arrive from a
DocuSign callback and a carrier's delivery event, possibly long after the
service restarted.
"""
from __future__ import annotations

import sys
import uuid

import httpx

BASE = "http://127.0.0.1:8077"


def show(label: str, data: dict) -> None:
    reply = data.get("reply")
    print(f"\n── {label} ──")
    if reply:
        print(f"   agent: {reply.strip()[:300]}")
    print(f"   step={data['current_step']}  "
          f"waiting_for={data.get('paused_waiting_for')}  "
          f"pending={data.get('pending_signals')}")
    if data.get("new_hire_details"):
        print(f"   details={data['new_hire_details']}")


def main() -> None:
    sid = f"onb-{uuid.uuid4().hex[:8]}"
    with httpx.Client(base_url=BASE, timeout=120) as c:
        try:
            c.get("/healthz").raise_for_status()
        except Exception:
            print(f"❌ server not reachable at {BASE} — start server.py first.")
            sys.exit(1)

        print(f"Onboarding session: {sid}")
        show("1. create session", c.post("/onboard", json={"session_id": sid}).json())

        # START → the coordinator collects details and sends the welcome packet.
        show("2. kick off (provide new-hire details)", c.post("/chat", json={
            "session_id": sid,
            "message": ("Start onboarding for Jane Doe, email jane@example.com, "
                        "start date 2026-07-01."),
        }).json())

        # WELCOME_SENT → try to skip ahead; the agent must REFUSE (safety gate).
        show("3. try to skip the wait (should refuse)", c.post("/chat", json={
            "session_id": sid,
            "message": "Can we skip the signature and provision IT accounts now?",
        }).json())

        # …days pass… contract signed → webhook resumes + delegates to it_agent.
        show("4. webhook: document_signed",
             c.post("/webhooks/document_signed", json={"session_id": sid}).json())

        # IT_PROVISIONED → provide the hardware tracking id.
        show("5. provide hardware tracking id", c.post("/chat", json={
            "session_id": sid,
            "message": "The laptop shipped, tracking id 1Z999AA10123456784.",
        }).json())

        # …days pass… laptop delivered → webhook resumes to day-one schedule.
        show("6. webhook: hardware_delivered",
             c.post("/webhooks/hardware_delivered", json={"session_id": sid}).json())

        final = c.get(f"/status/hr_coordinator/{sid}").json()
        show("7. final status", final)
        print("\n" + ("✅ onboarding COMPLETED" if final["complete"]
                      else f"⚠️  ended at {final['current_step']}"))


if __name__ == "__main__":
    main()

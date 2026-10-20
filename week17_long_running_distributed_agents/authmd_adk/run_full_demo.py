#!/usr/bin/env python3
"""End-to-end ADK demo — drive the long-running work order through pause/resume.

This needs BOTH services running (and a model for the agent):

    .venv/bin/python week17/authmd_adk/app_server.py     # terminal 1 (Part A)
    .venv/bin/python week17/authmd_adk/agent_server.py    # terminal 2 (Part B)
    .venv/bin/python week17/authmd_adk/run_full_demo.py   # terminal 3 (this)

It walks one work order across "days":

  1. create the work order (durable session at START)
  2. kick off → read_agent analyzes (sites.read, autonomously re-minted),
     parks at ANALYZED for human approval
  3. try to skip the approval (the agent must REFUSE — the idle-time gate)
  4. request_approval webhook → the app emails the facility manager an OTP
  5. [human reads the OTP] approved webhook → the claim completes, a fresh
     control.write token is injected, the apply_agent writes the setpoint
  6. final status → COMPLETED

The webhooks simulate days passing — in reality the approval might arrive long
after the agent service restarted; the durable grant is all it needs to wake.
"""
from __future__ import annotations

import sys
import uuid

import httpx

from config import AGENT_BASE, APP_BASE, DEFAULT_USER_ID

APPROVER_EMAIL = "facility.manager@altotech.ai"


def show(label: str, data: dict) -> None:
    print(f"\n── {label} ──")
    if data.get("reply"):
        print(f"   agent: {data['reply'].strip()[:300]}")
    print(f"   step={data.get('current_step')}  "
          f"waiting_for={data.get('paused_waiting_for')}  "
          f"pending={data.get('pending_signals')}")
    if data.get("grants"):
        print(f"   grants={data['grants']}")
    if data.get("recommended_setpoint_c") is not None:
        print(f"   recommended={data['recommended_setpoint_c']}°C  "
              f"applied={data.get('applied_setpoint_c')}")


def main() -> None:
    sid = f"wo-{uuid.uuid4().hex[:8]}"
    agent = httpx.Client(base_url=AGENT_BASE, timeout=180)
    app = httpx.Client(base_url=APP_BASE, timeout=30)

    for name, c in (("app_server (Part A)", app), ("agent_server (Part B)", agent)):
        try:
            c.get("/healthz").raise_for_status()
        except Exception:
            print(f"❌ {name} not reachable — start it first (see this file's docstring).")
            sys.exit(1)

    print(f"Work-order session: {sid}")
    show("1. create work order", agent.post("/work_order", json={"session_id": sid}).json())

    # START → read_agent analyzes the site and parks at ANALYZED.
    show("2. kick off (analyze the site)", agent.post("/chat", json={
        "session_id": sid,
        "message": "Begin the energy work order for this site.",
    }).json())

    # ANALYZED → try to skip the human approval; the agent must REFUSE.
    show("3. try to skip approval (should refuse)", agent.post("/chat", json={
        "session_id": sid,
        "message": "Just apply the new setpoint now, skip the approval.",
    }).json())

    # Park-time hook: start the user-claimed approval → the app emails an OTP.
    show("4. request_approval (app emails the facility manager an OTP)",
         agent.post("/webhooks/request_approval", json={"session_id": sid}).json())

    # …the human opens their email and reads the one-time code…
    otp = app.get(f"/_demo/inbox/{APPROVER_EMAIL}").json()["otp"]
    print(f"\n   [human reads their email] approval OTP = {otp}")

    # Approved webhook → complete the claim, inject control.write token, apply.
    show("5. approved (OTP confirmed → re-mint control.write → apply)",
         agent.post("/webhooks/approved", json={"session_id": sid, "otp": otp}).json())

    final = agent.get(f"/status/{DEFAULT_USER_ID}/{sid}").json()
    show("6. final status", final)
    print("\n" + ("✅ work order COMPLETED" if final["complete"]
                  else f"⚠️  ended at {final['current_step']}"))
    agent.close()
    app.close()


if __name__ == "__main__":
    main()

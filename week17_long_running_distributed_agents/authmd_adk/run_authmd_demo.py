#!/usr/bin/env python3
"""OFFLINE walk-through of the whole auth.md protocol — NO ADK, NO model.

This is the "it just works" demo. It boots Part A (the agent-ready AltoTech
Energy API) in a background thread and drives it with the Part B authmd_client,
so you can watch every step of the slides end to end with real crypto and zero
external network:

  1. Discovery        auth.md → PRM → AS metadata (the agent_auth block)
  2. Agent-verified   sign an ID-JAG → token → read the protected API
  3. Wake-time re-mint acquire() again → a brand-new token (the grant is durable)
  4. User-claimed      start_claim → human reads OTP → complete_claim → token
  5. Least privilege   the sites.read token is REFUSED on control.write (403)
  6. Apply             the control.write token writes the setpoint
  7. Revocation        revoke the token → the next call is 401
  8. Audit             dump the app's audit log

    .venv/bin/python week17/authmd_adk/run_authmd_demo.py

Only needs: fastapi, uvicorn, httpx, pyjwt[crypto]. No google-adk, no API key.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx
import uvicorn

from app_server import app as energy_api
from authmd_client import AuthGrant, AuthMdClient
from config import APP_BASE, APP_HOST, APP_PORT


def _boot_app() -> None:
    server = uvicorn.Server(uvicorn.Config(energy_api, host=APP_HOST, port=APP_PORT,
                                           log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(50):
        try:
            if httpx.get(f"{APP_BASE}/healthz", timeout=1).status_code == 200:
                return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("app_server did not come up")


def hr(title: str) -> None:
    print(f"\n{'─' * 70}\n{title}\n{'─' * 70}")


def main() -> None:
    _boot_app()
    client = AuthMdClient(APP_BASE)
    http = httpx.Client(base_url=APP_BASE, timeout=10)
    site = "site-bkk-01"

    # 1 ── discovery ─────────────────────────────────────────────────────────
    hr("1. DISCOVERY — auth.md → PRM → AS metadata")
    print(http.get("/auth.md").text)
    meta = client.discover()
    print("agent_auth block:")
    for k, v in meta.items():
        print(f"   {k}: {v}")

    # show the 401 discovery hint on the protected resource
    unauth = http.get(f"/sites/{site}/energy")
    hr("   the 401 hint (no token) — WWW-Authenticate points back at the PRM")
    print(f"   status={unauth.status_code}  "
          f"WWW-Authenticate={unauth.headers.get('WWW-Authenticate')}")

    # 2 ── agent-verified ──────────────────────────────────────────────────────
    hr("2. AGENT-VERIFIED — sign an ID-JAG, exchange it for a scoped token")
    read_grant = AuthGrant(service="altotech_read", flow="agent_verified",
                           scopes=["sites.read"], subject="energy-ops-bot",
                           email="ops@altotech.ai")
    cred = client.acquire(read_grant)
    print(f"   token={cred.token[:28]}…  scopes={cred.scopes}  expires_in={cred.expires_in}s")
    energy = http.get(f"/sites/{site}/energy",
                      headers={"Authorization": cred.bearer()}).json()
    print(f"   GET /sites/{site}/energy → {energy}")

    # 3 ── wake-time re-mint ──────────────────────────────────────────────────
    hr("3. WAKE-TIME RE-MINT — the durable grant yields a NEW token on demand")
    cred2 = client.acquire(read_grant)
    print(f"   first token : {cred.token[:28]}…")
    print(f"   re-minted   : {cred2.token[:28]}…")
    print(f"   different?  → {cred.token != cred2.token}  (an expired token is never a problem)")

    # 4 ── user-claimed (the human approval gate) ─────────────────────────────
    hr("4. USER-CLAIMED — the approval gate: human confirms an OTP")
    claim_ref = client.start_claim("facility.manager@altotech.ai", ["control.write"])
    print(f"   started claim, app emailed an OTP. claim_ref={claim_ref[:28]}…")
    otp = http.get("/_demo/inbox/facility.manager@altotech.ai").json()["otp"]
    print(f"   [human reads their email] OTP = {otp}")
    write_cred = client.complete_claim(claim_ref, otp)
    print(f"   complete_claim → control.write token={write_cred.token[:28]}… scopes={write_cred.scopes}")

    # 5 ── least privilege ────────────────────────────────────────────────────
    hr("5. LEAST PRIVILEGE — the sites.read token cannot write")
    denied = http.post(f"/sites/{site}/setpoint", json={"setpoint_c": 25.5},
                       headers={"Authorization": cred2.bearer()})
    print(f"   read token on POST /setpoint → {denied.status_code} {denied.json()}")

    # 6 ── apply with the right token ─────────────────────────────────────────
    hr("6. APPLY — the control.write token writes the setpoint")
    applied = http.post(f"/sites/{site}/setpoint",
                        json={"setpoint_c": energy["recommended_setpoint_c"]},
                        headers={"Authorization": write_cred.bearer()}).json()
    print(f"   POST /setpoint → {applied}")

    # 7 ── revocation ─────────────────────────────────────────────────────────
    hr("7. REVOCATION — revoke the token, the next call is 401")
    print(f"   revoke → {client.revoke(token=write_cred.token)}")
    after = http.post(f"/sites/{site}/setpoint", json={"setpoint_c": 24.0},
                      headers={"Authorization": write_cred.bearer()})
    print(f"   POST /setpoint after revoke → {after.status_code}")

    # 8 ── audit ──────────────────────────────────────────────────────────────
    hr("8. AUDIT LOG — one event per state change")
    for ev in http.get("/_demo/audit").json()["events"]:
        print(f"   {ev}")

    client.close()
    http.close()
    print("\n✅ auth.md protocol walk-through complete (offline, no model).")


if __name__ == "__main__":
    main()

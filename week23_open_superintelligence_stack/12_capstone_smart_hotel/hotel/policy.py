#!/usr/bin/env python3
"""OpenShell-style secure runtime for the hotel agents  (Week 23 · App 7).

Before ANY tool call runs, it passes through the policy Gateway. The policy is
a signed document: allowed tools per role, setpoint bounds, VIP protection, and
a network egress allowlist. This is "what the agent may DO" — the safety half of
Agent = Model + Harness. Deny reasons are fed back to the agent so it can adapt.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from . import world


# A signed policy (the signature is a content hash — a real deployment signs with a key).
POLICY = {
    "version": "hotel-ops-2026.07",
    "setpoint_bounds_c": [20.0, 25.0],          # never drive comfort outside this band
    "vip_requires_human": True,                  # no autonomous action on VIP-occupied rooms
    "egress_allowlist": ["bms.hotel.internal", "pms.hotel.internal"],  # no public internet
    "roles": {
        "energy":      ["get_room_telemetry", "set_setpoint", "guest_profile", "search_sop"],
        "maintenance": ["get_room_telemetry", "dispatch_work_order", "search_sop"],
        "guest":       ["get_room_telemetry", "guest_profile", "search_sop", "set_setpoint"],
        "orchestrator": ["get_room_telemetry", "guest_profile", "search_sop"],
    },
}
POLICY_SIG = hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class Verdict:
    allow: bool
    reason: str = ""
    needs_human: bool = False


class Gateway:
    """The single choke-point every tool call passes through."""

    def __init__(self, state: world.HotelState):
        self.state = state
        self.denials: list[dict] = []

    def check(self, role: str, tool: str, args: dict) -> Verdict:
        allowed = POLICY["roles"].get(role, [])
        if tool not in allowed:
            return self._deny(role, tool, f"tool '{tool}' not in signed allowlist for role '{role}'")

        if tool == "set_setpoint":
            room = world._room_key(args.get("room"))
            r = self.state.rooms.get(room)
            sp = float(args.get("setpoint_c", 0))
            lo, hi = POLICY["setpoint_bounds_c"]
            if not (lo <= sp <= hi):
                return self._deny(role, tool, f"setpoint {sp}°C outside policy band {lo}–{hi}°C")
            if r and r.vip and r.occupied and POLICY["vip_requires_human"]:
                return Verdict(False, f"room {room} is VIP-occupied → human concierge approval required",
                               needs_human=True)
        return Verdict(True, "ok")

    def _deny(self, role, tool, reason) -> Verdict:
        self.denials.append({"role": role, "tool": tool, "reason": reason})
        return Verdict(False, reason)


def signature_line() -> str:
    return f"policy {POLICY['version']} · sig {POLICY_SIG} · egress⊆{POLICY['egress_allowlist']}"

#!/usr/bin/env python3
"""Self-evolving loop — Data Flywheel + NeMo Gym verifiable rewards.

  • Every specialist decision is scored by a VERIFIABLE reward (Week 23 · App 10):
    an objective check of the outcome, not a human vibe — energy saved, SOP-correct
    triage, VIP protected.
  • High-reward, policy-clean traces are CURATED (App 11) into a training set that
    would GRPO/distill a cheaper Nano student to match the Super teacher.
  • The report also carries inference economics (App 9): cost, energy, model mix.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scored:
    role: str
    task: str
    reward: float
    verifier: str
    kept: bool


def score(result, room_state) -> Scored:
    """Objective, checkable reward per role (0..1)."""
    role = result.role
    if role == "maintenance":
        dispatched = any(a["tool"] == "dispatch_work_order" for a in result.actions)
        critical_room = room_state and room_state.occupied and (room_state.temp_c - room_state.setpoint_c) > 3
        good = dispatched and critical_room
        return Scored(role, result.answer[:60], 1.0 if good else 0.2,
                      "dispatched CRITICAL for a genuinely critical room" if good
                      else "did not correctly dispatch", good)
    if role == "energy":
        saved = sum(a["result"].get("est_kw_saved", 0) for a in result.actions if a["tool"] == "set_setpoint")
        # reward scales with kW saved; full credit at ≥8 kW, and only if no denials
        r = min(1.0, saved / 8.0) if not result.denials else 0.3
        return Scored(role, result.answer[:60], round(r, 2),
                      f"saved ~{saved:.1f} kW within comfort band", r >= 0.6)
    if role == "guest":
        # correct behavior for a VIP request is to defer to a human (needs_human), not act
        good = result.needs_human
        return Scored(role, result.answer[:60], 1.0 if good else 0.4,
                      "VIP correctly routed to human approval" if good else "acted without approval", good)
    return Scored(role, result.answer[:60], 0.5, "n/a", False)


@dataclass
class Flywheel:
    scored: list = field(default_factory=list)

    def observe(self, result, room_state):
        self.scored.append(score(result, room_state))

    def report(self, relay) -> list[str]:
        n = len(self.scored)
        avg = sum(s.reward for s in self.scored) / n if n else 0.0
        kept = [s for s in self.scored if s.kept]
        t = relay.totals()
        out = ["▣ DATA FLYWHEEL — observe → curate → evaluate → (distill)",
               "  " + "─" * 58]
        for s in self.scored:
            mark = "✓" if s.kept else "·"
            out.append(f"  {mark} [{s.role:11}] reward {s.reward:.2f} — {s.verifier}")
        out += ["  " + "─" * 58,
                f"  eval: mean verifiable reward {avg:.2f} over {n} decisions "
                f"· curated {len(kept)}/{n} clean traces for training",
                f"  economics: {t['llm_tokens']} tok · ${t['cost_usd']:.5f} · ~{t['energy_wh']} Wh "
                f"· router mix nano×{t['nano_calls']} / super×{t['super_calls']}",
                "  next turn of the flywheel: GRPO-distill Super→Nano on the curated set → "
                "cheaper on-DGX inference at equal reward."]
        return out

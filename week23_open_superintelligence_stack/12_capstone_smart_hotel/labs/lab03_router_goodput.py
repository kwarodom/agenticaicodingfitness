#!/usr/bin/env python3
"""Lab 03 — router right-sizing vs all-Super: same rewards, different bill.

Chapter 5 SHOWS the router's savings; this lab lets you A/B it. Run the full
morning sweep twice with the deterministic SimBrain (so outcomes are identical):

  A. the normal Relay — Nano for simple work, Super for hard reasoning;
  B. a hobbled Relay whose route() always answers Super.

Then score BOTH runs with the flywheel's verifiable rewards and compute the
only honest metric: GOODPUT — cost per successful (high-reward, curated)
decision. Equal reward at lower cost is the whole right-sizing argument.

Runs offline in <1 s; no endpoint needed (the economics table is the same one
the REAL path uses).
"""
from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from hotel.agents import Orchestrator  # noqa: E402
from hotel.brain import SimBrain  # noqa: E402
from hotel.flywheel import Flywheel  # noqa: E402
from hotel.policy import Gateway  # noqa: E402
from hotel.relay import MODELS, SUPER, Relay  # noqa: E402
from hotel.world import Tools, fresh_state  # noqa: E402


class AllSuperRelay(Relay):
    """A router with the optimization switched OFF — every request goes to Super."""

    def route(self, difficulty: str) -> str:
        return SUPER


def run_sweep(relay: Relay) -> dict:
    """One full morning sweep on a fresh hotel; returns economics + reward stats."""
    state = fresh_state()
    orch = Orchestrator(state, Tools(state), Gateway(state), relay, SimBrain())
    fly = Flywheel()
    for _role, rid, res in orch.morning_brief()["results"]:
        fly.observe(res, state.rooms.get(rid))
    t = relay.totals()
    kept = sum(1 for s in fly.scored if s.kept)
    mean = sum(s.reward for s in fly.scored) / max(1, len(fly.scored))
    t.update(decisions=len(fly.scored), kept=kept, mean_reward=round(mean, 2),
             goodput_usd=round(t["cost_usd"] / max(1, kept), 6))
    return t


def row(label: str, t: dict) -> None:
    print(f"  {label:12} nano×{t['nano_calls']:2} super×{t['super_calls']:2} · "
          f"{t['llm_tokens']:5} tok · ${t['cost_usd']:.5f} · ~{t['energy_wh']:.2f} Wh · "
          f"reward {t['mean_reward']:.2f} ({t['kept']}/{t['decisions']} curated) · "
          f"${t['goodput_usd']:.6f}/good decision")


def main() -> None:
    print("━" * 64)
    print("  LAB 03 — right-sizing A/B: goodput, the only honest metric")
    print("━" * 64)
    prices = " · ".join(f"{m.split(':')[0]} ${v['mtok_usd']}/Mtok @ {v['watts']}W"
                        for m, v in MODELS.items())
    print(f"\n▣ NIM price/energy table (illustrative): {prices}\n")

    print("▣ RUN A — normal router (Nano for simple, Super for hard)")
    a = run_sweep(Relay())
    row("A router", a)

    print("\n▣ RUN B — optimization OFF (every call → Super)")
    b = run_sweep(AllSuperRelay())
    row("B all-Super", b)

    print("\n▣ VERDICT")
    if a["mean_reward"] == b["mean_reward"] and a["kept"] == b["kept"]:
        print("  ✓ identical verifiable rewards — the cheap router lost NOTHING on outcome.")
    else:
        print("  ◈ rewards differ — inspect which decisions changed before trusting the router.")
    save = b["cost_usd"] - a["cost_usd"]
    pct = 100 * save / b["cost_usd"] if b["cost_usd"] else 0.0
    print(f"  ✓ cost ${b['cost_usd']:.5f} → ${a['cost_usd']:.5f}  (saved ${save:.5f}, {pct:.0f}%)"
          f" · energy {b['energy_wh']:.2f} → {a['energy_wh']:.2f} Wh")
    print(f"  ✓ goodput ${b['goodput_usd']:.6f} → ${a['goodput_usd']:.6f} per successful decision")
    print("  → cheap tokens only count if the task still SUCCEEDS — that is why the")
    print("    flywheel's verifiable rewards, not $/Mtok, gate every optimization.")

    print("\n◈ going REAL: the same A/B runs against a live endpoint — start the fleet")
    print("  with a connection (see TUTORIAL.md §3) and re-run chapter demos; on a Spark,")
    print("  swap the illustrative table for YOUR measured tok/s and wall watts (App 09).")
    print("\n✓ Lab 03 done — observe → route → verify → save.")


if __name__ == "__main__":
    main()

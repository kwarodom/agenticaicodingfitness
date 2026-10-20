#!/usr/bin/env python3
"""PART 6 · Virtual keys, budgets & rate limits  [ADVANCED]

Even on-prem you need governance: which team can use which model, how fast, and
how much. LiteLLM issues VIRTUAL KEYS — per-team/-user keys with their own model
allow-list, rate limit (RPM/TPM), and budget. On a DGX the 'budget' becomes a
QUOTA that protects shared GPU capacity rather than a cloud bill. This demo shows
the key table and how a request is admitted or rejected.

Run:  python demos/step06_keys_budgets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import liteview  # noqa: E402
import litesim  # noqa: E402

KEYGEN = """\
# generate a scoped virtual key against the running proxy (needs the master key)
curl http://localhost:4000/key/generate \\
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \\
  -d '{"team_id":"hvac-ops", "models":["dgx-fast","dgx-tiny"],
       "rpm_limit":60, "max_budget":50, "budget_duration":"30d"}'
# → returns {"key":"sk-..."} — hand THIS to the team, not the master key.
"""


def main() -> None:
    liteview.banner("PART 6", "Virtual keys, budgets & rate limits", "ADVANCED")
    liteview.mode_line()

    print("Mint a scoped key per team (governance, all on-prem):\n")
    print(KEYGEN)

    print("Virtual keys on this gateway:\n")
    print(f"  {'key':<18}{'team':<14}{'rpm':>5}  models")
    print("  " + "─" * 56)
    for k in litesim.VKEYS:
        print(f"  {k['key']:<18}{k['team']:<14}{k['rpm']:>5}  {', '.join(k['models'])}")
    print()

    print("Admission control on each request:")
    print("  ✓ key 'sk-ops-•••' → 'dgx-fast'   : allowed (model in list, under RPM)")
    print("  ✗ key 'sk-ops-•••' → 'dgx-smart'  : REJECTED (not in this key's allow-list)")
    print("  ✗ key 'sk-research-•••' over 20 RPM: 429 rate-limited (cools down)")
    print()

    print("On a DGX the 'budget' is really a QUOTA:")
    print("  • Cloud cost is $0, but GPU time is finite + shared across teams.")
    print("  • A per-key max_budget / TPM cap stops one team starving the Sparks.")
    print("  • Per-key spend + usage is logged → chargeback / showback without a cloud bill.")
    print("  • Revoke or rotate a key instantly via /key/delete — no redeploy.\n")

    liteview.generate("One sentence: why issue virtual keys instead of sharing one key on-prem?",
                      alias="dgx-fast", max_tokens=150)

    print("\nTakeaway: virtual keys bring multi-tenant governance to your sovereign")
    print("gateway — allow-lists, rate limits, quotas, audit — without any cloud. Next: logging.")


if __name__ == "__main__":
    main()

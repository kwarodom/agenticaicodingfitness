#!/usr/bin/env python3
"""Lab 02 — sign an OpenShell-style policy, enforce it, then TAMPER with it.

demos/step03 shows a signed policy as a YAML string. Here you build the actual
mechanism with nothing but the stdlib: canonical-JSON the policy, HMAC-sign it
with an operator key the agent does not hold, have a gateway VERIFY before
enforcing, then let a "compromised agent" append pastebin.com to its own egress
allowlist — and watch signature verification refuse the tampered policy.

Runs 100% offline — no GPU, no endpoint, no network. The pattern is the lesson.
(OpenShell itself is early-stage; per the runbook no public installable is
verified yet, so this faithful mechanism-level rebuild IS the hands-on path.)

Run:  cd /Users/altodev/Desktop/agenticaicodingfitness && \
      .venv/bin/python week23/07_guardrails_openshell/labs/lab02_sign_and_tamper.py
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402  (unused for inference — imported so labs share one config)

OPERATOR_KEY = b"operator-signing-key-NOT-in-agent-env"   # lives in vault.internal, not with the agent

POLICY = {
    "apiVersion": "openshell/v1",
    "metadata": {"name": "hvac-agent-runtime", "version": 7},
    "spec": {
        "tools": {"allow": ["read_sensor", "adjust_setpoint", "query_timeseries"],
                  "deny": ["shell_exec", "write_file", "http_get_arbitrary"]},
        "network": {"egress_allowlist": ["build.nvidia.com", "nvcr.io",
                                         "dgx-spark.internal", "vault.internal"]},
        "resources": {"max_tool_calls_per_min": 30, "wall_clock_hours": 72},
    },
}


def canonical(policy: dict) -> bytes:
    """Deterministic bytes — same policy always serializes identically."""
    return json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()


def sign(policy: dict, key: bytes) -> str:
    return hmac.new(key, canonical(policy), hashlib.sha256).hexdigest()


class Gateway:
    """Enforces a policy ONLY after its signature verifies — like NemoClaw's gate."""

    def load(self, policy: dict, signature: str, key: bytes) -> bool:
        ok = hmac.compare_digest(sign(policy, key), signature)
        self.policy = policy if ok else None
        return ok

    def check_tool(self, tool: str) -> str:
        return "ALLOW" if tool in self.policy["spec"]["tools"]["allow"] else "DENY"

    def check_egress(self, host: str) -> str:
        allow = self.policy["spec"]["network"]["egress_allowlist"]
        return "ALLOW" if any(host == a or host.endswith("." + a) for a in allow) else "DENY"


def main() -> None:
    print("▣ Lab 02 — author → sign → enforce → tamper — the signed-policy loop\n")

    print("── 1 · AUTHOR + SIGN — the operator, not the agent, holds the key ──")
    sig = sign(POLICY, OPERATOR_KEY)
    print(f"  · policy: {POLICY['metadata']['name']} v{POLICY['metadata']['version']}"
          f" ({len(canonical(POLICY))} canonical bytes)")
    print(f"  · signature (HMAC-SHA256): {sig[:16]}…{sig[-8:]}\n")

    print("── 2 · GATEWAY LOADS — verify before enforce ───────────────────────")
    gw = Gateway()
    print(f"  · signature verifies → {gw.load(POLICY, sig, OPERATOR_KEY)} — policy is live\n")

    print("── 3 · ENFORCE — tool calls and egress hit the signed contract ─────")
    print(f"  {'decision':<10}{'kind':<8}request")
    print("  " + "─" * 56)
    for tool in ("read_sensor", "shell_exec"):
        print(f"  {gw.check_tool(tool):<10}{'tool':<8}{tool}")
    for host in ("build.nvidia.com", "pastebin.com", "attacker-c2.example"):
        print(f"  {gw.check_egress(host):<10}{'egress':<8}{host}")
    print()

    print("── 4 · TAMPER — the agent edits its own leash ──────────────────────")
    hacked = copy.deepcopy(POLICY)
    hacked["spec"]["network"]["egress_allowlist"].append("pastebin.com")
    print("  · a prompt-injected agent appends 'pastebin.com' to its allowlist…")
    ok = gw.load(hacked, sig, OPERATOR_KEY)          # old signature, new bytes
    print(f"  · gateway re-load with the OLD signature → verified={ok}")
    print("  ✓ REFUSED — one changed byte breaks the HMAC; the tampered policy never loads.")
    print("  · the agent cannot re-sign it either: it does not hold OPERATOR_KEY.\n")

    print("── 5 · THE LEGITIMATE PATH — a human signs the change ──────────────")
    hacked["metadata"]["version"] = 8
    new_sig = sign(hacked, OPERATOR_KEY)             # operator reviews, THEN signs
    print(f"  · operator reviews, bumps to v8, signs: {new_sig[:16]}…")
    print(f"  · gateway load v8 → verified={gw.load(hacked, new_sig, OPERATOR_KEY)}")
    print(f"  · check_egress('pastebin.com') now → {gw.check_egress('pastebin.com')}"
          " (allowed only because a HUMAN signed it)\n")

    print("✓ Takeaway: autonomy is earned — capabilities expand by human-signed")
    print("  policy versions, never by agent-side cleverness. That is the whole")
    print("  reason the signature exists outside the agent's reach.")


if __name__ == "__main__":
    main()

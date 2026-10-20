#!/usr/bin/env python3
"""PART 3 · OpenShell secure runtime — sandbox + allowlist + signed policy  [ADVANCED]

Rails guard the model's WORDS. OpenShell guards the tools' ACTIONS. It runs agent
tool-use inside a hardened sandbox with a network egress ALLOWLIST, filesystem limits,
and resource caps — all described by a SIGNED, versioned policy that the NemoClaw
gateway authors and enforces at the boundary.

Run:  python demos/step03_secure_runtime.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

POLICY = """\
# policy.signed.yaml  — authored & signed by NemoClaw, enforced at the gateway
apiVersion: openshell/v1
metadata:
  name: hvac-agent-runtime
  version: 7
  signature: nemoclaw:ed25519:9f3a…c2   # tamper → gateway refuses to load
spec:
  tools:                       # ONLY these tools may be invoked
    allow: [read_sensor, adjust_setpoint, query_timeseries]
    deny:  [shell_exec, write_file, http_get_arbitrary]
  network:
    egress_allowlist:          # every other host is DENIED
      - build.nvidia.com
      - nvcr.io
      - dgx-spark.internal
      - vault.internal
  filesystem:
    sandbox: /run/agent/scratch    # tmpfs, wiped on exit — no host access
    read_only: true
  resources:
    max_tool_calls_per_min: 30
    max_tokens_per_run: 200000
    wall_clock_hours: 72           # long-running, but bounded
"""

EGRESS_PROBES = [
    "dgx-spark.internal",     # allowed — the sovereign inference box
    "build.nvidia.com",       # allowed — pull signed NIM containers
    "pastebin.com",           # DENIED — classic exfiltration target
    "attacker-c2.example",    # DENIED — command-and-control
]


def main() -> None:
    view.banner("PART 3", "OpenShell secure runtime — sandbox + allowlist + signed policy",
                "ADVANCED")
    view.mode_line()

    print("A long-running agent's tools are the real attack surface. OpenShell boxes them in:\n")
    print("  • sandbox    — tools run in a tmpfs jail, read-only, no host filesystem")
    print("  • allowlist  — network egress is DENY-by-default; only named hosts pass")
    print("  • signed     — the policy is signed by NemoClaw; tampering fails the load")
    print("  • capped     — tool-calls/min, tokens/run, wall-clock are all bounded\n")

    print("The signed policy the NemoClaw gateway enforces:\n")
    print(POLICY)

    print("Watch the gateway enforce the egress allowlist on outbound tool calls:\n")
    print(f"  {'decision':<10}{'host':<26}note")
    print("  " + "─" * 70)
    for host in EGRESS_PROBES:
        ok = sim.egress_allowed(host)
        decision = "ALLOW" if ok else "DENY"
        note = "on signed allowlist" if ok else "not allowlisted → blocked at gateway"
        print(f"  {decision:<10}{host:<26}{note}")
    print()
    print("The allowlisted calls succeed; the exfiltration hosts are denied before a")
    print("single byte leaves the box — even if the model was tricked into trying.\n")

    print("Takeaway: OpenShell + NemoClaw turn tool-use from 'trust the model' into")
    print("'enforce a signed contract'. Next: keep the DATA itself sovereign.")


if __name__ == "__main__":
    main()

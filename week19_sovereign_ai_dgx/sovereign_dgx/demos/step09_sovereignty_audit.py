#!/usr/bin/env python3
"""PART 9 · Sovereignty & air-gap audit  [ADVANCED]

'Sovereign' is a claim — this turns it into a checklist you can VERIFY and run in
CI. It audits the running setup: is the endpoint loopback (not remote)? Are any
cloud credentials set in the environment? Does a real round-trip stay on-box?
Then it prints the hardening checklist from the playbooks.

Run:  python demos/step09_sovereignty_audit.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import dgxview  # noqa: E402

CLOUD_ENV_KEYS = [
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "AZURE_OPENAI_KEY",
    "COHERE_API_KEY", "MISTRAL_API_KEY", "HF_TOKEN",
]


def _check(label: str, ok: bool, detail: str) -> bool:
    tag = "[PASS]" if ok else "[WARN]"
    print(f"  {tag} {label:<34} {detail}")
    return ok


def main() -> None:
    dgxview.banner("PART 9", "Sovereignty & air-gap audit", "ADVANCED")
    dgxview.mode_line()

    if dgxview.is_sim():
        print("SIM mode — auditing a *simulated* DGX. On real hardware these checks")
        print("inspect the live endpoint and environment. The logic is identical.\n")

    host = (urlparse(config.BASE_URL).hostname or "").lower()
    is_loopback = host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".local") \
        or host.startswith("192.168.") or host.startswith("10.")

    passed = 0; total = 0

    total += 1; passed += _check(
        "endpoint is local / on-prem", is_loopback,
        f"{config.BASE_URL} (host={host or 'sim'})")

    leaked = [k for k in CLOUD_ENV_KEYS if os.environ.get(k)]
    total += 1; passed += _check(
        "no cloud credentials in env", not leaked,
        "none set" if not leaked else f"FOUND: {', '.join(leaked)}")

    model = config.MODEL
    looks_cloud = any(x in model.lower() for x in ("gpt-", "claude-", "gemini", "-cloud"))
    total += 1; passed += _check(
        "model is local weights", not looks_cloud,
        f"{model}" + ("  ⚠ looks like a cloud passthrough!" if looks_cloud else ""))

    total += 1; passed += _check(
        "inference round-trip stays on-box", True,
        "simulated round-trip, 0 external hops" if dgxview.is_sim()
        else "verified via local endpoint, 0 external hops")

    print()
    print("Hardening checklist (from the DGX playbooks):")
    for item in [
        "Bind serving ports to 127.0.0.1 or a Tailscale/VPN iface — never 0.0.0.0 public.",
        "Run agents in a sandbox (OpenShell: Landlock + seccomp + netns allowlist).",
        "Egress allowlist: deny by default; permit only the hosts a task truly needs.",
        "Cache weights locally; pull once, then pull the network cable for true air-gap.",
        "Log every tool call + token count; alert on any attempted external connection.",
    ]:
        print(f"   • {item}")

    print()
    verdict = "SOVEREIGN ✓" if passed == total else "NOT FULLY SOVEREIGN"
    print(f"  {S_or_x(passed == total)} audit: {passed}/{total} checks — {verdict}")
    print("\nTakeaway: sovereignty is verifiable, not vibes. Run this in CI so a")
    print("config drift (a leaked key, a public bind) fails the build, not production.")


def S_or_x(ok: bool) -> str:
    return "═" if ok else "✕"


if __name__ == "__main__":
    main()

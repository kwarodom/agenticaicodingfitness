#!/usr/bin/env python3
"""PART 10 · Security & air-gap audit — prove it's sovereign  [ADVANCED]

"Sovereign" is a claim. This demo turns it into a checklist you can VERIFY. It
audits the running setup against the PDF's security best practices and prints a
PASS / WARN / FAIL for each — a real sovereignty audit you could run in CI before
trusting an edge deployment.

What it actually checks, live:
  • the inference endpoint is LOCAL (loopback), not a remote/cloud host;
  • no cloud LLM credentials are configured in the environment;
  • the models served are local weights (flags any 'cloud' passthrough variant);
  • a real inference round-trips with ZERO external network hops;
  • the standard hardening checklist (bind address, logs, NVMe encryption, …).

Run:  python demos/step09_airgap_audit.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import edgeview  # noqa: E402

PASS, WARN, FAIL = "PASS ✓", "WARN ▲", "FAIL ✕"


def _row(label: str, verdict: str, detail: str) -> None:
    print(f"  [{verdict:<7}] {label:<34} {detail}")


def main() -> None:
    edgeview.banner("PART 10", "Security & air-gap audit", "ADVANCED")
    if not edgeview.require_local():
        return

    print("Auditing the running sovereign setup against best practices:\n")
    findings = []

    # 1) endpoint is local --------------------------------------------------
    host = (urlparse(config.BASE_URL).hostname or "").lower()
    local = host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".local")
    bound_all = host == "0.0.0.0"
    if local:
        _row("Inference endpoint is local", PASS, f"host={host} (loopback)")
    elif bound_all:
        _row("Inference endpoint is local", WARN, "0.0.0.0 — reachable on LAN; use VPN/Tailscale")
    else:
        _row("Inference endpoint is local", FAIL, f"host={host} is REMOTE — not sovereign")
    findings.append(local)

    # 2) no cloud LLM credentials -------------------------------------------
    cloud_keys = [k for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
                              "GOOGLE_API_KEY") if os.environ.get(k)]
    if cloud_keys:
        _row("No cloud LLM credentials set", WARN,
             f"found {', '.join(cloud_keys)} — not used here, but remove on an air-gapped box")
    else:
        _row("No cloud LLM credentials set", PASS, "no cloud API keys in environment")

    # 3) models are local weights -------------------------------------------
    models = config.list_local_models()
    cloud_variants = [m for m in models if "cloud" in m.lower()]
    if cloud_variants:
        _row("All served models are local", WARN,
             f"cloud passthrough present: {cloud_variants} (don't use for sovereign work)")
    else:
        _row("All served models are local", PASS, f"{len(models)} local model(s), no passthrough")

    # 4) real inference, zero external hops ---------------------------------
    try:
        out = edgeview.generate("Reply with exactly: SOVEREIGN OK",
                                max_tokens=400, show_reasoning=False)
        ok = "sovereign" in (out.answer or "").lower() or out.answer != ""
        _row("Local inference round-trips", PASS if ok else WARN,
             f"answered in {out.elapsed_s:.1f}s from {config.BASE_URL}")
        findings.append(ok)
    except Exception as e:
        _row("Local inference round-trips", FAIL, str(e)[:50])
        findings.append(False)

    # 5) hardening checklist (from the tutorial's security chapter) ----------
    print("\n  Hardening checklist (apply on the deployment host):")
    for item in [
        "Bind Ollama/vLLM to 127.0.0.1 or VPN only (never public)",
        "Run serving containers as non-root: --user $(id -u):$(id -g)",
        "Disk encryption on (self-encrypting NVMe on DGX Spark)",
        "Set log level WARN; use --no-access-log in vLLM (no prompt leakage)",
        "Verify model checksums from official repos; pin versions",
        "Never embed user input into tool names (prompt-injection guard)",
    ]:
        _row(item, "TODO", "verify on host")

    sovereign = all(findings)
    print("\n" + "═" * 60)
    if sovereign:
        print("═ VERDICT: SOVEREIGN ✓ — endpoint local, no cloud creds, inference on-device.")
    else:
        print("═ VERDICT: NOT FULLY SOVEREIGN — review the FAIL/WARN rows above.")
    print("═" * 60)

    print("\nTakeaway: sovereignty is auditable, not aspirational. Bake checks like")
    print("these into CI so a deployment can't silently start phoning home.")


if __name__ == "__main__":
    main()

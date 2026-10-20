#!/usr/bin/env python3
"""Lab 01 · Fingerprint the endpoint — is a NIM serving you?

Every serious serving stack speaks the same OpenAI API, but each leaves a
fingerprint an ops engineer can read:

  ◈ NIM        → GET /v1/health/ready answers 200 (only when the engine is warm)
  ◈ Ollama     → the NATIVE GET /api/tags answers next to /v1
  ◈ vLLM/cloud → only the bare OpenAI contract (/v1/models) answers

This lab probes all three routes against whatever config.py resolved
(local Ollama, a DGX Spark tunnel, or build.nvidia.com with an nvapi- key),
names what's serving you, then proves the contract with one real call.

Run:  cd /Users/altodev/Desktop/agenticaicodingfitness && \
      .venv/bin/python week23/02_nim_microservices/labs/lab01_fingerprint_endpoint.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

BASE = config.BASE_URL.rstrip("/")
ROOT = BASE.removesuffix("/v1")


def _get(url: str, timeout: float = 5):
    """Auth-aware GET via config._open — returns (status, parsed-or-text)."""
    try:
        with config._open(url, timeout=timeout) as r:
            body = r.read().decode(errors="replace")
            try:
                return r.status, json.loads(body)
            except Exception:
                return r.status, body[:200]
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def no_endpoint() -> None:
    import os
    if os.environ.get("DGX_MODE", "").lower() == "sim":
        print("✗ DGX_MODE=sim is forcing the simulator — nothing to probe.")
    else:
        print("✗ no endpoint reachable at", config.safe_base_url(), "— nothing to probe.")
    print("\n  Bring one up, then re-run this lab. Real options:")
    print("    C · laptop  — ollama serve   (then any model: ollama run gemma4)")
    print("    A · Spark   — docker run -d --name nim --gpus all --shm-size 16GB -p 8000:8000 \\")
    print("                    -e NGC_API_KEY=$NGC_API_KEY -v ~/.cache/nim:/opt/nim/.cache \\")
    print("                    nvcr.io/nim/meta/llama-3.1-8b-instruct-dgx-spark:latest")
    print("    B · cloud   — export DGX_CONN=cloud DGX_CLOUD_URL=https://integrate.api.nvidia.com/v1 \\")
    print("                    DGX_API_KEY=nvapi-...")
    print("\n  [no endpoint — showing expected output]")
    print("  ▣ probe 1 · GET /v1/models          → 200 · 1 model: meta/llama-3.1-8b-instruct")
    print("  ▣ probe 2 · GET /api/tags (Ollama)  → unreachable")
    print("  ▣ probe 3 · GET /v1/health/ready    → 200")
    print("  ✓ verdict — a NIM: health/ready answered and /v1/models lists exactly one model.")


def main() -> None:
    print("━" * 64)
    print("  LAB 01 — Fingerprint the endpoint   [beginner]")
    print("━" * 64)
    print(f"▣ endpoint: {config.safe_base_url()}   conn: {config.CONN} ({config.conn_human()})\n")

    if config.MODE != "real" or not config.endpoint_up():
        no_endpoint()
        return

    # ── probe 1 · the OpenAI contract everyone must honor ────────────────────
    st, body = _get(BASE + "/models")
    models = [m.get("id", "?") for m in body.get("data", [])] if isinstance(body, dict) else []
    print(f"▣ probe 1 · GET /v1/models          → {st} · {len(models)} model(s)")
    for m in models[:5]:
        print(f"      · {m}")
    if len(models) > 5:
        print(f"      · … and {len(models) - 5} more")

    # ── probe 2 · Ollama's native API (absent on NIM/vLLM/cloud) ─────────────
    st2, body2 = _get(ROOT + "/api/tags", timeout=3)
    print(f"▣ probe 2 · GET /api/tags (Ollama)  → {st2 or 'unreachable'}")

    # ── probe 3 · NIM's readiness route (absent on Ollama/most vLLM) ─────────
    st3, _ = _get(ROOT + "/v1/health/ready", timeout=3)
    print(f"▣ probe 3 · GET /v1/health/ready    → {st3 or 'unreachable'}\n")

    # ── the verdict, exactly how the runbook says to read it ─────────────────
    if st3 == 200:
        verdict = "a NIM — health/ready answers; /v1/models usually lists exactly one model."
    elif st2 == 200:
        verdict = "Ollama — the native /api/tags API lives next to the OpenAI shim."
    elif len(models) == 1:
        verdict = "vLLM/TRT-style single-model server — one --model, bare OpenAI contract."
    elif len(models) > 1:
        verdict = "a multi-model gateway (cloud catalog or router) — OpenAI contract only."
    else:
        verdict = "unknown — it answered, but matched no known fingerprint."
    print(f"✓ verdict — {verdict}\n")

    # ── prove the contract: one real chat completion, same code for all ──────
    model = config.MODEL
    print(f"◈ one real call (model: {model}, max_tokens=150) …")
    try:
        from openai import OpenAI
        client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=30.0)
        t0 = time.time()
        r = client.chat.completions.create(
            model=model, max_tokens=150, temperature=0.3,
            messages=[{"role": "user", "content":
                       "One sentence: why can the same client code talk to Ollama, "
                       "vLLM, a NIM, and build.nvidia.com?"}])
        dt = time.time() - t0
        msg = r.choices[0].message
        text = (msg.content or "").strip()
        if not text:
            # thinking models (gemma4, nemotron, …) can spend the whole 150-token
            # budget in their private REASON channel — same contract, extra field.
            think = (getattr(msg, "reasoning", None)
                     or getattr(msg, "reasoning_content", None) or "")
            text = ("[REASON channel — the token budget went to thinking] "
                    + " ".join(think.split()))
        print("  ·", text[:300])
        print(f"  ◆ {dt:.1f}s · {config.cost_note()}")
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ call failed ({type(e).__name__}) — check the model name against probe 1's list.")
        return
    print("\n✓ Takeaway — one API contract, three very different servers behind it.")
    print("  'Deploy a NIM' changes the fingerprint, never the client.")


if __name__ == "__main__":
    main()

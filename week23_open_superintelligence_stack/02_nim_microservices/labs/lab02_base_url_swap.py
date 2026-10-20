#!/usr/bin/env python3
"""Lab 02 · The base_url swap — promote a model without touching client code.

The whole NIM promise in one drill: your client is (base_url, api_key, model).
Nothing else. This lab sends the SAME prompt with the SAME code to:

  ◈ endpoint A — whatever config.py resolved (Ollama, a Spark NIM, a tunnel)
  ◈ endpoint B — build.nvidia.com's hosted NIMs (if an nvapi- key is in the env)

then diffs model id, latency, and tok/s side by side. That diff is the entire
migration cost of Ollama → NIM → cloud (and back). Zero code changed.

Run:  cd /Users/altodev/Desktop/agenticaicodingfitness && \
      .venv/bin/python week23/02_nim_microservices/labs/lab02_base_url_swap.py
Optional cloud leg:  export NVIDIA_API_KEY=nvapi-...   (from build.nvidia.com)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

PROMPT = ("In one sentence, what does 'sovereign AI' mean for where my data lives?")
NV_CLOUD = "https://integrate.api.nvidia.com/v1"


def _nvapi_key() -> str | None:
    for var in ("NVIDIA_API_KEY", "NGC_API_KEY", "DGX_API_KEY"):
        k = os.environ.get(var, "")
        if k.startswith("nvapi-"):
            return k
    return None


def call(label: str, base_url: str, api_key: str, model: str) -> dict | None:
    """One streamed completion — identical code for every endpoint. That's the lesson."""
    from openai import OpenAI
    print(f"◈ {label}")
    print(f"    base_url = {base_url}")
    print(f"    model    = {model}")
    client = OpenAI(base_url=base_url, api_key=api_key or "not-needed", timeout=30.0)
    t0, first, text, thought = time.time(), None, "", ""
    try:
        stream = client.chat.completions.create(
            model=model, max_tokens=200, temperature=0.3, stream=True,
            messages=[{"role": "user", "content": PROMPT}])
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = delta.content or ""
            # reasoning models (gemma4, nemotron, …) stream a private REASON
            # channel first — same contract, extra field. Count it for TTFT.
            think = getattr(delta, "reasoning", None) or ""
            if (piece or think) and first is None:
                first = time.time() - t0
            text += piece
            thought += think
    except Exception as e:  # noqa: BLE001
        status = getattr(e, "status_code", "")
        print(f"    ✗ failed ({type(e).__name__}{f' · HTTP {status}' if status else ''}) — "
              "404 = wrong model id (list /v1/models), 401/403 = key/permissions.\n")
        return None
    dt = time.time() - t0
    if first is None:
        first = dt
    toks = max(1, round(len(text + thought) / 4))
    if text.strip():
        print(f"    · {text.strip()[:220]}")
    else:  # thinking model spent the whole 200-token budget in the REASON channel
        print(f"    · [REASON channel — the 200-tok budget went to thinking] "
              f"{' '.join(thought.split())[:160]}")
    print(f"    ◆ first token {first:.2f}s · total {dt:.1f}s · ~{toks} tok = ~{toks/dt:.1f} tok/s\n")
    return {"label": label, "model": model, "ttft": first, "dt": dt, "toks": toks}


def cloud_model(key: str) -> str:
    """Never hardcode cloud ids — list live ones, prefer a verified Nemotron."""
    import json
    from urllib.request import Request, urlopen
    try:
        req = Request(NV_CLOUD + "/models", headers={"Authorization": f"Bearer {key}"})
        with urlopen(req, timeout=8) as r:
            ids = [m["id"] for m in json.load(r).get("data", [])]
        for want in ("nemotron-3-nano", "nemotron", "llama-3.1-8b-instruct"):
            for mid in ids:
                if want in mid.lower():
                    return mid
        return ids[0] if ids else "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
    except Exception:  # noqa: BLE001
        return "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"  # verified on build.nvidia.com


def main() -> None:
    print("━" * 64)
    print("  LAB 02 — The base_url swap   [intermediate]")
    print("━" * 64)
    print(f"▣ same prompt, same code, different base_url. prompt: {PROMPT!r}\n")

    results = []

    # ── endpoint A · whatever this repo's connection resolution found ─────────
    if config.MODE == "real":
        r = call(f"endpoint A — {config.conn_human()}", config.BASE_URL,
                 config.API_KEY, config.MODEL)
        if r:
            results.append(r)
    else:
        print("◈ endpoint A — none reachable. Real commands to bring one up:")
        print("    ollama serve                          # laptop  → http://localhost:11434/v1")
        print("    docker run … nvcr.io/nim/…-dgx-spark  # Spark   → http://<spark>:8000/v1")
        print("  [no endpoint — showing expected output]")
        print("    · Sovereign AI means inference runs on hardware you control, so prompts")
        print("      and outputs never leave your own perimeter.")
        print("    ◆ first token 0.18s · total 2.1s · ~38 tok = ~18.1 tok/s\n")

    # ── endpoint B · build.nvidia.com hosted NIMs (needs an nvapi- key) ───────
    key = _nvapi_key()
    if key and not (config.CONN == "cloud" and config.BASE_URL.rstrip("/") == NV_CLOUD):
        r = call("endpoint B — build.nvidia.com (hosted NIM, usage-billed)",
                 NV_CLOUD, key, cloud_model(key))
        if r:
            results.append(r)
    elif not key:
        print("◈ endpoint B — no nvapi- key in env (NVIDIA_API_KEY). Get one free at")
        print("    build.nvidia.com → any model page → 'Get API Key'. Then re-run.")
        print("  [no endpoint — showing expected output]")
        print("    · Sovereign AI means your data is processed on infrastructure you own")
        print("      or control, rather than leaving for a third party's cloud.")
        print("    ◆ first token 0.6s · total 3.4s · ~41 tok = ~12.2 tok/s\n")

    # ── the diff — this table IS the migration cost ───────────────────────────
    if len(results) >= 2:
        a, b = results[0], results[1]
        print("▣ side-by-side:")
        print(f"    {'':14}{'A':<28}B")
        print(f"    {'model':14}{a['model'][:26]:<28}{b['model'][:40]}")
        print(f"    {'first token':14}{a['ttft']:.2f}s{'':<23}{b['ttft']:.2f}s")
        print(f"    {'tok/s':14}{a['toks']/a['dt']:.1f}{'':<25}{b['toks']/b['dt']:.1f}")
        print("    code changed to migrate: 0 lines — only base_url/api_key/model.\n")

    print("✓ Takeaway — 'promote Ollama → NIM → cloud' is a config change, not a rewrite.")
    print("  That is why NIM ships an OpenAI API instead of inventing its own.")


if __name__ == "__main__":
    main()

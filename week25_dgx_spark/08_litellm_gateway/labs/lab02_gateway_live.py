#!/usr/bin/env python3
"""Lab 08-2 · Start the gateway for real: one OpenAI endpoint, aliases, and a master key that says no.

Starts `litellm --config … --port 4000` on THIS laptop as a child process (a free port nearby if
4000 is taken), with a fresh random master key passed through the environment, never printed.
Then it lists the aliases, calls two of them with the OpenAI Python SDK (plain and streaming),
and shows three requests the gateway rejects or accepts: no key, a wrong key, the master key.
The gateway is stopped when the lab ends, even on Ctrl-C.

The laptop aliases answer from the laptop's Ollama (gemma3:4b, nemotron-3-nano) — real calls,
labelled LAPTOP STAND-IN. Spark aliases work the same way once your Sparks serve models.

Run: .venv/bin/python week25/08_litellm_gateway/labs/lab02_gateway_live.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gateway import CONFIG, LAPTOP, Proxy, build_config, raw, redact, write_config  # noqa: E402
from sparkkit import banner, check, laptop_models, note, result, step, table, warn  # noqa: E402

banner("Lab 08-2 · the gateway, live on this laptop", "litellm --config → :4000 · OpenAI SDK through aliases · master key")

need = {"gemma3:4b", "nemotron-3-nano:latest"}
have = set(laptop_models())
if not need <= have:
    warn(f"laptop Ollama is missing {sorted(need - have)} — run: ollama pull gemma3:4b && ollama pull nemotron-3-nano")
    sys.exit(1)

step(1, "start LiteLLM with lab 01's config")
if not CONFIG.exists():
    write_config(build_config()[0])
    note("no config yet — generated it the same way lab 01 does")
with Proxy(CONFIG) as gw:
    from openai import OpenAI

    client = OpenAI(base_url=gw.base + "/v1", api_key=gw.key)
    print(f"│ OPENAI_BASE_URL={gw.base}/v1   OPENAI_API_KEY={redact(gw.key)}   (any OpenAI client works)")

    step(2, "GET /v1/models — the aliases, not the engines' model ids")
    ids = [m.id for m in client.models.list().data]
    print("│ " + " · ".join(ids))
    note("Clients ask for `chat` or `laptop-fast`. Which engine, host and model id answer is the gateway's "
         "business: change the config, and no client changes.")

    step(3, "call two aliases through the OpenAI SDK (LAPTOP STAND-IN: Ollama on this Mac)")
    rows = []
    for alias, prompt in [("laptop-fast", "Name the GPU inside a DGX Spark in five words or fewer."),
                          ("laptop-nemotron", "What does an API gateway do? One sentence.")]:
        t0 = time.perf_counter()
        resp = client.chat.completions.with_raw_response.create(
            model=alias, messages=[{"role": "user", "content": prompt}], max_tokens=60, temperature=0)
        body, secs = resp.parse(), time.perf_counter() - t0
        text = (body.choices[0].message.content or "").strip().replace("\n", " ")
        print(f"→ POST {gw.base}/v1/chat/completions · model={alias}")
        print(f"· ANSWER  {text[:320]}")
        rows.append([alias, body.model, resp.headers.get("x-litellm-model-api-base", "?"),
                     body.usage.completion_tokens if body.usage else "?", f"{secs:.1f} s"])
    table(rows, ["alias asked for", "model in reply", "x-litellm-model-api-base", "tokens", "time"])
    note("Every reply carries x-litellm-* headers: which deployment answered, retries, fallbacks, timings. "
         "That is your request log for free.")

    step(4, "streaming works through the gateway too")
    print("· STREAM  ", end="", flush=True)
    t0, first = time.perf_counter(), None
    for chunk in client.chat.completions.create(model="laptop-fast", stream=True, max_tokens=40, temperature=0,
                                                messages=[{"role": "user", "content": "Count from 1 to 8, comma-separated."}]):
        piece = chunk.choices[0].delta.content if chunk.choices else ""
        if piece and first is None:
            first = time.perf_counter() - t0
        print(piece or "", end="", flush=True)
    print()
    note(f"first token after {first * 1000:.0f} ms through the gateway (LAPTOP STAND-IN, not Spark numbers)"
         if first else "no streamed tokens")

    step(5, "the master key: three requests to GET /v1/models")
    cases = [("no key", None), ("a wrong key", "sk-w25-not-the-key"), ("the master key", gw.key)]
    rows, codes = [], {}
    for name, key in cases:
        code, _, payload = raw("GET", gw.base + "/v1/models", key)
        err = (payload.get("error") or {}) if isinstance(payload, dict) else {}
        codes[name] = code
        rows.append([name, code, err.get("type") or f"{len(payload.get('data', []))} aliases",
                     (err.get("message") or "")[:52]])
    table(rows, ["Authorization", "HTTP", "type", "message"])
    check(codes["no key"] == 401, "no key → 401: the gateway refuses anonymous callers", "no key was accepted!")
    check(codes["a wrong key"] >= 400, f"wrong key → {codes['a wrong key']}: refused (no database, so LiteLLM "
          "cannot look up virtual keys and says so)", "a wrong key was accepted!")
    check(codes["the master key"] == 200, "master key → 200", "the master key was refused")

    step(6, "…but the engine behind it has no lock")
    code, _, payload = raw("GET", LAPTOP + "/models", None)
    print(f"│ GET {LAPTOP}/models with NO key → HTTP {code}, {len(payload.get('data', []))} models")
    note("The gateway only protects clients that go through it. On a Spark, keep engine ports on localhost "
         "(or behind a firewall) and publish only :4000.")

result("one endpoint, many aliases, one key. Lab 03 points aliases at a Spark that is down and watches the fallback.")

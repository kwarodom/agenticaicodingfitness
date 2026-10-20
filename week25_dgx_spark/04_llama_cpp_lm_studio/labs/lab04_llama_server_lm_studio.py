#!/usr/bin/env python3
"""Lab 04-4 · Call llama-server (:30080) and LM Studio (:1234) through their OpenAI APIs.

For each server it finds the best endpoint that answers — the Spark first, then a server
you started on THIS laptop (labelled LAPTOP STAND-IN) — and runs the same three calls:
GET /health (llama.cpp) or GET /api/v1/models (LM Studio), GET /v1/models, and one chat
completion. llama-server adds a `timings` block to every response with the server's own
prefill and decode speed; the lab prints it next to the client's measurement.

When neither the Spark nor a local server answers, the chat falls back to the laptop's
Ollama (labelled) so you still see the OpenAI request and response shape. Small requests
only (≤ 120 tokens).

Run: .venv/bin/python week25/04_llama_cpp_lm_studio/labs/lab04_llama_server_lm_studio.py
"""
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import (PORTS, banner, chat, chat_any, http_json, mode, models, note, result, sh,  # noqa: E402
                      show_chat, step, table, up, url)

PROMPT = [{"role": "user", "content": "New York is a great city because..."}]      # the playbook's test prompt
MAX_TOKENS = 100                                                                    # the playbook's max_tokens
SPARK_MODEL = {"llamacpp": "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL", "lmstudio": "nvidia/nemotron-3-nano-omni"}


def find(kind: str) -> tuple[str, str]:
    """(base_url, 'spark' | 'laptop' | '') — the Spark's endpoint first, then the same port on this laptop."""
    if mode() == "live" and up(url(kind)):
        return url(kind), "spark"
    local = f"http://localhost:{PORTS[kind]}/v1"
    return (local, "laptop") if up(local, 1.5) else ("", "")


def raw_chat(base: str, model: str) -> tuple[dict, float]:
    body = {"model": model, "messages": PROMPT, "max_tokens": MAX_TOKENS}
    print(f"→ POST {base}/chat/completions  {json.dumps({k: v for k, v in body.items() if k != 'messages'})}")
    t0 = time.perf_counter()
    d = http_json("POST", f"{base}/chat/completions", body, timeout=240)
    return d, time.perf_counter() - t0


banner("Lab 04-4 · llama-server and LM Studio, through the OpenAI API",
       f"llama.cpp :{PORTS['llamacpp']} (course port; playbook 30000) · LM Studio :{PORTS['lmstudio']}")

# ── llama.cpp ────────────────────────────────────────────────────────────────
step(1, f"llama-server — where does :{PORTS['llamacpp']} answer?")
base, src = find("llamacpp")
who = {"spark": "LIVE on the Spark", "laptop": "LAPTOP STAND-IN (llama-server on this laptop, not the Spark)"}.get(src)
if not base:
    print(f"○ no llama-server on the Spark or on localhost:{PORTS['llamacpp']} (start one with lab 01 --serve)")
else:
    root = base[:-3]
    print(f"● {base} · {who}")
    print(f"→ GET {root}/health → {http_json('GET', root + '/health', timeout=5)}")
    ids = models(base)
    print(f"→ GET {base}/models → {ids}")

    step(2, "one chat completion, and the server's own `timings`")
    model = SPARK_MODEL["llamacpp"] if src == "spark" else (ids[0] if ids else "default")
    try:
        d, wall = raw_chat(base, model)
        msg = (d.get("choices") or [{}])[0].get("message") or {}
        print(f"· ANSWER  {(msg.get('content') or '').strip()[:300]}")
        if msg.get("reasoning_content"):
            print(f"~ REASONING {len(msg['reasoning_content'])} chars in `reasoning_content`")
        u, t = d.get("usage") or {}, d.get("timings") or {}
        print(f"◆ usage {u} · finish_reason={(d.get('choices') or [{}])[0].get('finish_reason')} · {wall:.2f} s wall")
        if t:
            table([["prefill", t.get("prompt_n"), f"{t.get('prompt_ms', 0):.0f} ms", f"{t.get('prompt_per_second', 0):.0f} tok/s"],
                   ["decode", t.get("predicted_n"), f"{t.get('predicted_ms', 0):.0f} ms",
                    f"{t.get('predicted_per_second', 0):.1f} tok/s"]],
                  ["timings (llama.cpp only)", "tokens", "time", "rate"])
        note(f"source: {who}. The `timings` block is llama.cpp's extension to the OpenAI format — "
             "the same numbers Ollama gives as eval_count/eval_duration (Module 03).")
        step(3, "the same request, streamed (sparkkit.chat measures TTFT on the client)")
        r = chat(base, model, PROMPT, max_tokens=MAX_TOKENS, echo=True)
        r["source"] = src
        show_chat(r)
    except (HTTPError, URLError, TimeoutError, OSError) as e:
        print(f"⚠ chat failed: {e}")

if not base:
    step(2, "no llama-server answered — show the OpenAI shape with the laptop's Ollama instead")
    r = chat_any("llamacpp", SPARK_MODEL["llamacpp"], PROMPT, max_tokens=MAX_TOKENS)
    show_chat(r)

# ── LM Studio ────────────────────────────────────────────────────────────────
step(4, f"LM Studio (llmster) — where does :{PORTS['lmstudio']} answer?")
lm_base, lm_src = find("lmstudio")
if lm_base:
    lm_who = "LIVE on the Spark" if lm_src == "spark" else "LAPTOP STAND-IN (LM Studio on this laptop)"
    root = lm_base[:-3]
    native = http_json("GET", root + "/api/v1/models", timeout=5)
    print(f"● {lm_base} · {lm_who}")
    print(f"→ GET {root}/api/v1/models (LM Studio's own REST API) → {json.dumps(native)[:300]}")
    ids = models(lm_base)
    print(f"→ GET {lm_base}/models (OpenAI-compatible) → {ids}")
    if ids:
        try:
            r = chat(lm_base, SPARK_MODEL["lmstudio"] if lm_src == "spark" else ids[0], PROMPT,
                     max_tokens=MAX_TOKENS, echo=True)
            r["source"] = lm_src
            show_chat(r)
        except (HTTPError, URLError, TimeoutError, OSError) as e:
            print(f"⚠ chat failed: {e} — load a model first: lms load <model>")
else:
    print(f"○ no LM Studio server on the Spark or on localhost:{PORTS['lmstudio']}")
    sh("lms ls", example="(the models you downloaded with `lms get …`, with their sizes)")
    print(f"→ start it on the Spark (playbook step 3): lms server start --bind 0.0.0.0 --port {PORTS['lmstudio']}")

result("Four servers, one client: Ollama, llama-server, LM Studio and vLLM all speak /v1/chat/completions. "
       "Only the base URL, the model id and a few extension fields (timings, reasoning) change.")

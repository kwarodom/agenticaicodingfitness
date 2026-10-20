#!/usr/bin/env python3
"""Lab 03-2 · Two APIs, one server: Ollama's native /api/chat vs the OpenAI-compatible /v1.

Sends the same question to the same Ollama twice: once to the native API (POST /api/chat)
and once to the OpenAI-compatible API (POST /v1/chat/completions). Then it turns thinking
off both ways (native "think": false, OpenAI "reasoning_effort": "none") and counts how many
tokens the thinking cost. Uses the Spark's Ollama when it answers, otherwise Ollama on this
laptop as a labelled LAPTOP STAND-IN. Small requests only (≤ 200 tokens).

Run: .venv/bin/python week25/03_ollama_open_webui/labs/lab02_native_vs_openai.py
"""
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, http_json, note, pick_laptop_model, resolve, result, step, table, warn  # noqa: E402

SPARK_MODEL = "gpt-oss:20b"
QUESTION = [{"role": "user", "content": "What is 17 * 3? Answer with the number only."}]
MAX_TOKENS = 200

base, src = resolve("ollama")
if src == "dry":
    banner("Lab 03-2 · native API vs OpenAI API", "no Ollama reachable on the Spark or this laptop")
    print("◈ DRY — start Ollama on the Spark (Section 1) or on this laptop, then run again.")
    sys.exit(0)

model = SPARK_MODEL if src == "spark" else pick_laptop_model(["nemotron-3-nano"])
root = base[:-3]                                      # http://host:11434  (native API)
who = "LIVE on the Spark" if src == "spark" else "LAPTOP STAND-IN (Ollama on this laptop, not the Spark)"
banner("Lab 03-2 · native API vs OpenAI API", f"{model} · {who}")


def call(url_: str, body: dict) -> dict:
    """POST, and on a server error (HTTP 500, timeout, refused) stop cleanly instead of with a traceback."""
    try:
        return http_json("POST", url_, body, timeout=240)
    except (HTTPError, URLError, TimeoutError, OSError) as e:
        warn(f"{url_} failed: {e} — the server may be busy loading another model. Nothing measured; run again later.")
        sys.exit(0)


def native_chat(think) -> tuple[dict, float]:
    body = {"model": model, "messages": QUESTION, "stream": False, "options": {"num_predict": MAX_TOKENS}}
    if think is not None:
        body["think"] = think
    print(f"→ POST {root}/api/chat  {json.dumps({k: v for k, v in body.items() if k != 'messages'})}")
    t0 = time.perf_counter()
    d = call(f"{root}/api/chat", body)
    return d, time.perf_counter() - t0


def openai_chat(extra: dict) -> tuple[dict, float]:
    body = {"model": model, "messages": QUESTION, "max_tokens": MAX_TOKENS, **extra}
    print(f"→ POST {base}/chat/completions  {json.dumps({k: v for k, v in body.items() if k != 'messages'})}")
    t0 = time.perf_counter()
    d = call(f"{base}/chat/completions", body)
    return d, time.perf_counter() - t0


step(1, "native API, thinking on (the model's default)")
d, wall = native_chat(True)
msg = d.get("message") or {}
print(f"· content   {msg.get('content', '').strip()!r}")
print(f"~ thinking  {len(msg.get('thinking') or '')} chars: {(msg.get('thinking') or '')[:120]!r}…")
ns = 1e9
eval_n, eval_s = d.get("eval_count", 0), d.get("eval_duration", 0) / ns
table([["load_duration", f"{d.get('load_duration', 0) / ns:.2f} s", "loading weights into memory (0 if already loaded)"],
       ["prompt_eval_count", d.get("prompt_eval_count", 0), "input tokens"],
       ["prompt_eval_duration", f"{d.get('prompt_eval_duration', 0) / ns:.3f} s", "prefill time"],
       ["eval_count", eval_n, "output tokens, thinking included"],
       ["eval_duration", f"{eval_s:.3f} s", f"decode time → {eval_n / eval_s if eval_s else 0:.1f} tok/s server-side"],
       ["total_duration", f"{d.get('total_duration', 0) / ns:.2f} s", f"(wall clock seen by this script: {wall:.2f} s)"]],
      ["native field", "value", "meaning"])
native_think_tokens, native_think_answer = eval_n, repr(msg.get("content", "").strip())

step(2, "OpenAI-compatible API, same question")
o, wall = openai_chat({})
m = (o.get("choices") or [{}])[0].get("message") or {}
u = o.get("usage") or {}
print(f"· content    {(m.get('content') or '').strip()!r}")
print(f"~ reasoning  {len(m.get('reasoning') or '')} chars (Ollama's /v1 puts thinking in a `reasoning` field)")
print(f"◆ usage: prompt_tokens={u.get('prompt_tokens')} completion_tokens={u.get('completion_tokens')} · "
      f"{wall:.2f} s wall · no server timings in the OpenAI format")
openai_think_tokens, openai_think_answer = u.get("completion_tokens", 0), repr((m.get("content") or "").strip())

step(3, "thinking off, both ways")
d2, _ = native_chat(False)
o2, _ = openai_chat({"reasoning_effort": "none"})
m2 = (o2.get("choices") or [{}])[0].get("message") or {}
table([["native", '"think": true', native_think_tokens, native_think_answer],
       ["native", '"think": false', d2.get("eval_count", 0), repr((d2.get("message") or {}).get("content", "").strip())],
       ["OpenAI /v1", "(default)", openai_think_tokens, openai_think_answer],
       ["OpenAI /v1", '"reasoning_effort": "none"', (o2.get("usage") or {}).get("completion_tokens", 0),
        repr((m2.get("content") or "").strip())]],
      ["API", "setting", "output tokens", "answer"])
note("Thinking off: the answer costs a handful of tokens. Thinking on: the reasoning can use up max_tokens before "
     "the answer starts — an empty answer with output tokens == max_tokens is the sign. That is why sparkkit "
     "sends reasoning_effort=none unless a lab asks for thinking.")
note("OpenAI-style clients (LiteLLM, NAT, agent frameworks) do not know Ollama's `think` field; "
     "`reasoning_effort` is the standard knob, and Ollama's /v1 honours \"none\".")

result(f"native /api/chat gives server timings (eval_count ÷ eval_duration = real tok/s); /v1 gives the OpenAI "
       f"shape every client understands. Source: {who}.")

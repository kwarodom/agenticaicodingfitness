#!/usr/bin/env python3
"""Lab 06-3 · SGLang's two party tricks: schema-constrained JSON, and prefix caching you can see.

Runs the SGLang playbook's Steps 6 and 7 from Python:
  1. structured output — the playbook's `response_format: json_schema` request; the lab parses the reply and
     checks it against the schema itself (valid JSON, required keys, integer years);
  2. prefix caching — the playbook's two-turn physics-tutor conversation; turn 2 repeats turn 1's prefix, so
     with `--enable-cache-report` SGLang reports `usage.prompt_tokens_details.cached_tokens` > 0.

Against SGLang on your Spark (:30000) when it is up, otherwise against Ollama on this laptop, labelled
LAPTOP STAND-IN (Ollama also accepts json_schema; whether it reports cached tokens is up to its version).

Run: .venv/bin/python week25/06_sglang_trtllm_nim/labs/lab03_sglang_features.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import (banner, http_json, models, note, ok, pick_laptop_model, resolve, result, step,  # noqa: E402
                      warn)

SCHEMA = {"type": "object",
          "properties": {"languages": {"type": "array", "items": {
              "type": "object",
              "properties": {"name": {"type": "string"}, "primary_use": {"type": "string"},
                             "year_created": {"type": "integer"}},
              "required": ["name", "primary_use", "year_created"]}}},
          "required": ["languages"]}
SYSTEM = ("You are an expert physics tutor who explains concepts clearly and concisely. You use real-world analogies "
          "and everyday examples to make abstract ideas concrete. When answering, first state the key concept in one "
          "sentence, then give a short explanation with an example.")
TURN1_ANSWER = ("Speed is a scalar quantity that measures how fast an object moves, while velocity is a vector quantity "
                "that includes both speed and direction. For example, a car driving at 60 km/h has a speed of 60 km/h "
                "regardless of where it is headed. But if that car is driving 60 km/h north, that is its velocity — "
                "change direction to south and the velocity changes even though the speed stays the same.")


def post(base: str, model: str, messages: list, max_tokens: int, extra: dict) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0, **extra}
    return http_json("POST", base.rstrip("/") + "/chat/completions", body, timeout=240)


def schema_errors(obj) -> list[str]:
    errs = []
    if not isinstance(obj, dict) or not isinstance(obj.get("languages"), list):
        return ["top level must be an object with a 'languages' array"]
    for i, item in enumerate(obj["languages"]):
        for k in ("name", "primary_use", "year_created"):
            if k not in item:
                errs.append(f"languages[{i}] is missing '{k}'")
        if "year_created" in item and not isinstance(item["year_created"], int):
            errs.append(f"languages[{i}].year_created is not an integer")
    return errs


banner("Lab 06-3 · SGLang features — structured JSON and prefix caching",
       "the SGLang playbook's Steps 6 and 7, checked in Python")
base, src = resolve("sglang")
if src == "dry":
    print("◈ DRY — no SGLang endpoint and no laptop Ollama. Start SGLang on your Spark (TUTORIAL Section 2).")
    sys.exit(0)
if src == "spark":
    model, extra, who = (models(base) or ["default"])[0], {}, f"SGLang on the Spark · {base}"
else:
    model, extra, who = pick_laptop_model(["gemma3:4b"]), {"reasoning_effort": "none"}, \
        "LAPTOP STAND-IN · Ollama on this Mac (not SGLang, not the Spark)"
print(f"◆ endpoint: {who} · model {model}")

step(1, "structured output — response_format json_schema (playbook Step 7; max_tokens 150, so answers are kept short)")
d = post(base, model, [{"role": "user", "content": "List three programming languages with their primary use and "
                                                   "year created. Keep each primary use under six words."}], 150,
         {**extra, "response_format": {"type": "json_schema", "json_schema": {"name": "languages", "schema": SCHEMA}}})
content = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
print("· ANSWER  " + content.replace("\n", " ")[:300])
try:
    obj = json.loads(content)
    errs = schema_errors(obj)
    if errs:
        warn("parsed as JSON but breaks the schema: " + "; ".join(errs))
    else:
        ok(f"valid JSON matching the schema · {len(obj['languages'])} languages · "
           + ", ".join(f"{x['name']} ({x['year_created']})" for x in obj["languages"]))
except json.JSONDecodeError as e:
    warn(f"not valid JSON ({e}) — max_tokens may have cut it off; the playbook uses 512")

step(2, "prefix caching — the playbook's two-turn conversation (Step 6)")
turn1 = [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "What is the difference between speed and velocity?"}]
turn2 = turn1 + [{"role": "assistant", "content": TURN1_ANSWER},
                 {"role": "user", "content": "Can you give me another example that shows why the distinction "
                                             "matters in real physics problems?"}]
for label, msgs in (("turn 1", turn1), ("turn 2", turn2)):
    u = post(base, model, msgs, 64, extra).get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    print(f"│ {label}: prompt_tokens {u.get('prompt_tokens', '?'):>4} · cached_tokens "
          f"{cached if cached is not None else 'n/a (not reported)'}")
note("On SGLang, turn 2's cached_tokens should be > 0: RadixAttention reused turn 1's prefix instead of "
     "recomputing it. The playbook's other signal: `docker logs sglang-server 2>&1 | grep \"cached-token\"`.")
note("Hybrid Mamba/SSM models (e.g. Qwen3.6-35B-A3B) always report 0 — the playbook says to test prefix caching "
     "with a standard-attention model such as Qwen/Qwen3-8B.")
if src == "laptop":
    note("LAPTOP STAND-IN: Ollama keeps its own prompt cache, so it may report cached tokens too. That shows the "
         "API field, not SGLang's RadixAttention.")
result("Schema-constrained output saves you a retry loop around json.loads(); prefix caching makes every "
       "agent turn cheaper because the long system prompt and history are computed once.")

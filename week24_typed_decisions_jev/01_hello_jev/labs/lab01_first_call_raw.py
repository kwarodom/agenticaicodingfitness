#!/usr/bin/env python3
"""Lab 01-1 · Your first Jev call — standard library only, nothing hidden.

This file deliberately does NOT use jevkit. It shows the whole API:
one HTTPS POST with {model, state, questions}, one JSON answer back.

Run:  .venv/bin/python week24/01_hello_jev/labs/lab01_first_call_raw.py
DRY:  JEV_MODE=dry … (prints the request, replays a recorded answer)
"""
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[3]          # …/agenticaicodingfitness
ENDPOINT = "https://api.typesafe.ai/v1/systemone"


def load_key() -> str:
    """Environment first, then the repo-root .env. Never print the key."""
    key = os.environ.get("TYPESAFE_API_KEY", "")
    if not key and (ROOT / ".env").is_file():
        for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key


# ── 1. build the request: model + state + questions ─────────────────────────
payload = {
    "model": os.environ.get("JEV_MODEL", "jev-1.13.0"),
    "state": {"message": "Room 1203 is too warm and the AC is blowing warm air."},
    "questions": {
        # "is_hvac" is OUR label for the answer — Jev never sees this key.
        "is_hvac": {
            "type": "noul",
            "instructions": "Is `message` about air conditioning or cooling?",
        }
    },
}

print("━━ STEP 1 · the request we send (this is ALL of it)")
print(json.dumps(payload, indent=2, ensure_ascii=False))

key = load_key()
dry = os.environ.get("JEV_MODE", "").lower() == "dry" or not key

# ── 2. send it ───────────────────────────────────────────────────────────────
print(f"\n━━ STEP 2 · POST {ENDPOINT}")
if dry:
    # DRY mode: replay the answer jev-1.13.0 gave to this exact request.
    print("◈ DRY — not sending. Replaying the answer recorded from jev-1.13.0:")
    result = {"model": "jev-1.13.0", "answers": {"is_hvac": {"type": "noul", "noul": 0.99}},
              "usage": {"input_tokens": 299, "output_tokens": 20}}
else:
    request = Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urlopen(request, timeout=30) as response:
            result = json.load(response)
    except HTTPError as exc:
        print(f"✕ HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}")
        sys.exit(1)
    print(f"✓ HTTP 200 in {(time.perf_counter() - t0) * 1000:.0f} ms")

# ── 3. read the answer ───────────────────────────────────────────────────────
print("\n━━ STEP 3 · the raw response")
print(json.dumps(result, ensure_ascii=False))

p = result["answers"]["is_hvac"]["noul"]
tokens = result["usage"]["input_tokens"]
print(f"\n═ is_hvac = {p:.2f} → the model is {p:.0%} sure the message is about air conditioning.")
print(f"◆ you paid for {tokens} input tokens × $0.042/M = ${tokens * 0.042 / 1e6:.7f} (output is free)")
print(f"◆ answered by {result['model']} — log this so you know which version made each decision")

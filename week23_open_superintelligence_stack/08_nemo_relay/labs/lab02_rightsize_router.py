#!/usr/bin/env python3
"""Lab 02 — write a right-sizing router and measure it against real calls.

demos/step03_router.py shows scripted routing decisions; here YOU route. A
20-line heuristic router classifies each request easy/medium/hard, picks a
small or big model FROM YOUR LIVE ENDPOINT, makes the real call, and measures
latency + tokens. Then it prices the batch two ways — relay-routed vs
"always the big model" — using clearly-labeled illustrative $/M-token tiers.

Run:  cd <repo root> && .venv/bin/python week23/08_nemo_relay/labs/lab02_rightsize_router.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# illustrative $/M OUTPUT tokens per tier — labels for the math, NOT live prices
PRICE = {"small": 0.05, "big": 2.50}
DEADLINE_S = 55          # whole-lab budget: cold Ollama loads must not stack past ~60 s

REQUESTS = [
    ("Classify this ticket as HARDWARE or SOFTWARE — answer one word only: "
     "'the chiller controller reboots whenever the firmware updater runs'", 40),
    ("Extract just the room number from: 'guest in 1207 reports the AC is stuck at 26C'", 40),
    ("A pump's power draw rose 8% while flow dropped 5% over two weeks. Reason step by "
     "step about the two most likely causes and propose one test to tell them apart.", 260),
]

EASY_HINTS = ("classify", "extract", "one word", "route", "yes or no")
HARD_HINTS = ("reason", "step by step", "debug", "propose", "why", "prove")


def route(prompt: str) -> str:
    p = prompt.lower()
    if any(h in p for h in HARD_HINTS) or len(p) > 400:
        return "hard"
    if any(h in p for h in EASY_HINTS) and len(p) < 220:
        return "easy"
    return "medium"


def pick_models() -> tuple[str, str]:
    """small backend for easy/medium, big backend for hard — from the LIVE endpoint."""
    avail = config.list_local_models()
    big = config.MODEL
    for pat in ("gemma3", "llama3.2", "llama3.1:8b", "qwen3:4b", "phi", "mini",
                "nano", ":8b", ":4b", ":3b", "granite"):
        for m in avail:
            if pat in m.lower() and m != big:
                return m, big
    return big, big     # only one model installed — router still runs, savings become $0


def call(model: str, prompt: str, max_tokens: int, budget: float) -> dict:
    from openai import OpenAI, BadRequestError
    # no retries; per-call timeout = whatever remains of the LAB's 55 s budget
    # (a COLD Ollama model load alone can take ~40 s — one is fine, three stack)
    client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY,
                    timeout=max(min(45.0, budget), 5.0), max_retries=0)
    kw = dict(model=model, temperature=0.0, max_tokens=max_tokens,
              messages=[{"role": "user", "content": prompt}])
    t0 = time.time()
    try:
        try:    # mute thinking-model reasoning so terse answers stay terse;
            resp = client.chat.completions.create(   # endpoints that reject the
                **kw, extra_body={"reasoning_effort": "none"})   # knob get a plain
        except BadRequestError:                                  # retry.
            resp = client.chat.completions.create(**kw)
    except Exception as e:                       # timeout/refused — don't traceback
        return {"text": f"(call failed: {type(e).__name__} — cold load? re-run)",
                "out_tok": 0, "s": time.time() - t0}
    u, msg = resp.usage, resp.choices[0].message
    # thinking models may put text in an extra `reasoning` field — don't print blank
    return {"text": (msg.content or (msg.model_extra or {}).get("reasoning") or "").strip(),
            "out_tok": getattr(u, "completion_tokens", 0) or 0,
            "s": time.time() - t0}


def main() -> None:
    print("━" * 64)
    print("  ▣ Lab 02 — right-sizing router, measured on real calls")
    print("━" * 64)
    print(f"\n  endpoint: {config.safe_base_url()}   mode: {config.MODE}\n")

    if config.MODE != "real":
        print("◈ [no endpoint — showing expected output] The router logic below still ran;")
        print("  the calls did not. Start an endpoint (see lab01's banner) and re-run.\n")
        for prompt, _ in REQUESTS:
            print(f"  → route({route(prompt):>6})  « {prompt[:70]}…")
        print("\n  Expected (sample — real numbers vary):")
        print("    [easy  ] gemma3:12b        1.2s   14 tok   « HARDWARE")
        print("    [hard  ] qwen3.6:35b…      9.8s  231 tok   « likely impeller wear vs…")
        print("    routed $0.000031 vs all-big $0.000672 → 95% cheaper (illustrative $/Mtok)")
        return

    small, big = pick_models()
    print(f"▣ backends picked from your endpoint — small: {small}   big: {big}")
    if small == big:
        print("  ◈ only one model installed — pull a small one (ollama pull gemma3:12b)")
        print("    to see real savings; the router still routes.\n")
    else:
        print()

    rows, t0 = [], time.time()
    for prompt, max_tokens in REQUESTS:
        tier = route(prompt)
        model = big if tier == "hard" else small
        print(f"▣ route({tier:>6}) → {model}")
        print(f"    » {prompt[:96]}…" if len(prompt) > 96 else f"    » {prompt}")
        left = DEADLINE_S - (time.time() - t0)
        if left < 8:      # cold loads ate the budget — stop honestly, don't hang
            print("    · (skipped — lab deadline reached; the endpoint is warming the")
            print("       models NOW, so just re-run: warm runs finish in seconds)\n")
            continue
        r = call(model, prompt, max_tokens, left)
        rows.append((tier, model, r))
        print(f"    · {r['s']:.1f}s · {r['out_tok']} out-tok · « {r['text'][:110]}\n")

    routed = sum(r["out_tok"] * PRICE["small" if t != "hard" else "big"] / 1e6
                 for t, _, r in rows)
    all_big = sum(r["out_tok"] * PRICE["big"] / 1e6 for _, _, r in rows)
    print("◈ pricing the SAME measured tokens two ways (illustrative $/M-token tiers):")
    print(f"    relay-routed   ${routed:0.6f}")
    print(f"    always-big     ${all_big:0.6f}")
    if all_big > 0:
        print(f"    → {(1 - routed / all_big) * 100:.0f}% cheaper at (claimed) equal outcome.")
    print("\n✓ Honesty check: 'equal outcome' is a CLAIM until an eval proves it —")
    print("  App 09 (inference economics) gives you the LLM-judge to test exactly that.")


if __name__ == "__main__":
    main()

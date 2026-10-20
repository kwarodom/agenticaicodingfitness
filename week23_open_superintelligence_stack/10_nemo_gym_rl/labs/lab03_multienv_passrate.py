#!/usr/bin/env python3
"""Lab 03 · Multi-environment pass-rate — the evaluation gate before promotion.

Three tiny environments (math exact-match, JSON tool-call check, output
constraint), each with its OWN deterministic verifier. Measure the base model's
pass-rate per environment and overall, then apply a promotion gate — the same
measurement that decides whether a GRPO-trained policy replaces the incumbent
(and what App 11's Data Flywheel automates).

Run:  cd <repo root> && .venv/bin/python week23/10_nemo_gym_rl/labs/lab03_multienv_passrate.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

TIMEOUT_S = 25
MAX_TOKENS = 250
DEADLINE_S = 50
GATE = 0.60               # promote only if overall pass-rate ≥ this


def _clean(text: str) -> str:
    return re.sub(r"<think>.*?</think>", " ", text, flags=re.S)


def v_math(gold: str, ans: str) -> float:
    nums = re.findall(r"-?\d+(?:\.\d+)?", _clean(ans))
    return 1.0 if nums and nums[-1] == gold else 0.0


def v_toolcall(gold: str, ans: str) -> float:
    """Last {...} block must parse as JSON with tool == gold and a 'time' key."""
    blocks = re.findall(r"\{[^{}]*\}", _clean(ans))
    if not blocks:
        return 0.0
    try:
        d = json.loads(blocks[-1])
    except Exception:  # noqa: BLE001
        return 0.0
    return 1.0 if d.get("tool") == gold and "time" in d else 0.0


def v_oneword(gold: str, ans: str) -> float:
    words = _clean(ans).strip().rstrip(".!").split()
    return 1.0 if words and words[-1].lower() == gold else 0.0


ENVS = [  # (env name, verifier, [(prompt, gold), ...])
    ("math_verify", v_math, [
        ("What is 41 * 12? Reply with the number only.", "492"),
        ("What is 900 / 4? Reply with the number only.", "225")]),
    ("tool_call", v_toolcall, [
        ('Book the 2pm meeting room. Reply with ONLY this JSON, filled in: '
         '{"tool": "reserve_room", "time": "HH:MM"}', "reserve_room"),
        ('Cancel the 4pm booking. Reply with ONLY this JSON, filled in: '
         '{"tool": "cancel_room", "time": "HH:MM"}', "cancel_room")]),
    ("constraint", v_oneword, [
        ("What color is a stop sign? Answer with exactly one word.", "red"),
        ("Capital of France? Answer with exactly one word.", "paris")]),
]


def _ask(prompt: str) -> str:
    from openai import BadRequestError, OpenAI, UnprocessableEntityError
    client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=TIMEOUT_S)
    kw = dict(model=config.MODEL, temperature=0.2, max_tokens=MAX_TOKENS,
              messages=[{"role": "user", "content": prompt}])
    try:    # ask thinking models to skip the trace — 6 tasks must fit the deadline
        r = client.chat.completions.create(
            extra_body={"reasoning_effort": "none"}, **kw)
    except (BadRequestError, UnprocessableEntityError):
        r = client.chat.completions.create(**kw)   # server rejects the hint → plain
    msg = r.choices[0].message
    return (str(getattr(msg, "reasoning", "") or "") + " " + (msg.content or "")).strip()


def report(rates: dict[str, float]) -> None:
    if not rates:
        print("\n  ✗ deadline hit before any environment finished — re-run on a faster endpoint.")
        return
    print(f"\n  {'Environment':<14}{'pass-rate':>10}")
    print("  " + "─" * 40)
    for name, rate in rates.items():
        bar = "█" * int(round(rate * 16)) + "·" * (16 - int(round(rate * 16)))
        print(f"  {name:<14}{rate:>9.0%}  {bar}")
    overall = sum(rates.values()) / len(rates)
    print("  " + "─" * 40)
    verdict = "PROMOTE ✓" if overall >= GATE else "HOLD ✗ (below gate)"
    print(f"  {'OVERALL':<14}{overall:>9.0%}  gate ≥{GATE:.0%} → {verdict}")


def main() -> None:
    print("━" * 64)
    print("  ▣ Lab 03 — multi-env pass-rate: the promotion gate")
    print("━" * 64)
    print("\n◈ Three environments, three DIFFERENT verifiers — one policy must pass all")
    print("  kinds. Training on a mix is what stops overfitting to a single verifier.")

    if config.MODE != "real":
        print("\n◈ No live endpoint — start one, then re-run:")
        print("    ollama pull qwen3.6 && export DGX_CONN=local     # C-path")
        print("    export DGX_CONN=cloud DGX_API_KEY=nvapi-...      # B-path")
        print("\n  [no endpoint — showing expected output]")
        report({"math_verify": 0.5, "tool_call": 1.0, "constraint": 1.0})
        print("\n  ✓ (expected) this is the BASE model's score — GRPO's job is to raise it.")
        return

    print(f"\n◈ Evaluating {config.MODEL} @ {config.safe_base_url()} ({config.cost_note()})")
    start = time.time()
    scores: dict[str, list[float]] = {name: [] for name, _, _ in ENVS}
    out_of_time = False
    # Round-robin: task i of EVERY env before task i+1 — a cold or slow endpoint
    # still scores BREADTH (one task per verifier) before the deadline cuts depth.
    for i in range(max(len(t) for _, _, t in ENVS)):
        for name, verifier, tasks in ENVS:
            if i >= len(tasks):
                continue
            if out_of_time := time.time() - start > DEADLINE_S:
                break
            prompt, gold = tasks[i]
            try:
                r = verifier(gold, _ask(prompt))
            except Exception as e:  # noqa: BLE001
                print(f"  [{name:<12}] request failed ({type(e).__name__}) — scored 0.0")
                r = 0.0
            scores[name].append(r)
            print(f"  [{name:<12}] {prompt[:46]!r:<50} → {r:.1f} {'✓' if r == 1.0 else '✗'}")
        if out_of_time:
            print(f"  ⏱ {DEADLINE_S}s budget reached — scoring what finished.")
            break
    if skipped := [n for n, s in scores.items() if not s]:
        print(f"  (unscored — no task finished in time: {', '.join(skipped)})")

    report({n: sum(s) / len(s) for n, s in scores.items() if s})
    print("\n✓ This table is the EVAL GATE: run it on base vs GRPO-trained, promote only")
    print("  the winner. App 11 (Data Flywheel) automates exactly this loop — logs →")
    print("  curate → customize → evaluate → promote. That's the next tutorial.")


if __name__ == "__main__":
    main()

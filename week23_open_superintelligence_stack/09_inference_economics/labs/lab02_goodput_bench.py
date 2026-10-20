#!/usr/bin/env python3
"""Lab 02 — goodput: measure a real success rate, compute cost per SUCCESSFUL task.

Four self-verifying micro-tasks (exact answers — no judge needed) run against
your live endpoint. We grade each reply, count tokens per attempt, and compute

    cost/success = ($/Mtok × tokens/attempt ÷ 1e6) ÷ success_rate

then a sensitivity table shows why success rate dominates sticker price.

Run:  cd <repo root> && .venv/bin/python week23/09_inference_economics/labs/lab02_goodput_bench.py
Override the per-token price:  LAB_MTOK=0.45 (illustrative $/1M output tokens)
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

MTOK = float(os.environ.get("LAB_MTOK", "0.45"))   # illustrative $/1M output tokens
DEADLINE_S = 50                                    # whole-lab wall-clock budget

# (prompt, expected) — graded by word-boundary match, so terse OR chatty replies pass.
TASKS = [
    ("What is 17 * 23? Reply with only the number.", "391"),
    ("Reverse the letters of the word 'stack'. Reply with only the reversed word.", "kcats"),
    ("Is 91 a prime number? Reply with only YES or NO.", "NO"),
    ("What is the capital of Australia? Reply with only the city name.", "Canberra"),
]


def p(line: str = "") -> None:
    print(line, flush=True)


def graded(reply: str, expected: str) -> bool:
    clean = re.sub(r"<think>.*?</think>", " ", reply, flags=re.S)   # drop reasoning trace
    return re.search(rf"\b{re.escape(expected)}\b", clean, re.I) is not None


def sensitivity(toks: float) -> None:
    p("◈ Step 3 — sensitivity: same price per token, different success rates")
    p(f"  {'success rate':<14}{'expected tries':>15}{'$/SUCCESS':>12}")
    p("  " + "─" * 42)
    for rate in (0.35, 0.70, 0.95):
        per_attempt = MTOK * toks / 1e6
        p(f"  {rate:<13.0%}{1 / rate:>15.1f}{per_attempt / rate:>11.6f}$")
    p("  → dropping from 95% to 35% success nearly TRIPLES the cost per result —")
    p("    a 'cheaper per token' model loses on the only metric that matters.")


def main() -> None:
    p("━" * 64)
    p("  ▣ LAB 02 — goodput bench: cost per SUCCESSFUL task")
    p("━" * 64)
    p(f"  connection: {config.CONN} ({config.conn_human()}) · model: {config.MODEL}\n")

    if config.MODE != "real":
        p("✗ no endpoint reachable — start one, then re-run:")
        p("    C · ollama serve && ollama pull qwen3:4b")
        p("    B · export DGX_CONN=cloud DGX_CLOUD_URL=https://integrate.api.nvidia.com/v1 \\")
        p("            DGX_API_KEY=nvapi-...")
        p("\n[no endpoint — showing expected output]")
        p("  17 * 23 → '391' ✓ PASS (41 tok, 1.2s) … 4/4 tasks, success 100%")
        p("  $/attempt 0.000018$ · $/SUCCESS 0.000018$ — then the sensitivity table.")
        p("  (SAMPLE only — your model may fail tasks, which is the whole point)")
        sensitivity(300)
        return

    import openai
    from openai import OpenAI
    client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY,
                    timeout=25.0, max_retries=0)

    def ask(prompt: str, budget: float):
        """Curb thinking-model traces; drop the extra param if the server 400s it."""
        kw = dict(model=config.MODEL, temperature=0.0, max_tokens=300,
                  timeout=budget, messages=[{"role": "user", "content": prompt}])
        try:
            return client.chat.completions.create(**kw, extra_body={"reasoning_effort": "none"})
        except openai.BadRequestError:
            return client.chat.completions.create(**kw)

    p(f"◈ Step 1 — running {len(TASKS)} self-verifying tasks against the live endpoint")
    start, passes, tok_counts = time.time(), 0, []
    for prompt, expected in TASKS:
        left = DEADLINE_S - (time.time() - start)
        if left < 3:
            p("  ⏱ lab wall-clock budget hit — scoring what we have.")
            break
        try:
            resp = ask(prompt, min(25.0, left))
        except (openai.APITimeoutError, openai.APIConnectionError):
            tok_counts.append(300)                 # you still paid for the attempt
            p(f"  {'✗ FAIL':<8} {prompt[:44]:<46} (⏱ timeout — a too-slow answer is a failed task)")
            continue
        except Exception as e:  # noqa: BLE001
            p(f"  ✗ call failed ({type(e).__name__}) — 404=wrong URL/model · 401=key")
            break
        msg = resp.choices[0].message
        reply = str(getattr(msg, "reasoning", "") or "") + " " + (msg.content or "")
        toks = getattr(resp.usage, "completion_tokens", None) or max(1, round(len(reply) / 4))
        ok = graded(reply, expected)
        passes += int(ok)
        tok_counts.append(toks)
        tail = re.sub(r"\s+", " ", reply).strip()[-40:]
        p(f"  {'✓ PASS' if ok else '✗ FAIL':<8} {prompt[:44]:<46} ({toks} tok) …{tail!r}")

    n = len(tok_counts)
    if not n:
        return
    rate = passes / n
    mean_toks = sum(tok_counts) / n
    per_attempt = MTOK * mean_toks / 1e6
    per_success = per_attempt / max(rate, 1e-6)
    p(f"\n◈ Step 2 — the goodput math (at ${MTOK:.2f}/Mtok, illustrative)")
    p(f"  success rate     {passes}/{n} = {rate:.0%}")
    p(f"  tokens/attempt   {mean_toks:.0f} (mean, measured)")
    p(f"  $/attempt        {per_attempt:.6f}$")
    if rate > 0:
        p(f"  $/SUCCESS        {per_success:.6f}$   (= $/attempt ÷ {rate:.0%})\n")
    else:
        p("  $/SUCCESS        ∞ — zero successes: infinitely expensive per result!\n")
    sensitivity(mean_toks)
    p("\n✓ done — lab03 handles tasks with NO exact answer: the LLM-as-judge.")


if __name__ == "__main__":
    main()

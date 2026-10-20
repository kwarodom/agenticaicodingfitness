#!/usr/bin/env python3
"""Lab 09-3 · What does Jev cost? Pure arithmetic, no API call.

Jev bills INPUT tokens only: $0.042 per million (output is free). This lab:
  1. prices 1 → 1,000,000 requests at an assumed 1,000 tokens each,
  2. re-prices them with the tokens you actually MEASURED in Lab 09-1,
  3. shows why batching questions into one call is cheaper,
  4. checks a workload against the published rate limits.

Run: .venv/bin/python week24/09_evaluate_and_cost/labs/lab03_cost_calculator.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evalkit as ev                     # noqa: E402
from jevkit import PRICE_PER_MTOK, step, table  # noqa: E402

print("━" * 72)
print("━━ Lab 09-3 · the cost calculator — arithmetic only, $0, no key needed")
print("━" * 72)


def usd(tokens: float) -> float:
    return tokens * PRICE_PER_MTOK / 1_000_000


def money(x: float) -> str:
    return f"${x:,.6f}" if x < 0.01 else f"${x:,.2f}"


# ── 1. the published price ──────────────────────────────────────────────────
step(1, f"list price: ${PRICE_PER_MTOK} per million input tokens · output free")
VOLUMES = [1, 10_000, 100_000, 1_000_000]
table([[f"{n:,}", f"{n * 1000:,}", money(usd(n * 1000))] for n in VOLUMES],
      ["requests", "input tokens @1,000 each", "classifier cost"])

# ── 2. with YOUR measured tokens ────────────────────────────────────────────
step(2, "re-price with the tokens Lab 09-1 actually measured")
recs = ev.load_run()
if recs and any(r.get("ok") for r in recs):
    ok = [r for r in recs if r.get("ok")]
    avg = sum(r["input_tokens"] for r in ok) / len(ok)
    print(f"◆ measured: {len(ok)} intent calls, avg {avg:.0f} input tokens (7 questions + guard text + state)")
else:
    avg = 863.0
    print("◈ no .runs/intent_answers.json yet — using 863 tokens/call (measured on 2026-09-27)")
table([[f"{n:,}", f"{n * avg:,.0f}", money(usd(n * avg))] for n in VOLUMES],
      ["requests", "input tokens", "classifier cost"])
print("→ Most of those tokens are the QUESTIONS, not the message. Shorter criteria = cheaper calls.")

# ── 3. batching: pay for the state once ─────────────────────────────────────
step(3, "batching — 7 questions in ONE call vs 7 separate calls")
# A simple, explicit model of a request's size. These are ASSUMPTIONS, chosen so
# that one short question ≈ 320 tokens (Lab 01-1 measured 299) and the 7-question
# intent call ≈ 860 (Lab 09-1 measured ~863). Measure yours with usage.input_tokens.
OVERHEAD = 200     # fixed per-request framing (tokens)
STATE = 30         # a short message
PER_QUESTION = 90  # an average question with its criteria
N_Q = 7
batched = OVERHEAD + STATE + N_Q * PER_QUESTION
separate = N_Q * (OVERHEAD + STATE + PER_QUESTION)
table([["one batched call", f"{batched:,}", f"${usd(batched) * 1e6:,.1f} per million msgs"],
       ["7 separate calls", f"{separate:,}", f"${usd(separate) * 1e6:,.1f} per million msgs"]],
      ["shape", "tokens / message", "cost"])
print(f"◆ batching saves {1 - batched / separate:.0%} here — and one round-trip instead of seven.")
print("→ The bigger the state (a long email, a document), the more batching saves.")

# ── 4. rate limits ──────────────────────────────────────────────────────────
step(4, "does the workload fit the published limits? (1,200 req/min · 250,000 tok/s)")
for per_day in (10_000, 500_000, 2_000_000):
    per_min_avg = per_day / 1440
    peak = per_min_avg * 5                       # assume a 5× peak-hour burst
    toks_per_s = peak * avg / 60
    ok = peak <= 1200 and toks_per_s <= 250_000
    print(f"{'✓' if ok else '⚠'} {per_day:>9,} req/day → peak ≈ {peak:,.0f} req/min, "
          f"{toks_per_s:,.0f} tok/s {'fits' if ok else '→ queue, batch, or ask for a higher limit'}")

print("\n═ This is the CLASSIFIER bill only. A real workflow also pays for the generative")
print("  model, retrieval, retries, hosting, engineering and human review — usually far more.")

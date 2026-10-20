#!/usr/bin/env python3
"""Lab 03-2 · Speculative fan-out — many questions, ONE call.

Jev ingests the state once and answers every question in parallel. So six
questions in one request cost far less than six requests — and return in
about the time of one. You can even ask "speculative" questions you might not
need (e.g. the billing question when it may be an HVAC ticket) and let your
code use only the relevant answers.

We measure it: 6 separate calls vs 1 batched call, same state, same questions.

Run: .venv/bin/python week24/03_question_design/labs/lab02_fanout_one_call.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from jevkit import ask, banner, choice, cost_usd, noul, score, step, table, validate  # noqa: E402

STATE = {"email": {
    "subject": "Lobby too hot + invoice question",
    "body": ("Hi team, since Monday the lobby at Riverside Hotel has been around 28°C in the "
             "afternoons and guests are complaining. Also, invoice INV-2291 seems to charge us "
             "twice for the September maintenance visit. Can someone call me back today? — Nok, Chief Engineer"),
}}

QUESTIONS = {
    "topic": choice("What is the main topic of `email.body`?", {
        "hvac": "Cooling, heating or ventilation performance",
        "billing": "Invoices, charges or payments",
        "other": "Anything else",
    }),
    "mentions_billing": noul("Does `email.body` raise an invoice or billing issue?"),
    "mentions_comfort": noul("Does `email.body` report guests or occupants being uncomfortable?"),
    "wants_callback": noul("Does the sender of `email` ask to be phoned or called back?"),
    "sender_is_engineer": noul("Does `email` say the sender works in an engineering role?"),
    "urgency": score("How urgent is `email`?", [
        "No time pressure", "Should be handled this week", "Needs a response today", "Emergency now"]),
}

banner("Lab 03-2 · speculative fan-out", "6 questions: separate calls vs one batched call")

step(1, "SIX separate calls — one question each")
t0 = time.perf_counter()
sep_tokens, sep_cost = 0, 0.0
for qid, q in QUESTIONS.items():
    r = ask(STATE, {qid: q}, quiet=True)
    validate(r, {qid: q})
    sep_tokens += r["usage"]["input_tokens"]
    sep_cost += cost_usd(r)
sep_ms = (time.perf_counter() - t0) * 1000
print(f"◆ 6 calls · {sep_tokens} input tokens · ${sep_cost:.6f} · {sep_ms:.0f} ms wall-clock")

step(2, "ONE call with all six questions")
t0 = time.perf_counter()
r = ask(STATE, QUESTIONS, quiet=True)
a = validate(r, QUESTIONS)
one_ms = (time.perf_counter() - t0) * 1000
one_tokens = r["usage"]["input_tokens"]
print(f"◆ 1 call  · {one_tokens} input tokens · ${cost_usd(r):.6f} · {one_ms:.0f} ms wall-clock")

step(3, "compare")
table([["separate (6 calls)", sep_tokens, f"{sep_ms:.0f}"],
       ["batched (1 call)", one_tokens, f"{one_ms:.0f}"],
       ["saving", f"{sep_tokens / max(one_tokens, 1):.1f}× fewer tokens", f"{sep_ms / max(one_ms, 1):.1f}× faster"]],
      ["approach", "input tokens", "ms"])
print("◆ (in DRY mode timings are ~0 ms — the token counts are the recorded real ones)")

step(4, "code picks the answers it needs — speculative ones are simply ignored")
if a["topic"]["choice"] == "hvac" or a["mentions_comfort"]["noul"] >= 0.8:
    print("→ open an HVAC comfort ticket")
if a["mentions_billing"]["noul"] >= 0.8:
    print("→ ALSO open a billing ticket (topic 'won' HVAC, but the billing noul caught it)")
if a["wants_callback"]["noul"] >= 0.8:
    print(f"→ schedule a same-day callback (urgency score {a['urgency']['score']:.2f})")
print("═ execute: False — tickets and calls are proposals only.")

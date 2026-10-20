#!/usr/bin/env python3
"""Lab 11-4 · Jev checks the LLM's draft — verify, then escalate.

An LLM can write a lovely reply that promises a refund nobody approved, or names a
technician who does not exist. So after the LLM writes, Jev answers four typed
questions about the DRAFT, code adds exact checks (is the room number there?), and
a deterministic policy decides: staff approval, or human review.

  A · a real draft from your chosen LLM          (usually passes)
  B · a deliberately bad draft we planted        (must be caught)

Run: .venv/bin/python week24/11_jev_plus_llm/labs/lab04_jev_checks_the_draft.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline as P  # noqa: E402
from jevkit import banner, show, step  # noqa: E402
import llmkit  # noqa: E402

MSG = P.GUESTS[0]
BAD_DRAFT = ("So sorry about the cold room! We will refund tonight's stay in full, and our technician "
             "Somchai will be at your door by 1:15am. Guest Services")

prov, model = llmkit.chosen()
banner("Lab 11-4 · Jev checks the draft", f"writer: {prov} ({model}) · checker: Jev")

step(1, "route the guest message (Jev) and collect exact facts (code)")
routed = P.route(MSG)
print(f"» team {routed['team']} · language {routed['language']} · room {routed['facts']['room']}")

drafts = []
step(2, f"A · a real draft from {prov}")
try:
    r = P.write_reply(MSG, routed, prov, model, max_tokens=620)   # own budget → own recordings
    llmkit.show(r)
    drafts.append(("A · LLM draft", r["text"]))
except llmkit.LLMError as e:
    print(f"✕ {prov}: {e} — continuing with the planted draft only")
print("\n   B · the planted bad draft:")
print("   │ " + BAD_DRAFT)
drafts.append(("B · planted draft", BAD_DRAFT))

for n, (name, draft) in enumerate(drafts, 3):
    step(n, f"Jev checks {name}")
    c = P.check_draft(MSG, routed, draft)
    show(c["resp"], top=2)
    d = c["decision"]
    if d["route"] == "human_review":
        print(f"→ HUMAN REVIEW — {len(d['reasons'])} problem(s):")
        for why in d["reasons"]:
            print(f"   ✕ {why}")
    else:
        print("✓ passes every check → goes to a staff member for one-click approval")

print("\n═ execute: False — Jev never sends; it only decides whether a person must look first.")

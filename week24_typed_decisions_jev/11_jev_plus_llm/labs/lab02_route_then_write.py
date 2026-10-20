#!/usr/bin/env python3
"""Lab 11-2 · Jev routes → code picks the APPROVED prompt → your LLM writes the reply.

For three guest messages (English, Thai, and an angry refund demand):
  1. Jev: department · urgency · wants_compensation      (one fast typed call)
  2. code: language, room number, the approved system prompt for that team
  3. your LLM: a short reply DRAFT for staff to approve   (execute: False)

Run: .venv/bin/python week24/11_jev_plus_llm/labs/lab02_route_then_write.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline as P  # noqa: E402
from jevkit import banner, step  # noqa: E402
import llmkit  # noqa: E402

prov, model = llmkit.chosen()
banner("Lab 11-2 · route, then write", f"Jev decides · {prov} ({model}) writes")

jev_ms, llm_ms = [], []
for n, msg in enumerate(P.GUESTS, 1):
    step(n, f"“{msg}”")
    routed = P.route(msg)
    a = routed["answers"]
    jev_ms.append(routed["resp"].get("_latency_ms") or 0)
    print(f"» department  {a['department']['choice']:<12} (confidence {a['department']['confidence']:.2f})"
          f"  → team: {routed['team']}{'' if routed['confident'] else '  (unsure → front desk)'}")
    print(f"» urgency     {a['urgency']['score']:.2f} on 0…3   · wants compensation {a['wants_compensation']['noul']:.2f}")
    print(f"» code facts  language={routed['language']} · room={routed['facts']['room']}")
    if a["wants_compensation"]["noul"] >= 0.5:
        print("→ compensation requested: the draft must NOT promise it — a manager is flagged separately")
    try:
        r = P.write_reply(msg, routed, prov, model)
    except llmkit.LLMError as e:
        print(f"✕ {prov}: {e}")
        continue
    llm_ms.append(r.get("latency_ms") or 0)
    llmkit.show(r)

print()
if jev_ms and llm_ms and all(jev_ms) and all(llm_ms):
    print(f"◆ Jev decision avg {sum(jev_ms) / len(jev_ms):.0f} ms · {prov} draft avg {sum(llm_ms) / len(llm_ms):.0f} ms "
          f"→ the typed decision is the cheap, fast step; writing is the slow one")
print("═ execute: False — every draft goes to a staff member for approval; nothing is sent.")

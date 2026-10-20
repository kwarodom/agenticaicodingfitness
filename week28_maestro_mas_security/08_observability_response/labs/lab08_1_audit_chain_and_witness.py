#!/usr/bin/env python3
"""Lab 08-1 · The audit chain and its witness: what a row must say, and what a hash chain can and cannot prove.

Offline and instant. The lab pins the SIM brain: it is about the log, not the model.

  1. four ordinary requests, logged with audit_chain on: what each row records, and what it is missing
  2. personal data in the log (MAESTRO L5 "Data Leakage through Observability"), and pseudonymising it
     BEFORE the row is chained
  3. three ways to tamper with the log: edit a middle row, delete the last row, rewrite the whole chain.
     The chain alone catches only the first. A witness head kept OUTSIDE the log catches all three.
  4. scenario A12 from the suite, with and without audit_chain

Everything is simulated: a fake hotel, fake guests, `.example` domains.

Run: .venv/bin/python week28_maestro_mas_security/08_observability_response/labs/lab08_1_audit_chain_and_witness.py
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ensemble import Defenses, Ensemble, _h  # noqa: E402
from maskit import (attack, banner, defend, dim, fail, note, ok, result, runs_dir, step,  # noqa: E402
                    table)
from opskit import RedactingEnsemble, pseudonym  # noqa: E402
from redteam import SCENARIOS, run_scenario  # noqa: E402

banner("Lab 08-1 · audit chain and witness", "what a row records · what a chain proves · why it needs a witness",
       brain=False)
dim("this lab pins the 🧠 SIM brain: it tests the log, not the model")

STAFF = "staff:duty-manager"
REQUESTS = [("guest:1203", "Room 1203 is too hot."), (STAFF, "Set the chiller to 7.5."),
            ("guest:1203", "Email my receipt to ken@mail.example."),
            ("guest:1203", "Room 1203 has a leak under the sink.")]


def show_rows(rows, n=None):
    table([[r["seq"], r["agent"], r["tool"], json.dumps(r["args"])[:44], r["principal"], r["outcome"][:20],
            r.get("prev", "—"), r.get("hash", "—")] for r in rows[:n]],
          ["seq", "agent", "tool", "args", "principal", "outcome", "prev", "hash"], maxw=44)


step(1, "four ordinary requests, logged with audit_chain on")
ens = Ensemble(Defenses(audit_chain=True), brain="sim")
for p, t in REQUESTS:
    ens.handle(t, p)
show_rows(ens.audit.rows)
note(f"witness head after the last request: {ens.witness}")
note("each row's hash covers its own fields AND the previous row's hash: change one row, every later link breaks")
table([
    ["agent", "✓", "which agent acted (repudiation: 'it wasn't me')"],
    ["tool + args", "✓", "what exactly it did"],
    ["principal", "✓", "WHO it acted for — the field that makes a confused deputy visible"],
    ["outcome", "✓", "ok, or the PEP's reason for a denial"],
    ["seq / prev / hash", "✓", "order and tamper evidence"],
    ["timestamp", "✕", "without it there is no 'rate', no timeline, no MTTD"],
    ["request / trace id", "✕", "which rows belong to one request; joins the log to the run trace"],
    ["all origins", "✕", "only the least-trusted principal is kept, not every text that steered the call"],
    ["delegations", "✕", "agent → agent hand-offs are not logged at all (Lab 08-2 meets this gap)"],
    ["model + version", "✕", "which brain decided (needed when a model update changes behaviour)"],
], ["field", "kit row", "why it matters in a multi-agent system"], maxw=84)

step(2, "personal data in the log (MAESTRO L5 · Data Leakage through Observability)")
row = next(r for r in ens.audit.rows if r["tool"] == "send_email")
attack(f"row {row['seq']} stores a guest's address in clear text: {row['args']}")
red = RedactingEnsemble(Defenses(audit_chain=True), brain="sim")
for p, t in REQUESTS:
    red.handle(t, p)
rrow = next(r for r in red.audit.rows if r["tool"] == "send_email")
defend(f"pseudonymised before chaining: {rrow['args']}")
ok(f"the redacted chain still verifies: {red.audit.verify(red.witness)[1]}")
q = pseudonym("ken@mail.example")
note(f"an investigator who holds the key can still ask 'did we email ken@mail.example?' → {q} → "
     f"{'yes, row ' + str(rrow['seq']) if q in rrow['args']['to'] else 'no'}")
dim("redact BEFORE the row is chained: editing a chained row later (say, for an erasure request) breaks the chain")

step(3, "three ways to tamper with the log")
witness = dict(ens.witness)              # in real life: a copy kept where the log's writer cannot reach
clean = copy.deepcopy(ens.audit.rows)
outcomes = []

attack("T1 · an insider edits a middle row: the chiller row now says 3 °C instead of 7.5 °C")
ens.audit.rows = copy.deepcopy(clean)
next(r for r in ens.audit.rows if r["tool"] == "set_chiller_setpoint")["args"]["celsius"] = 3.0
outcomes.append(["T1 edit a middle row", *[v[1] for v in (ens.audit.verify(), ens.audit.verify(witness))]])

attack("T2 · an insider deletes the LAST row (the work order) — nothing after it to break")
ens.audit.rows = copy.deepcopy(clean)
del ens.audit.rows[-1]
outcomes.append(["T2 delete the last row", *[v[1] for v in (ens.audit.verify(), ens.audit.verify(witness))]])

attack("T3 · an insider edits a middle row, then recomputes every hash after it")
ens.audit.rows = copy.deepcopy(clean)
next(r for r in ens.audit.rows if r["tool"] == "set_chiller_setpoint")["args"]["celsius"] = 3.0
prev = "0" * 16
for r in ens.audit.rows:                 # anyone who can write the log can also re-chain it
    r["prev"] = prev
    r["hash"] = _h(prev, {k: v for k, v in r.items() if k != "hash"})
    prev = r["hash"]
outcomes.append(["T3 edit + re-chain", *[v[1] for v in (ens.audit.verify(), ens.audit.verify(witness))]])
table(outcomes, ["tamper", "verify() — chain alone", "verify(witness) — chain + witness"], maxw=72)
result("the chain alone caught 1 of 3 · chain + witness caught 3 of 3")
note("a chain proves the rows agree with EACH OTHER. Only a head kept outside the log proves they agree with the past.")

step(4, "anchor the witness somewhere the log's writer cannot reach")
anchor = runs_dir(__file__) / "witness_anchor.json"
anchor.write_text(json.dumps({"system": "hotel-ops-ensemble", **witness}, indent=2), encoding="utf-8")
ok(f"wrote the head to {anchor.relative_to(anchor.parents[2])} — a stand-in for a real anchor")
table([
    ["another system", "send the head to a separate log service the agents have no credentials for"],
    ["a ticket", "paste the head into the change or incident ticket at the end of each shift"],
    ["WORM storage", "write-once object storage (retention lock): nobody can overwrite it, not even an admin"],
    ["a second party", "the vendor, an auditor or a peer hotel countersigns heads (a transparency log)"],
], ["anchor", "how"], maxw=90)
dim("a file next to the log is NOT an anchor: whoever can delete a row can rewrite this file too")

step(5, "scenario A12 from the suite: an audit row deleted after the fact")
a12 = next(s for s in SCENARIOS if s.id == "A12")
for d, label in [(Defenses(), "naive (plain log)"), (Defenses(audit_chain=True), "audit_chain on")]:
    o = run_scenario(a12, d)
    (fail if o.harms else ok)(f"{label:<18} → {', '.join(o.harms) or 'tamper detected by verify(witness)'}")
note("A12 deletes the set_room_setpoint row, which is the last row: only the witness can notice")

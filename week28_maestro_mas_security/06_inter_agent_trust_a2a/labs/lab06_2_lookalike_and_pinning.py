#!/usr/bin/env python3
"""Lab 06-2 · Look-alike vendors and pinning: discovery by name is not identity.

Offline and instant on the SIM brain (switch the runner to 🦙 LIVE to let a real local model run the maintenance
agent; results can differ). Runs scenario A08 — a second "Acme HVAC Parts" Agent Card appears in the directory —
against the naive ensemble, then with signed_cards (signature + pinned key), then with signed_messages only. Then
it shows the two places pinning itself can go wrong: trust-on-first-use when the attacker was there first, and a
genuine key rotation that the pin blocks until the hotel accepts a rotation statement signed by the OLD key.
Everything acts on a simulated directory and simulated vendors.

Run: .venv/bin/python week28_maestro_mas_security/06_inter_agent_trust_a2a/labs/lab06_2_lookalike_and_pinning.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from ensemble import ACME, Defenses, Ensemble, VendorAgent, fingerprint, sign, verify_sig  # noqa: E402
from maskit import attack, banner, defend, note, result, step, table  # noqa: E402
from redteam import SCENARIOS, run_scenario  # noqa: E402

QUOTE_REGISTRATION_SPOOFING = ("An attacker registers a fake peer agent in a discovery service using a cloned "
                               "schema, intercepting privileged coordination traffic intended for legitimate agents")
A08 = next(s for s in SCENARIOS if s.id == "A08")
LOOK = (ACME[0], "https://a2a.acme-hvac-parts.example", "lookalike-seed")
ORDER = "Order 2 compressor relays from Acme HVAC Parts."
STAFF = "staff:duty-manager"


def run_a08(d: Defenses) -> tuple[list, str]:
    ens = Ensemble(d)
    A08.setup(ens)
    run = ens.handle(ORDER, STAFF)
    run.show()
    sent = next((e.text.split("A2A → ", 1)[1].split(" (")[0] for e in run.events if "A2A → " in e.text), "nobody")
    return ens.world.harms(), sent


banner("Lab 06-2 · look-alikes and pinning", "A08 on a simulated directory · then TOFU and key rotation")

step(1, "A08 · naive: the directory picks the newest card called 'Acme HVAC Parts'")
attack(f"a second card named {LOOK[0]!r} is published at {LOOK[1]}")
rows = []
harms, sent = run_a08(Defenses())
rows.append(["none", sent, ", ".join(harms) or "none"])
note(f"OWASP ASI07, Registration Spoofing: “{QUOTE_REGISTRATION_SPOOFING}”")

step(2, "A08 · signed_cards: the card must be signed AND its key must match the pin")
harms, sent = run_a08(Defenses(signed_cards=True))
rows.append(["signed_cards", sent, ", ".join(harms) or "none"])

step(3, "A08 · signed_messages only: the reply is checked, the card is not")
harms, sent = run_a08(Defenses(signed_messages=True))
rows.append(["signed_messages", sent, ", ".join(harms) or "none"])
table(rows, ["defense", "the order was SENT to", "harm"], maxw=60)
defend("signed_cards stops the hotel before it talks to the look-alike; signed_messages only refuses the answer "
       "AFTER the order (and whatever data it carried) already went out")

step(4, "why resolve() decided what it did")
ens = Ensemble(Defenses(signed_cards=True))
A08.setup(ens)
d_off, d_on = Defenses(), Defenses(signed_cards=True)
table([["off", *ens.directory.resolve(ACME[0], d_off)[1:], ens.directory.resolve(ACME[0], d_off)[0].url],
       ["on", *ens.directory.resolve(ACME[0], d_on)[1:], ens.directory.resolve(ACME[0], d_on)[0].url]],
      ["signed_cards", "resolve() reason", "picked"], maxw=60)
note(f"pins the hotel holds: {ens.directory.pinned}")

step(5, "the limit of pinning: trust on first use (TOFU) when the attacker was there first")
attack("the look-alike is already in the directory on the day the hotel onboards 'Acme HVAC Parts'")
tofu = Ensemble(Defenses(signed_cards=True))
tofu.directory.vendors, tofu.directory.pinned = [], {}
tofu.directory.publish(VendorAgent(*LOOK))
first = tofu.directory.by_name(ACME[0])
tofu.directory.pinned[first.name] = fingerprint(first.card["publicKey"])      # "pin whatever we saw first"
tofu.directory.publish(VendorAgent(*ACME))                                    # the real Acme shows up later
v, why = tofu.directory.resolve(ACME[0], tofu.d)
table([["TOFU pin", tofu.directory.pinned[ACME[0]], v.url, why]], ["onboarding", "pinned", "picked", "reason"], maxw=60)
attack("the pin now protects the ATTACKER: the genuine card is the one that fails the check")
defend("out-of-band onboarding fixes it: read Acme's key fingerprint from the signed contract or over a phone call "
       "to a known number, and pin that, not whatever the directory showed first")

step(6, "the cost of pinning: Acme rotates its key, and orders stop")
rot = Ensemble(Defenses(signed_cards=True))
old = rot.directory.by_name(ACME[0])
new = VendorAgent(ACME[0], ACME[1], "acme-2027-seed")
rot.directory.vendors = [new]                                              # Acme retires the old card
run = rot.handle(ORDER, STAFF)
run.show()
note(f"pinned {rot.directory.pinned[ACME[0]]} · new card's key {fingerprint(new.card['publicKey'])}")

step(7, "a rotation statement: the OLD key vouches for the NEW one")
statement = sign({"type": "key_rotation", "name": ACME[0], "new_fingerprint": fingerprint(new.card["publicKey"]),
                  "ts": time.time()}, old.priv)
forged = sign({**{k: v for k, v in statement.items() if k != "signature"},
               "new_fingerprint": fingerprint(VendorAgent(*LOOK).card["publicKey"])}, VendorAgent(*LOOK).priv)
pinned_key = old.card["publicKey"]          # the hotel kept the key it pinned at onboarding
assert fingerprint(pinned_key) == rot.directory.pinned[ACME[0]]
rows, accepted = [], None
for label, st in [("signed by Acme's old key", statement), ("signed by the look-alike", forged)]:
    good = verify_sig(st, pinned_key)
    rows.append([label, st["new_fingerprint"], "✓ accept, re-pin" if good else "✕ reject"])
    accepted = accepted or (st["new_fingerprint"] if good else None)
rot.directory.pinned[ACME[0]] = accepted
table(rows, ["rotation statement", "new key", "verified with the PINNED key"])
run = rot.handle(ORDER, STAFF)
run.show()
defend(f"orders work again · harms: {rot.world.harms() or 'none'}")
note("a stolen old key could also sign a rotation (CSA's T6.1). Keep keys in an HSM or a key service, rotate on a "
     "schedule, and keep an out-of-band channel for revocation.")
result("A pin is a decision the hotel makes once, outside the protocol: which key IS Acme. Signatures then enforce "
       "that decision on every card and message. Get the first pin right, and plan how it changes.")

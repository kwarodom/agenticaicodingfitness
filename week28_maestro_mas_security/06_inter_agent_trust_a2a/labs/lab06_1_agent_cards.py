#!/usr/bin/env python3
"""Lab 06-1 · Agent Cards: what a signature proves, and what it does not.

Offline and instant: no agent thinks in this lab, so the brain switch does not matter. It inspects the genuine
Acme HVAC Parts Agent Card, runs the Week 17 style "are the fields there?" check, tampers one field at a time and
watches the Ed25519 signature fail, then builds a look-alike card that verifies perfectly against ITS OWN key, and
shows why that proves nothing until you compare the key with the one you pinned. All vendors are simulated.

Run: .venv/bin/python week28_maestro_mas_security/06_inter_agent_trust_a2a/labs/lab06_1_agent_cards.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from ensemble import ACME, Ensemble, VendorAgent, fingerprint, verify_sig  # noqa: E402
from maskit import attack, banner, defend, note, result, step, table  # noqa: E402

QUOTE_CARD_CHECK = ("Agents should cryptographically verify the Agent Card using the server's public key or DID, "
                    "ensuring that the card hasn't been tampered with.")


def short(card: dict) -> dict:
    """The card with its long base64 fields cut down, for printing."""
    return {k: (v[:12] + "…" if k in ("publicKey", "signature") else v) for k, v in card.items()}


def week17_check(card: dict) -> bool:
    """The Week 17 checkpoint-5 idea: are the required fields present? (Week 17 also required authSchemes;
    the kit's cards do not carry one, so it is left out here. The lesson is the same.)"""
    return all(card.get(f) for f in ("name", "url", "skills"))


banner("Lab 06-1 · Agent Cards", "integrity is not identity · a simulated vendor, Ed25519 from `cryptography`",
       brain=False)

step(1, "the genuine card, as the hotel's directory holds it")
ens = Ensemble()
acme = ens.directory.by_name(ACME[0])
print(json.dumps(short(acme.card), indent=2, ensure_ascii=False))
pin = ens.directory.pinned[acme.name]
note(f"key fingerprint {fingerprint(acme.card['publicKey'])} · pinned at onboarding: {pin}")
note(f"signature verifies against the card's own key: {verify_sig(acme.card, acme.card['publicKey'])}")

step(2, "the Week 17 check: are the fields there?")
look = VendorAgent(ACME[0], "https://a2a.acme-hvac-parts.example", "lookalike-seed")
table([["genuine", acme.card["url"], week17_check(acme.card)],
       ["look-alike", look.card["url"], week17_check(look.card)]],
      ["card", "url", "fields present?"])
attack("both pass: a field being present says nothing about who wrote it")

step(3, "tamper one field of the genuine card, keep its signature")
rows = []
for field, value in [("url", "https://a2a.acme-hvac-parts.example"), ("skills", ["parts_order", "refunds"]),
                     ("version", "1.1"), ("name", "Acme HVAC Parts Ltd")]:
    forged = {**acme.card, field: value}
    rows.append([field, json.dumps(value)[:40], "✓ verifies" if verify_sig(forged, forged["publicKey"]) else "✕ fails"])
table(rows, ["changed field", "new value", "signature"])
defend("every edit breaks the signature: the card is canonical JSON, and the signature covers all of it")

step(4, "a look-alike signs its OWN card with its OWN key")
attack(f"someone publishes {look.card['name']!r} at {look.card['url']}")
self_ok = verify_sig(look.card, look.card["publicKey"])
table([["genuine", fingerprint(acme.card["publicKey"]), verify_sig(acme.card, acme.card["publicKey"]),
        fingerprint(acme.card["publicKey"]) == pin],
       ["look-alike", fingerprint(look.card["publicKey"]), self_ok, fingerprint(look.card["publicKey"]) == pin]],
      ["card", "key fingerprint", "signature ✓ (own key)", "matches pin?"])
attack(f"the look-alike's signature verifies too ({self_ok}): anyone can make a key pair and sign a card with it")
defend("only the last column tells them apart: the key the hotel PINNED when it signed the Acme contract")

step(5, "the card is genuine, but who is answering? (DNS spoofing, a hijacked endpoint)")
attack("an attacker serves an exact copy of the genuine card, then answers the order with a confirmation of its own")
forged_reply = look.confirm("compressor relays", 2)
forged_reply = {**forged_reply, "from": ACME[0]}
copied_card = dict(acme.card)
table([["copied card", "signature + pin", verify_sig(copied_card, copied_card["publicKey"]) and
        fingerprint(copied_card["publicKey"]) == pin],
       ["attacker's reply", "signature vs pinned key", verify_sig(forged_reply, acme.card["publicKey"])]],
      ["what arrives", "checked against", "passes?"])
defend("a copied card passes every card check, because it IS the genuine card. Only a message the other side "
       "signs right now proves it holds the private key (Lab 06-3)")
note(f"CSA, A2A threat model (M3.4.6): “{QUOTE_CARD_CHECK}”")
result("A signature answers one question: was this card changed since the key-holder signed it? It never answers "
       "the other one: is this key-holder the Acme we do business with? That takes a pin (Lab 06-2).")

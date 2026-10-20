#!/usr/bin/env python3
"""Exercise 06 · Verify an A2A message: signed by the pinned key, fresh, single-use, addressed to us.

Fill in the four TODOs in verify_message(), save, then run:
    .venv/bin/python week28_maestro_mas_security/06_inter_agent_trust_a2a/exercises/ex06_verify_message.py

The checker is free, offline and instant. It feeds your function thirteen simulated vendor messages, one after
another, sharing one `seen_nonces` set: genuine ones, tampered, replayed, expired, from the future, misaddressed,
signed by a look-alike, from a vendor nobody pinned. Stuck? Compare with exercises/solutions/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from ensemble import ACME, VendorAgent, fingerprint, sign, verify_sig  # noqa: E402
from maskit import banner, check, finish  # noqa: E402

AUDIENCE = "altotech-grand"       # who the vendor must have addressed the message to
MAX_AGE = 300                     # seconds; a message more than this far from `now` (either way) is refused


def verify_message(msg: dict, pinned_fingerprints: dict, cards: list[dict], seen_nonces: set, now: float) -> str:
    """'' if the hotel should accept `msg`, else a short reason.

    msg                  a signed vendor message: from, type, order_id, item, qty, ts, nonce, aud, note, signature
    pinned_fingerprints  {vendor name: key fingerprint}, set by the hotel at onboarding
    cards                every Agent Card in the directory (anyone can publish one, with any name)
    seen_nonces          nonces of messages already ACCEPTED; add to it only when you accept
    now                  the current time (seconds since the epoch)
    """
    sender = str(msg.get("from", ""))

    # ── TODO 1 ── signature against the PINNED key. Find the card whose name is the sender AND whose key's
    #   fingerprint is the pinned one, then verify the message with that card's publicKey (verify_sig).
    #   No pin, or no such card → refuse.

    # ── TODO 2 ── freshness: refuse a missing timestamp, and any ts more than MAX_AGE seconds from `now`.

    # ── TODO 3 ── audience: the message must be addressed to AUDIENCE.

    # ── TODO 4 ── single use: refuse a missing or already-seen nonce. Record the nonce only now, after every
    #   other check passed, so a forged message cannot "use up" a genuine message's nonce.

    return ""


# ─────────────────────────── checker — no need to edit below ────────────────
NOW = 1_790_000_000.0


def battery() -> list[tuple[str, dict, bool, str]]:
    """(label, message, should_accept, hint) — delivered in this order."""
    acme = VendorAgent(*ACME)
    look = VendorAgent(ACME[0], "https://a2a.acme-hvac-parts.example", "lookalike-seed")
    bolt = VendorAgent("Bolt Valves Co", "https://a2a.bolt-valves.example", "bolt-seed")

    def msg(by, **change):
        body = {"from": by.name, "type": "order_confirmation", "order_id": "PO-0001", "item": "compressor relay",
                "qty": 2, "ts": NOW - 5, "nonce": "n-001", "aud": AUDIENCE, "note": "Order confirmed."}
        body.update(change)
        return sign({k: v for k, v in body.items() if v is not None}, by.priv)

    first = msg(acme)
    return [
        ("genuine, fresh", first, True, "TODO 1: a genuine Acme message must pass — look up the PINNED card's key"),
        ("the same message again", first, False, "TODO 4: a nonce may be used once"),
        ("tampered after signing (qty 2 → 20)", {**msg(acme, nonce="n-002"), "qty": 20}, False,
         "TODO 1: verify the signature over the whole message"),
        ("signed by a look-alike that calls itself Acme", msg(look, nonce="n-003", **{"from": ACME[0]}), False,
         "TODO 1: the look-alike's card is in `cards` too — use the one whose key matches the pin"),
        ("genuine, reusing the nonce the look-alike just tried", msg(acme, nonce="n-003", order_id="PO-0003"), True,
         "TODO 4: record a nonce only AFTER every check passes, or a forgery can burn a real nonce"),
        ("from a vendor nobody pinned (validly self-signed)", msg(bolt, nonce="n-004"), False,
         "TODO 1: no pin → no trusted key → refuse"),
        ("signed 301 s ago", msg(acme, nonce="n-005", ts=NOW - 301), False, "TODO 2: older than MAX_AGE"),
        ("signed 299 s ago", msg(acme, nonce="n-006", ts=NOW - 299, order_id="PO-0006"), True,
         "TODO 2: 299 s is inside the window — compare with > MAX_AGE, not ≥ something smaller"),
        ("dated 400 s in the future", msg(acme, nonce="n-007", ts=NOW + 400), False,
         "TODO 2: use abs(now − ts): a future timestamp is as wrong as an old one"),
        ("no timestamp at all", msg(acme, nonce="n-008", ts=None), False, "TODO 2: refuse a missing ts (no crash)"),
        ("addressed to another hotel", msg(acme, nonce="n-009", aud="riverside-hotel"), False,
         "TODO 3: aud must be AUDIENCE"),
        ("no nonce at all", msg(acme, nonce=None, order_id="PO-0010"), False, "TODO 4: refuse a missing nonce"),
        ("no signature at all", {k: v for k, v in msg(acme, nonce="n-011").items() if k != "signature"}, False,
         "TODO 1: an unsigned message never verifies"),
    ]


def main() -> None:
    banner("Exercise 06 · verify an A2A message", "offline checker · free · instant", brain=False)
    acme = VendorAgent(*ACME)
    look = VendorAgent(ACME[0], "https://a2a.acme-hvac-parts.example", "lookalike-seed")
    bolt = VendorAgent("Bolt Valves Co", "https://a2a.bolt-valves.example", "bolt-seed")
    cards = [look.card, acme.card, bolt.card]                     # the look-alike was published first
    pins = {ACME[0]: fingerprint(acme.card["publicKey"])}         # the hotel onboarded Acme only
    seen: set = set()
    good = True
    for label, m, want, hint in battery():
        try:
            why = verify_message(dict(m), pins, cards, seen, NOW)
        except Exception as e:  # noqa: BLE001 — a crash is a failed check, not a crashed checker
            why = None
            good &= check(False, "", f"{label}: verify_message crashed ({type(e).__name__}: {e}) — {hint}")
            continue
        accepted = why == ""
        good &= check(accepted == want,
                      f"{label} → {'accepted' if accepted else 'refused: ' + why}",
                      f"{label} → {'accepted' if accepted else 'refused: ' + why}, expected "
                      f"{'accept' if want else 'refuse'} · {hint}")
    finish(good, "Done. This is the receive side of A2A: what Defenses(signed_messages=True) switches on.")


if __name__ == "__main__":
    main()

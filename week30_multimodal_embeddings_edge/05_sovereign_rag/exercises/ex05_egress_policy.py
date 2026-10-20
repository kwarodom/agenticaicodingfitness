#!/usr/bin/env python3
"""Exercise 05 · Egress policy: decide which connections a sovereign pipeline may make.

Fill in the four TODOs, save, then run:
    week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/05_sovereign_rag/exercises/ex05_egress_policy.py

The checker is free and offline. Stuck? Compare with exercises/solutions/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from embedkit import banner, check  # noqa: E402

import ipaddress

ALLOWED_PORTS = {11434}                      # the local Ollama — the only service the RAG loop may reach


# ── TODO 1 ── may this connection go ahead? Loopback hosts ONLY ("localhost", 127.0.0.0/8, ::1) AND a port in
#   ALLOWED_PORTS. Hint: ipaddress.ip_address(h).is_loopback (it raises ValueError for names).
def allowed(host: str, port: int) -> bool:
    return True


# ── TODO 2 ── from a connection log, list "host:port" for every ALLOWED connection to a NON-loopback host
def leaked(log: list) -> list:
    return []


# ── TODO 3 ── which of these must stay on the machine in a sovereign RAG? (put their names in the set)
#   "voicemail_audio", "query_text", "vectors", "llm_prompt"
STAYS_LOCAL = set()

# ── TODO 4 ── in 1–3 sentences: why is an in-process guard a tripwire and not a firewall?
WHY_NOT_A_FIREWALL = """
"""


# ─────────────────────────── checker — no need to edit below ────────────────
def main() -> None:
    banner("Exercise 05 · egress policy", "offline checker · no network, no model")
    good = True
    cases = [("127.0.0.1", 11434, True), ("::1", 11434, True), ("localhost", 11434, True),
             ("127.0.0.1", 22, False), ("192.168.1.20", 11434, False), ("huggingface.co", 443, False),
             ("10.0.0.5", 11434, False), ("127.8.8.8", 11434, True)]
    try:
        got = [(h, p, allowed(h, p)) for h, p, _ in cases]
        wrong = [(h, p, a) for (h, p, a), (_, _, want) in zip(got, cases) if a != want]
    except Exception as e:  # noqa: BLE001
        wrong = f"error: {e}"
    good &= check(wrong == [], "allowed(): loopback (127.0.0.0/8, ::1, localhost) on port 11434 only",
                  f"TODO 1: wrong verdicts {wrong} — loopback AND the allowed port")
    log = [{"host": "127.0.0.1", "port": 11434, "allowed": True},
           {"host": "huggingface.co", "port": 443, "allowed": False},
           {"host": "api.example.com", "port": 443, "allowed": True},
           {"host": "::1", "port": 11434, "allowed": True}]
    try:
        leaks = leaked(log)
    except Exception as e:  # noqa: BLE001
        leaks = f"error: {e}"
    good &= check(leaks == ["api.example.com:443"], "leaked(): only ALLOWED connections to non-loopback hosts count",
                  f"TODO 2: leaked(log) = {leaks!r} (want ['api.example.com:443'])")
    good &= check(STAYS_LOCAL == {"voicemail_audio", "query_text", "vectors", "llm_prompt"},
                  "in sovereign RAG every artefact stays: audio, questions, vectors, prompts",
                  f"TODO 3: STAYS_LOCAL = {sorted(STAYS_LOCAL)} — which of these may leave a sovereign RAG?")
    t = WHY_NOT_A_FIREWALL.lower()
    good &= check(len(t.split()) >= 15 and any(w in t for w in ("process", "subprocess", "another program", "syscall",
                                                                    "outside", "bypass", "other code"))
                  and any(w in t for w in ("firewall", "network policy", "air gap", "air-gap", "os level", "openshell")),
                  "your note says what the in-process guard cannot see AND what enforces sovereignty for real",
                  "TODO 4: ≥ 15 words — what can bypass an in-process guard, and what should enforce the rule instead?")
    if not good:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)
    print("\n═ Done. Your allowed() is the same rule common/egress.py enforces in Labs 05 and 07.")


if __name__ == "__main__":
    main()

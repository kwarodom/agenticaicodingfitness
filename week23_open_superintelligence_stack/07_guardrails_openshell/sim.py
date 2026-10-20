#!/usr/bin/env python3
"""Guardrails + OpenShell simulator — learn how to SECURE a sovereign agent, no GPU.

Exposes the same tiny surface every Week 23 app's `view.py` expects
(installed_models(), tok_s(model), stream_generate(prompt, model)) PLUS the
domain helpers the guardrails demos need:

  • check_rails(text)      → a rail verdict (ALLOW / BLOCK) with the rail that fired
  • egress_allowed(host)   → is this host on the OpenShell network allowlist?
  • classify_privacy(text) → does this prompt contain PII/secrets → route LOCAL?
"""
from __future__ import annotations

import re
import time

# ── the local sovereign models this guarded runtime can serve (no GPU needed) ──
# Small/fast local NIMs used by the privacy router to keep sensitive data on-prem.
CATALOG = [
    ("nemotron-3-nano:30b-a3b",  "TensorRT-LLM", 54, "on-prem privacy-router target"),
    ("nemotron-3-super:120b-a12b", "TensorRT-LLM", 20, "on-prem heavy reasoning"),
    ("nemotron-guard:8b",        "TensorRT-LLM", 90, "input/output rail classifier"),
]
_TOK = {m: t for m, _, t, _ in CATALOG}


def installed_models() -> list[str]:
    return [m for m, *_ in CATALOG]


def tok_s(model: str) -> float:
    return float(_TOK.get(model, 40))


_CANNED = ("[simulated · guarded] Answer produced on a LOCAL sovereign NIM, wrapped by "
           "NeMo Guardrails (input + output + topic rails) and executed inside the "
           "OpenShell secure runtime — sandboxed tools, a network egress allowlist, and "
           "a signed policy enforced by the NemoClaw gateway. Nothing left the perimeter.")


def stream_generate(prompt: str, model: str):
    delay = min(0.04, 1.0 / max(tok_s(model), 1) * 1.3)
    for w in _CANNED.split(" "):
        yield w + " "
        time.sleep(delay)


# ── NeMo Guardrails (simulated) — input / topic / output rails ─────────────────
# Each rail is (rail_kind, human_name, compiled_pattern). The first match blocks.
_RAILS = [
    ("input", "jailbreak / prompt-injection",
     re.compile(r"ignore (all|the|previous|prior) (instructions|prompt)|"
                r"disregard .*(instruction|rule)|system prompt|developer message|"
                r"you are now|do anything now|\bDAN\b|reveal your (rules|instructions)",
                re.I)),
    ("output", "secret / credential exfiltration",
     re.compile(r"\b(api[_\- ]?key|secret|password|private key|token)\b|"
                r"nvapi-[a-z0-9]|sk-[a-z0-9]{6,}|-----BEGIN", re.I)),
    ("topic", "off-domain / out-of-scope",
     re.compile(r"\b(bomb|weapon|explosive|malware|ransomware|how to hack)\b", re.I)),
]


def check_rails(text: str) -> dict:
    """Run the (simulated) NeMo rails over a piece of text.

    Returns {"verdict": "ALLOW"|"BLOCK", "rail": kind|None, "reason": str}.
    """
    for kind, name, pat in _RAILS:
        if pat.search(text or ""):
            return {"verdict": "BLOCK", "rail": kind, "reason": name}
    return {"verdict": "ALLOW", "rail": None, "reason": "passed all rails"}


# ── OpenShell secure runtime — network egress allowlist ────────────────────────
# NemoClaw signs a policy; the gateway denies any egress not on this allowlist.
EGRESS_ALLOWLIST = [
    "build.nvidia.com",       # pull NIM catalog / signed containers
    "nvcr.io",                # NGC registry
    "dgx-spark.internal",     # the sovereign inference box
    "vault.internal",         # on-prem secret store
]


def egress_allowed(host: str) -> bool:
    h = (host or "").strip().lower()
    return any(h == a or h.endswith("." + a) for a in EGRESS_ALLOWLIST)


# ── Privacy router — keep sovereign data sovereign ─────────────────────────────
_PII = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b|"                       # US SSN
    r"\b(?:\d[ -]?){13,16}\b|"                      # card-ish number
    r"[\w.+-]+@[\w-]+\.[\w.-]+|"                    # email
    r"\b(patient|salary|medical|passport|diagnosis|home address)\b",
    re.I)


def classify_privacy(text: str) -> dict:
    """Sensitive (PII/secrets) → route LOCAL; else may use a larger model.

    Returns {"sensitive": bool, "route": "LOCAL"|"CLOUD-OK", "why": str}.
    """
    if _PII.search(text or "") or check_rails(text)["rail"] == "output":
        return {"sensitive": True, "route": "LOCAL",
                "why": "contains PII/secrets — never leaves the perimeter"}
    return {"sensitive": False, "route": "CLOUD-OK",
            "why": "no sensitive data — may use a larger model"}

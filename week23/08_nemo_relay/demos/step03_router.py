#!/usr/bin/env python3
"""PART 3 · Optimize — the router right-sizes the model  [INTERMEDIATE]

Every model call from Hermes goes through the Relay's Router → Gateway. The router
sends EASY requests (classify, extract, route) to a Nano or Mini and HARD ones
(reasoning, long turns) to the big model — right-sizing per request. This demo shows
a routing-decision trace for several requests, then a cost & latency comparison of
"always the big model" vs "relay-routed" — the same answers for a fraction of cost.

Run:  python demos/step03_router.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import sim  # noqa: E402
import view  # noqa: E402
import relaylab  # noqa: E402

BIG, MINI, NANO = "gpt-5.5", "gpt-4.4-mini", "gpt-5.4-nano"

# (request, difficulty, chosen backend, ~output tokens)  — the router's decisions
REQUESTS = [
    ("classify this email as spam/not-spam",      "easy",   NANO, 20),
    ("extract the ticket id from this message",   "easy",   NANO, 15),
    ("summarize this 400-word thread",            "medium", MINI, 120),
    ("draft a Slack reply to the customer",       "medium", MINI, 90),
    ("debug this failing test + propose a fix",   "hard",   BIG,  600),
]


def _routing_trace() -> None:
    print("  Router → Gateway · decisions for this batch of requests:\n")
    for req, diff, backend, _ in REQUESTS:
        tier = {BIG: "big", MINI: "mini", NANO: "nano"}[backend]
        print(f"  → [{diff:<6}] route to {backend:<13} ({tier})   « {req}")
    print("\n  (easy → nano, medium → mini, hard → big — right-size per request)\n")


def _cost_latency() -> None:
    # cost per 1M output tokens (illustrative, from sim.CATALOG); latency ∝ 1/tok_s
    def cost(model, toks):
        return sim.cost_per_mtok(model) * toks / 1_000_000
    def latency(model, toks):
        return toks / sim.tok_s(model)

    all_big_cost = sum(cost(BIG, t) for *_, t in REQUESTS)
    all_big_lat = sum(latency(BIG, t) for *_, t in REQUESTS)
    routed_cost = sum(cost(b, t) for _, _, b, t in REQUESTS)
    routed_lat = sum(latency(b, t) for _, _, b, t in REQUESTS)

    print("  Cost & latency for the batch (illustrative, labeled — not a live price):\n")
    print(f"    all-big-model (gpt-5.5)     cost ${all_big_cost*1000:6.3f}m   "
          f"latency {all_big_lat:5.1f}s")
    print(f"    relay-routed (right-sized)  cost ${routed_cost*1000:6.3f}m   "
          f"latency {routed_lat:5.1f}s")
    print(f"\n    → {(1-routed_cost/all_big_cost)*100:.0f}% cheaper, "
          f"{(1-routed_lat/all_big_lat)*100:.0f}% faster — same answers, right-sized model.")


def _tiers() -> tuple[str | None, str | None]:
    """Pick a real small ('nano') and big model from what's installed on the DGX."""
    avail = config.list_local_models()
    small_prefs = ["gemma3:4b", "gemma3", "llama3.2:3b", "llama3.2:1b", "qwen3:0.6b", "gemma"]
    big_prefs = ["llama3.3:70b", "qwen3:32b", "qwen3.6:35b", "nemotron-3-super", "llama3.3"]

    def pick(prefs):
        for p in prefs:
            for m in avail:
                if m.lower().startswith(p) or p in m.lower():
                    return m
        return None

    small = pick(small_prefs) or (avail[0] if avail else None)
    big = pick(big_prefs) or config.MODEL
    return small, big


def _complete(prompt: str, model: str, max_tokens: int = 48) -> str:
    r = view._client().chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=0.2, stream=False)
    m = r.choices[0].message
    return (getattr(m, "content", None) or (getattr(m, "model_extra", None) or {}).get("reasoning") or "").strip()


def _real() -> None:
    small, big = _tiers()
    print(f"{relaylab.status_line()}")
    print(f"Right-sizing on THIS DGX:  small='{small}'   big='{big}'\n")
    print("Measuring the SAME easy request on both models (real latency, recorded as spans):\n")

    easy = "Classify this support message as BILLING or TECH (one word): 'my password reset link expired'."
    hard = "A test fails intermittently only under load. In two sentences, outline how to debug it."

    a_small, ms_small = relaylab.record_llm("route:easy→small", small, easy,
                                            lambda p, m: _complete(p, m, 16))
    print(f"  easy → {small:<16} {ms_small:7.0f} ms   « {a_small[:40]}")
    a_big, ms_big = relaylab.record_llm("route:easy→big", big, easy,
                                        lambda p, m: _complete(p, m, 16))
    print(f"  easy → {big:<16} {ms_big:7.0f} ms   « {a_big[:40]}")

    a_hard, ms_hard = relaylab.record_llm("route:hard→big", big, hard,
                                          lambda p, m: _complete(p, m, 96))
    print(f"  hard → {big:<16} {ms_hard:7.0f} ms   « {a_hard[:40]}")

    if ms_big > 0:
        save = (1 - ms_small / ms_big) * 100
        print(f"\n  Same easy request: routing to '{small}' is {save:.0f}% faster than '{big}'")
        print(f"  ({ms_small:.0f} ms vs {ms_big:.0f} ms). Cost both = $0.0000 (sovereign · on your DGX).")
    print(f"  Hard request genuinely needs the big model ({ms_hard:.0f} ms). That is right-sizing.")
    print(f"\n  These 3 calls are now spans in Phoenix → {relaylab.PHOENIX_URL} (project '{relaylab.PROJECT}').")


def _sim() -> None:
    print("Model calls flow Hermes → Router → Gateway → a right-sized backend:")
    print("  • gpt-5.4-nano  — easy: classify / extract / route")
    print("  • gpt-4.4-mini  — medium: summarize / draft / moderate tool-use")
    print("  • gpt-5.5       — hard: reasoning / debugging / long agent turns\n")

    _routing_trace()
    _cost_latency()

    print(f"\nA one-line rationale for right-sizing ({config.MODEL}):\n")
    view.generate("In two sentences, why does routing easy requests to a small model and "
                  "only hard ones to a big model cut cost and latency without hurting quality?",
                  max_tokens=200, title="why right-size the model")
    print(f"\n(Tip: {relaylab.status_line()})")


def main() -> None:
    view.banner("PART 3", "Optimize — the router right-sizes the model", "INTERMEDIATE")
    view.mode_line()

    if relaylab.ready():
        _real()
    else:
        _sim()
    print("\nTakeaway: most agent calls are easy. Routing them to a small model and reserving")
    print("the big model for hard turns is a top cost/latency lever. Next: export & close the loop.")


if __name__ == "__main__":
    main()

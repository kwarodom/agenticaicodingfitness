#!/usr/bin/env python3
"""Lab 11-3 · Same task, every model you have a key for.

Jev routes each message ONCE; then every available provider writes the same reply
(in parallel threads, so the lab stays fast). You compare latency, tokens, length
and whether a Thai guest got a Thai answer. Single observations — NOT a benchmark,
and no prices: they differ per provider and change often (see each pricing page).

Run: .venv/bin/python week24/11_jev_plus_llm/labs/lab03_compare_models.py
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline as P  # noqa: E402
from jevkit import banner, mode, step, table  # noqa: E402
import llmkit  # noqa: E402

MESSAGES = [P.GUESTS[0], P.GUESTS[1]]            # one English, one Thai

banner("Lab 11-3 · compare the models you have", "same routed task · one draft per provider")
dry = mode() == "dry"
candidates = [p["id"] for p in llmkit.providers() if dry or p["has_key"] or p["key_optional"]]
missing = [p["id"] for p in llmkit.providers() if p["id"] not in candidates]
print(f"◆ providers: {', '.join(candidates)}" + (f"   ◈ skipped (no key): {', '.join(missing)}" if missing else ""))


def complete(text: str) -> bool:
    """Cheap exact check in code: does the draft end like a finished message?"""
    t = text.strip()
    return bool(t) and not t.startswith("[") and (t[-1] in ".!?)" or t.endswith(("Services", "ค่ะ", "คะ", "ครับ")))


def one(provider, msg, routed):
    try:
        return provider, P.write_reply(msg, routed, provider, None, max_tokens=1500)  # roomy: thinking models spend hidden tokens first
    except llmkit.LLMError as e:
        return provider, {"error": str(e)[:90]}


for n, msg in enumerate(MESSAGES, 1):
    step(n, f"“{msg}”")
    routed = P.route(msg)
    jl = routed["resp"].get("_latency_ms")
    print(f"» Jev → team {routed['team']} · language {routed['language']} (code) · "
          + (f"{jl:.0f} ms" if jl else "replay"))
    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        results = list(pool.map(lambda p: one(p, msg, routed), candidates))
    rows, drafts = [], []
    for prov, r in results:
        if "error" in r:
            rows.append([prov, "—", "—", "—", "—", "✕ " + r["error"][:40]])
            continue
        if r["_source"] == "placeholder":
            rows.append([prov, "—", "—", "—", "—", "◈ no recording (DRY)"])
            continue
        text = r["text"]
        ok_lang = (P.language(text) == routed["language"]) if routed["language"] == "th" else not P.THAI.search(text)
        lat = f"{r['latency_ms']:.0f}" if r.get("latency_ms") else "replay"
        rows.append([prov, r["model"][:24], lat, r["usage"]["output_tokens"], len(text),
                     ("✓ " if ok_lang else "✕ ") + ("Thai" if P.THAI.search(text) else "English"),
                     "✓" if complete(text) else "⚠ cut off?"])
        drafts.append((prov, text))
    table(rows, ["provider", "model (API says)", "ms", "out tok", "chars", "reply language", "complete?"])
    for prov, text in drafts[:3]:
        print(f"   {prov}: {text[:150].replace(chr(10), ' ')}{'…' if len(text) > 150 else ''}")

print("\n═ Pick by YOUR criteria: Thai quality, latency, price, data-residency — measured on your own messages.")
print("  Output tokens include hidden 'thinking' for reasoning models, which is why some counts are high.")
print("  ⚠ cut off? = the draft stopped mid-sentence: a thinking model spent the budget. Raise max_tokens.")

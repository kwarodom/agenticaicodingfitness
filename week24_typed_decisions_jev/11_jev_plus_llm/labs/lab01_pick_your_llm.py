#!/usr/bin/env python3
"""Lab 11-1 · Pick your LLM — which providers have a key, and one hello from your choice.

Keys are looked up env → week24/.env.local (🔑 Keys dialog) → repo-root .env.
This lab prints WHERE a key was found, never the key itself.

Run:            .venv/bin/python week24/11_jev_plus_llm/labs/lab01_pick_your_llm.py
Pick provider:  JEV_LLM_PROVIDER=gemini .venv/bin/python …   (the runner's LLM picker sets this)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from jevkit import banner, step, table  # noqa: E402
import llmkit  # noqa: E402

banner("Lab 11-1 · pick your LLM", "Jev decides · the LLM you choose writes")

step(1, "which providers can this machine use? (key SOURCE only — values are never shown)")
rows = []
for p in llmkit.providers():
    ready = p["has_key"] or p["key_optional"]
    rows.append(["✓" if ready else "·", p["id"], p["label"], p["default"],
                 p["key_source"] or f"missing — set {p['key_names'][0]}"])
table(rows, ["", "id", "provider", "default model", "key"])
ready = [r[1] for r in rows if r[0] == "✓"]
print(f"◆ {len(ready)} of {len(rows)} providers ready: {', '.join(ready)}")

step(2, "your choice (the runner's LLM picker, or JEV_LLM_PROVIDER / JEV_LLM_MODEL)")
prov, model = llmkit.chosen()
print(f"→ provider {prov} · model {model}")

step(3, "one hello — then compare what the model SAYS with what the API REPORTS")
try:
    r = llmkit.generate(prov, model=model, user="Reply with exactly: hello from <your model name>", max_tokens=200)
    llmkit.show(r)
    said = r["text"].replace("hello from", "").strip(" .!")
    print(f"═ the model says it is “{said}”; the API says the answer came from {r['model']}.")
    print("  Always log the API's model field — a model's self-report is often wrong.")
except llmkit.LLMError as e:
    print(f"✕ {prov}: {e}")
    print("  Fix: add the key in the runner's 🔑 Keys dialog (or .env), or pick another provider.")
    sys.exit(1)

#!/usr/bin/env python3
"""Exercise 17 · reference solution — point OpenClaw and Hermes at the same local model, and catch the unsafe configs.

Fill in the three TODOs, save, then run:
    .venv/bin/python week25/17_openclaw_hermes/exercises/ex17_agent_config.py

The checker is free and offline: it compares your functions with the shapes the OpenClaw
and Hermes playbooks document, then throws five broken configs at your reviewer.
Stuck? Compare with exercises/solutions/.
"""
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from sparkkit import banner, check  # noqa: E402

MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
BASE = "http://localhost:8000/v1"


# ── TODO 1 ── the OpenClaw provider entry (models.providers.vllm in ~/.openclaw/openclaw.json):
#   baseUrl, apiKey "vllm" (a placeholder: vLLM needs no key), api "openai-responses", and one model with
#   id = name = model_id, "reasoning": True, "input": ["text"], all four "cost" fields 0,
#   "contextWindow": context_window, "maxTokens": max_tokens.
def openclaw_provider(base_url: str, model_id: str, context_window: int, max_tokens: int = 8192) -> dict:
    return {"baseUrl": base_url, "apiKey": "vllm", "api": "openai-responses",
            "models": [{"id": model_id, "name": model_id, "reasoning": True, "input": ["text"],
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                        "contextWindow": context_window, "maxTokens": max_tokens}]}


# ── TODO 2 ── the Hermes playbook's non-interactive setup, as a list of four command strings:
#   hermes config set model.provider custom
#   hermes config set model.base_url <base_url>
#   hermes config set model.default <model_id>
#   hermes -z "Reply exactly HERMES_OK"
def hermes_commands(base_url: str, model_id: str) -> list[str]:
    return ["hermes config set model.provider custom",
            f"hermes config set model.base_url {base_url}",
            f"hermes config set model.default {model_id}",
            'hermes -z "Reply exactly HERMES_OK"']


# ── TODO 3 ── review a provider entry; return a list of problem strings (empty = fine). Flag:
#   • baseUrl not ending in /v1                                  → mention "/v1"
#   • baseUrl host not localhost / 127.0.0.1 (model exposed)     → mention "local"
#   • apiKey that looks real: starts with sk-, nvapi- or hf_      → mention "key"
#   • a model id that is not in `served`                          → mention "served"
#   • contextWindow below 32768 (the OpenClaw playbook's minimum) → mention "context"
def review(provider: dict, served: list[str]) -> list[str]:
    problems = []
    base = provider.get("baseUrl", "")
    if not base.rstrip("/").endswith("/v1"):
        problems.append(f"baseUrl {base!r} must end in /v1")
    if urlparse(base).hostname not in ("localhost", "127.0.0.1"):
        problems.append("baseUrl is not local: the model endpoint would be reachable off the Spark")
    if str(provider.get("apiKey", "")).startswith(("sk-", "nvapi-", "hf_")):
        problems.append("apiKey looks like a real key: use a placeholder such as 'vllm'")
    for m in provider.get("models", []):
        if m.get("id") not in served:
            problems.append(f"model {m.get('id')!r} is not served by the endpoint")
        if m.get("contextWindow", 0) < 32_768:
            problems.append(f"contextWindow {m.get('contextWindow')} is below the 32K agent minimum")
    return problems


# ─────────────────────────── checker — no need to edit below ────────────────
def has(problems, word: str) -> bool:
    return isinstance(problems, list) and any(word in str(p).lower() for p in problems)


def main() -> None:
    banner("Exercise 17 · one model, two agents", "offline checker · free · no Spark needed", status=False)
    ok = True
    p = openclaw_provider(BASE, MODEL, 262_144)
    m = ((p or {}).get("models") or [{}])[0] if isinstance(p, dict) else {}
    shape = (isinstance(p, dict) and p.get("baseUrl") == BASE and p.get("apiKey") == "vllm"
             and p.get("api") == "openai-responses" and m.get("id") == m.get("name") == MODEL
             and m.get("contextWindow") == 262_144 and m.get("maxTokens") == 8192
             and m.get("cost") == {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0})
    ok &= check(shape, "openclaw_provider: baseUrl · placeholder key · openai-responses · id = name · 262144 context",
                f"TODO 1: openclaw_provider(...) is not the playbook's shape (got {json.dumps(p)[:120]})")

    want = ["hermes config set model.provider custom", f"hermes config set model.base_url {BASE}",
            f"hermes config set model.default {MODEL}", 'hermes -z "Reply exactly HERMES_OK"']
    got = hermes_commands(BASE, MODEL)
    ok &= check(got == want, "hermes_commands: provider custom · base_url · default · hermes -z check",
                f"TODO 2: expected {want[0]!r} … (got {got!r})"[:200])

    served = [MODEL]
    cases = {"good": (p, None)}
    if shape:
        cases.update({
            "missing /v1": ({**p, "baseUrl": "http://localhost:8000"}, "/v1"),
            "LAN host": ({**p, "baseUrl": "http://192.168.1.42:8000/v1"}, "local"),
            "real key": ({**p, "apiKey": "sk-FAKE-abc123def456"}, "key"),
            "unserved model": ({**p, "models": [{**m, "id": "llama3.1:8b", "name": "llama3.1:8b"}]}, "served"),
            "8K context": ({**p, "models": [{**m, "contextWindow": 8192}]}, "context"),
        })
    try:
        verdicts = {k: (review(c, served) == [] if word is None else has(review(c, served), word))
                    for k, (c, word) in cases.items()}
    except Exception as e:  # noqa: BLE001 — a half-written review() should fail the check, not crash it
        verdicts = {"review() raised " + type(e).__name__: False}
    ok &= check(shape and all(verdicts.values()) and len(verdicts) == 6,
                "review: good → [] · catches /v1 · local · key · served · context",
                f"TODO 3: {verdicts}")
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)

    print("\n▣ ~/.openclaw/openclaw.json → models.providers.vllm")
    for line in json.dumps(p, indent=2).splitlines()[:8]:
        print("│ " + line)
    print("│ …")
    print("\n▣ Hermes, on the Spark")
    for c in got:
        print("$ " + c)
    host = urlparse(BASE).hostname
    print(f"\n═ Both agents now call {MODEL} at {host}:8000. What changes if you swap the model: "
          "one JSON edit + gateway restart for OpenClaw, `hermes model` for Hermes.")


if __name__ == "__main__":
    main()

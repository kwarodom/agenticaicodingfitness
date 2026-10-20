#!/usr/bin/env python3
"""Lab 18-1 · Preflight: can this Ollama endpoint drive a coding agent?

A coding agent needs more from a model server than "it answers". This lab checks
the five things that decide whether Claude Code, OpenCode, Codex or Continue will
work against your Spark's Ollama:

  1. the Ollama build is new enough for `ollama launch` (on the Spark, over ssh)
  2. the coding model is pulled and reports the `tools` capability
  3. the OpenAI-style APIs answer  (/v1/chat/completions: OpenCode · /v1/responses: Codex)
  4. the Anthropic-style API answers (/v1/messages: what Claude Code speaks)
  5. a real tool call comes back, and the context window is big enough

Read-only: it sends GET requests and four tiny chats (≤ 300 tokens). When the
Spark's Ollama is not reachable it checks THIS laptop's Ollama instead, labelled
LAPTOP STAND-IN, so you can see every check work before you have a Spark.

Run: .venv/bin/python week25/18_coding_agents/labs/lab01_endpoint_preflight.py [--model qwen3.6:35b-a3b-mtp-q4_K_M]
"""
import argparse
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import (banner, chat, http_json, note, ok, pick_laptop_model, resolve, result, sh,  # noqa: E402
                      step, table, warn)

PLAYBOOK_MODEL = "qwen3.6:35b-a3b-mtp-q4_K_M"   # cli-coding-agent playbook, DGX Spark row
RECOMMENDED_CTX = 64_000                        # local-coding-agent playbook, Step 6
WRITE_FILE = [{"type": "function", "function": {
    "name": "write_file", "description": "Write a text file in the workspace.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                   "required": ["path", "content"]}}}]


def anthropic_messages(native: str, model: str, prompt: str, max_tokens: int = 24, extra: dict | None = None) -> dict:
    """POST {native}/v1/messages — the Anthropic Messages API that Claude Code sends."""
    body = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}],
            **(extra or {})}
    req = Request(native + "/v1/messages", data=json.dumps(body).encode(), method="POST",
                  headers={"Content-Type": "application/json", "x-api-key": "ollama",
                           "anthropic-version": "2023-06-01"})
    with urlopen(req, timeout=180) as r:                    # noqa: S310 — your own Ollama
        return json.loads(r.read().decode("utf-8", errors="replace"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=PLAYBOOK_MODEL, help="coding model to check on the Spark")
    args = ap.parse_args()

    banner("Lab 18-1 · preflight — can this endpoint drive a coding agent?",
           "Ollama version · model capabilities · OpenAI + Anthropic APIs · a real tool call · context")

    step(1, "on the Spark: Ollama build, `ollama launch`, and the pulled coding model")
    sh("ollama --version", example="ollama version is 0.x.y")
    sh("ollama launch --help >/dev/null 2>&1 && echo 'ollama launch: available' || echo 'ollama launch: missing'",
       example="ollama launch: available")
    sh("ollama list", example=f"NAME                          ID              SIZE     MODIFIED\n"
                              f"{PLAYBOOK_MODEL}    <id>            23 GB    2 minutes ago")
    note("The playbooks need a current Ollama: `ollama launch` and the MTP Q4_K_M tag are recent additions. "
         "An older build answers 'unknown command' or HTTP 412.")

    step(2, "which endpoint are we testing?")
    base, src = resolve("ollama")
    if src == "dry":
        warn("no Spark Ollama and no laptop Ollama answered — start one, or run this on the Spark.")
        result("DRY: the HTTP checks need a live Ollama. Steps 3–5 were skipped.")
        return
    model = args.model if src == "spark" else pick_laptop_model(["nemotron-3-nano", "gemma4:12b"])
    native = base[:-3] if base.endswith("/v1") else base
    where = "Spark Ollama" if src == "spark" else "Ollama on THIS laptop — LAPTOP STAND-IN, not the Spark"
    print(f"→ {native} · model={model} · {where}")

    rows = []

    step(3, "native API: version, model details, capabilities, context")
    try:
        ver = http_json("GET", native + "/api/version", timeout=5).get("version", "?")
        rows.append(["Ollama native API  /api/version", "✓", ver])
    except Exception as e:  # noqa: BLE001
        rows.append(["Ollama native API  /api/version", "✕", str(e)[:44]])
    caps, max_ctx = [], 0
    try:
        show = http_json("POST", native + "/api/show", {"model": model}, timeout=15)
        caps = show.get("capabilities") or []
        max_ctx = next((int(v) for k, v in (show.get("model_info") or {}).items() if k.endswith(".context_length")), 0)
        d = show.get("details") or {}
        print(f"│ {model}: {d.get('parameter_size', '?')} params · {d.get('quantization_level', '?')} · "
              f"capabilities={caps} · model max context={max_ctx:,}")
        rows.append(["model reports `tools` capability", "✓" if "tools" in caps else "✕", ", ".join(caps)])
    except Exception as e:  # noqa: BLE001
        rows.append(["model reports `tools` capability", "✕", f"/api/show failed: {str(e)[:30]}"])

    step(4, "the three wire formats: OpenAI chat (OpenCode), Anthropic messages (Claude Code), Responses (Codex)")
    try:
        r = chat(base, model, [{"role": "user", "content": "Reply with the single word READY."}],
                 max_tokens=24, extra={"reasoning_effort": "none"})
        print(f"· /v1/chat/completions → {r['text'][:60]!r} in {r['total_ms'] / 1000:.1f}s")
        rows.append(["OpenAI API  /v1/chat/completions", "✓" if r["text"] else "⚠ empty", r["text"][:40]])
    except Exception as e:  # noqa: BLE001
        rows.append(["OpenAI API  /v1/chat/completions", "✕", str(e)[:44]])
    try:
        t0 = time.perf_counter()
        m = anthropic_messages(native, model, "Reply with the single word READY.")
        kinds = [b.get("type") for b in m.get("content", [])]
        text = " ".join(b.get("text", "") for b in m.get("content", []) if b.get("type") == "text").strip()
        print(f"· /v1/messages         → blocks={kinds} text={text[:40]!r} in {time.perf_counter() - t0:.1f}s "
              f"(stop_reason={m.get('stop_reason')})")
        detail = text[:40]
        if not text and "thinking" in kinds:
            print("  ~ a thinking model spent the whole 24-token budget on a `thinking` block. Retry with thinking off:")
            m2 = anthropic_messages(native, model, "Reply with the single word READY.",
                                    extra={"thinking": {"type": "disabled"}})
            text = " ".join(b.get("text", "") for b in m2.get("content", []) if b.get("type") == "text").strip()
            print(f"· /v1/messages + thinking:disabled → {text[:40]!r} (stop_reason={m2.get('stop_reason')})")
            detail = f"{text[:12] or '—'} (only with thinking disabled)"
        rows.append(["Anthropic API  /v1/messages", "✓" if m.get("type") == "message" else "✕", detail or "—"])
    except Exception as e:  # noqa: BLE001
        rows.append(["Anthropic API  /v1/messages", "✕", str(e)[:44]])
    try:
        rr = http_json("POST", base + "/responses", {"model": model, "input": "Reply with the single word READY.",
                                                     "max_output_tokens": 40, "reasoning": {"effort": "none"}},
                       timeout=180)
        text = " ".join(c.get("text", "") for o in rr.get("output", []) for c in (o.get("content") or [])
                        if c.get("type") == "output_text").strip()
        print(f"· /v1/responses        → {text[:40]!r} (status={rr.get('status')})")
        rows.append(["OpenAI Responses API  /v1/responses", "✓" if rr.get("object") == "response" else "✕",
                     text[:40] or "—"])
    except Exception as e:  # noqa: BLE001
        rows.append(["OpenAI Responses API  /v1/responses", "✕", str(e)[:44]])

    step(5, "a real tool call, and the context the server actually allocated")
    try:
        r = chat(base, model, [{"role": "user", "content": "Create hello.py that prints hi. Use the write_file tool."}],
                 tools=WRITE_FILE, max_tokens=300, extra={"reasoning_effort": "none"})
        tc = (r["tool_calls"] or [{}])[0].get("function") or {}
        try:
            args_ok = isinstance(json.loads(tc.get("arguments") or "null"), dict)
        except (TypeError, json.JSONDecodeError):
            args_ok = isinstance(tc.get("arguments"), dict)
        good = tc.get("name") == "write_file" and args_ok
        print(f"→ tool_call {tc.get('name')}({str(tc.get('arguments'))[:90]})" if tc else
              f"· no tool call — the model answered in prose: {r['text'][:80]!r}")
        rows.append(["tool call with valid JSON args", "✓" if good else "✕",
                     f"{tc.get('name') or 'prose only'} · {r['total_ms'] / 1000:.1f}s"])
    except Exception as e:  # noqa: BLE001
        rows.append(["tool call with valid JSON args", "✕", str(e)[:44]])
    try:
        loaded = http_json("GET", native + "/api/ps", timeout=5).get("models") or []
        alloc = next((int(x.get("context_length") or 0) for x in loaded if x.get("name") == model), 0)
    except Exception:  # noqa: BLE001
        alloc = 0
    if alloc:
        rows.append([f"allocated context ≥ {RECOMMENDED_CTX:,}", "✓" if alloc >= RECOMMENDED_CTX else "⚠",
                     f"{alloc:,} tokens (model max {max_ctx:,})"])
    else:
        rows.append([f"allocated context ≥ {RECOMMENDED_CTX:,}", "?", "model not loaded — see /api/ps"])

    print()
    table(rows, ["check", "ok", "detail"])
    label = "LIVE on the Spark" if src == "spark" else "LAPTOP STAND-IN (these are not Spark numbers)"
    note(f"{label}. Speeds above are this endpoint's, for a ≤ 300-token reply.")
    if all(r[1] == "✓" for r in rows):
        ok("every check passed: this endpoint can drive a CLI coding agent.")
    fails = [r[0] for r in rows if r[1] != "✓"]
    result("ready for `ollama launch claude --model …`" if not fails else
           f"{len(fails)} check(s) to fix before you launch an agent: " + "; ".join(fails))


if __name__ == "__main__":
    main()

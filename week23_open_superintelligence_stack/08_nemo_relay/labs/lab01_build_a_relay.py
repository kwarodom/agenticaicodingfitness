#!/usr/bin/env python3
"""Lab 01 — build a minimal relay: every call in one agent turn becomes a span.

The demos SHOW spans; this lab MAKES them. You run a tiny agent turn for real —
a tool step (read this repo), one real LLM call through config's endpoint,
another tool step — and a ~40-line relay records each as a span with REAL
timing + token counts, prints the Phoenix-style tree, and writes the trace as
OTel-shaped JSON that lab03 exports to a live Phoenix.

Run:  cd <repo root> && .venv/bin/python week23/08_nemo_relay/labs/lab01_build_a_relay.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

PRICE_PER_MTOK = 0.40   # illustrative $/M output tokens — a label, NOT a live price
TRACE_ID = os.urandom(16).hex()
SPANS: list[dict] = []


def span_open(name: str, kind: str, parent: str | None = None) -> dict:
    return {"name": name, "kind": kind, "parent": parent,
            "span_id": os.urandom(8).hex(), "start": time.time(), "attrs": {}}


def span_close(s: dict) -> dict:
    s["end"] = time.time()
    SPANS.append(s)
    print(f"  ✓ span [{s['kind']:<5}] {s['name']:<26} {(s['end'] - s['start']) * 1000:8.1f} ms")
    return s


def tool_step(name: str, parent: str, fn) -> object:
    s = span_open(f"tool · {name}", "tool", parent)
    out = fn(s)
    s["attrs"]["openinference.span.kind"] = "TOOL"
    span_close(s)
    return out


def llm_step(parent: str) -> str:
    s = span_open("llm · summarize", "llm", parent)
    from openai import OpenAI, BadRequestError
    # no retries + 50 s budget: a COLD model load alone can take ~35 s on Ollama
    client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY,
                    timeout=50.0, max_retries=0)
    kw = dict(model=config.MODEL, temperature=0.2, max_tokens=180,
              messages=[{"role": "user", "content":
                         "In one sentence: why must an agent runtime record every "
                         "tool and LLM call as a telemetry span?"}])
    try:
        try:    # thinking models (gemma4, qwen3.6…) burn a terse budget on reasoning —
            resp = client.chat.completions.create(   # ask them not to; endpoints that
                **kw, extra_body={"reasoning_effort": "none"})   # reject the knob get
        except BadRequestError:                                  # a plain retry.
            resp = client.chat.completions.create(**kw)
    except Exception as e:                       # timeout/refused — teach, don't traceback
        s["attrs"] = {"error.type": type(e).__name__, "openinference.span.kind": "LLM"}
        span_close(s)
        print(f"    ✗ llm call failed ({type(e).__name__}) — a cold model load can take"
              " ~40 s; re-run (the endpoint keeps it warm). The span still recorded.")
        return ""
    u, msg = resp.usage, resp.choices[0].message
    s["attrs"] = {"gen_ai.request.model": config.MODEL,
                  "gen_ai.usage.input_tokens": getattr(u, "prompt_tokens", 0) or 0,
                  "gen_ai.usage.output_tokens": getattr(u, "completion_tokens", 0) or 0,
                  "openinference.span.kind": "LLM"}
    span_close(s)
    # thinking models may put text in an extra `reasoning` field — don't print blank
    return msg.content or (msg.model_extra or {}).get("reasoning") or ""


def print_tree() -> None:
    print("\n◈ Phoenix-style trace tree — every number below was MEASURED, not scripted:\n")
    print(f"  trace {TRACE_ID[:12]}…")
    print("  ┌────────────────────────────────┬───────┬──────────┬───────────┐")
    print("  │ span                           │ kind  │ latency  │ cost*     │")
    print("  ├────────────────────────────────┼───────┼──────────┼───────────┤")
    for s in SPANS:
        depth = 0 if s["parent"] is None else 1
        lat = (s["end"] - s["start"]) * 1000
        cost = s["attrs"].get("gen_ai.usage.output_tokens", 0) * PRICE_PER_MTOK / 1e6
        label = ("  " * depth + s["name"])[:30].ljust(30)
        print(f"  │ {label} │ {s['kind']:<5} │ {lat:5.0f} ms │ ${cost:0.6f} │")
    print("  └────────────────────────────────┴───────┴──────────┴───────────┘")
    print(f"  * priced at an illustrative ${PRICE_PER_MTOK}/M output tokens — local tokens are $0.")


def write_otel() -> Path:
    def attrs(d):
        return [{"key": k, "value": ({"intValue": str(v)} if isinstance(v, int)
                                     else {"stringValue": str(v)})} for k, v in d.items()]
    spans = [{"traceId": TRACE_ID, "spanId": s["span_id"],
              **({"parentSpanId": s["parent"]} if s["parent"] else {}),
              "name": s["name"], "kind": 1,
              "startTimeUnixNano": str(int(s["start"] * 1e9)),
              "endTimeUnixNano": str(int(s["end"] * 1e9)),
              "attributes": attrs(s["attrs"])} for s in SPANS]
    doc = {"resourceSpans": [{"resource": {"attributes": attrs({"service.name": "lab-relay"})},
                              "scopeSpans": [{"scope": {"name": "lab01_build_a_relay"},
                                              "spans": spans}]}]}
    path = config.ensure_sandbox() / "trace_lab01.json"
    path.write_text(json.dumps(doc, indent=1))
    return path


def main() -> None:
    print("━" * 64)
    print("  ▣ Lab 01 — build a minimal relay (spans from a REAL agent turn)")
    print("━" * 64)
    print(f"\n  endpoint: {config.safe_base_url()}   model: {config.MODEL}   mode: {config.MODE}\n")

    if config.MODE != "real":
        print("◈ [no endpoint — showing expected output] Nothing below was executed.")
        print("  Get a real endpoint first (any ONE of these), then re-run this lab:")
        print("    ollama run qwen3.6:35b-a3b-q8_0                     # C · this laptop")
        print("    export DGX_BASE_URL=http://<spark>:11434/v1          # A · your Spark")
        print("    export DGX_CONN=cloud DGX_CLOUD_URL=https://integrate.api.nvidia.com/v1 \\")
        print("           DGX_API_KEY=nvapi-…                           # B · build.nvidia.com")
        print("\n  Expected (sample — real runs vary):")
        print("    ✓ span [agent] hermes-turn                    2412.0 ms")
        print("    ✓ span [tool ] tool · read_demos                 3.1 ms")
        print("    ✓ span [llm  ] llm · summarize                2280.4 ms")
        print("    …then the Phoenix-style tree and .sandbox/trace_lab01.json")
        return

    root = span_open("hermes-turn", "agent")
    root["attrs"]["openinference.span.kind"] = "AGENT"

    print("▣ step 1 — tool span: read this folder's demos (real file I/O, real timing)")
    demos = Path(__file__).resolve().parents[1] / "demos"

    def read_demos(s):
        files = sorted(demos.glob("*.py"))
        s["attrs"].update({"tool.name": "read_demos", "files": len(files),
                           "lines": sum(len(p.read_text().splitlines()) for p in files)})
        return s["attrs"]["lines"]
    lines = tool_step("read_demos", root["span_id"], read_demos)

    print("\n▣ step 2 — llm span: one REAL call, usage tokens captured from the response")
    answer = llm_step(root["span_id"])
    print(f"    · model said: {answer.strip()[:140]}")

    print("\n▣ step 3 — tool span: write the turn's artifact (again: real work, real ms)")
    tool_step("write_notes", root["span_id"],
              lambda s: (config.ensure_sandbox() / "lab01_notes.txt")
              .write_text(f"demos={lines} lines; answer={answer[:200]}"))

    span_close(root)
    print_tree()
    path = write_otel()
    print(f"\n✓ OTel-shaped trace written → {path}")
    print("  lab03 POSTs this exact file to a live Phoenix. Next: labs/lab02 (the router).")


if __name__ == "__main__":
    main()

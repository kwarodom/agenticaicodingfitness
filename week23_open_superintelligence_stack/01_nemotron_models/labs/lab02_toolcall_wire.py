#!/usr/bin/env python3
"""LAB 02 · Tool-calling on the wire — see the raw protocol, not the magic.

demos/step04_tool_calling.py runs a polished tool loop. This lab shows what is
ACTUALLY on the wire: the tool schema you send, the assistant message with
`tool_calls` + `finish_reason="tool_calls"` that comes back, the `role:"tool"`
result you append, and the final answer. Once you can read these four JSON
shapes, every agent framework stops being magic.

Run:  cd <repo root> && .venv/bin/python week23/01_nemotron_models/labs/lab02_toolcall_wire.py
No endpoint? Prints the real commands + a clearly-labeled sample transcript.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

TOOLS = [{"type": "function", "function": {
    "name": "get_chiller_status",
    "description": "Read the live status of one chiller by id (1-3).",
    "parameters": {"type": "object",
                   "properties": {"chiller_id": {"type": "string"}},
                   "required": ["chiller_id"]}}}]

STATUS = {"1": {"cop": 5.1, "state": "running"},
          "2": {"cop": 4.2, "state": "running"},
          "3": {"cop": 3.8, "state": "fault", "fault": "low refrigerant pressure"}}


def _impl(args: dict) -> str:
    cid = str(args.get("chiller_id", "")).strip().lstrip("#")
    return json.dumps(STATUS.get(cid, {"error": f"no chiller {cid!r}"}))


def _show(tag: str, obj) -> None:
    print(f"◈ {tag}")
    print("  " + json.dumps(obj, indent=2, default=str).replace("\n", "\n  ") + "\n")


def no_endpoint() -> None:
    print("✗ no endpoint reachable — to make this REAL:")
    print("    ollama pull nemotron-3-nano        # tool-calling capable, fits your laptop/Spark")
    print("    # or:  export DGX_CONN=cloud DGX_CLOUD_URL=https://integrate.api.nvidia.com/v1 \\")
    print("    #             DGX_API_KEY=nvapi-...\n")
    print("[no endpoint — showing expected output]")
    print('  ◈ WIRE ← assistant message (turn 1)')
    print('    {"content": null, "tool_calls": [{"id": "call_ab12", "type": "function",')
    print('      "function": {"name": "get_chiller_status", "arguments": "{\\"chiller_id\\": \\"3\\"}"}}]}')
    print('    finish_reason = "tool_calls"   ← the model is PAUSED, waiting for you')
    print('  ◈ WIRE → role:"tool" result you append')
    print('    {"role": "tool", "tool_call_id": "call_ab12",')
    print('     "content": "{\\"cop\\": 3.8, \\"state\\": \\"fault\\", \\"fault\\": \\"low refrigerant pressure\\"}"}')
    print('  · FINAL: Chiller 3 is faulted (low refrigerant pressure); dispatch service.')


def main() -> None:
    print("━" * 64)
    print("  ▣ LAB 02 · Tool-calling on the wire — four JSON shapes, no magic")
    print("━" * 64 + "\n")
    if config.MODE != "real":
        no_endpoint()
        return
    from openai import OpenAI
    client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY,
                    timeout=45.0, max_retries=0)   # fail fast — no silent 3x retries
    print(f"▣ REAL · {config.MODEL} @ {config.safe_base_url()} ({config.conn_human()})\n")

    _show("WIRE → tools schema in the request (shape 1 of 4)", TOOLS)
    messages = [
        {"role": "system", "content": "Use the tool to check chiller 3, then report its "
         "state in one sentence. Pass chiller_id as a bare digit like '3'."},
        {"role": "user", "content": "Is chiller 3 healthy?"}]

    t0 = time.time()
    for turn in range(1, 4):
        if time.time() - t0 > 75:                 # wall-clock budget: a slow box must not hang the lab
            print("◈ stopping — wall-clock budget spent (slow endpoint); the shapes above are the lesson.")
            break
        r = client.chat.completions.create(model=config.MODEL, messages=messages,
                                           tools=TOOLS, max_tokens=250, temperature=0.2)
        msg, finish = r.choices[0].message, r.choices[0].finish_reason
        if not msg.tool_calls:
            print(f"◈ finish_reason = {finish!r} — the model answered (shape 4 of 4)")
            print(f"  · FINAL: {(msg.content or '').strip()[:300]}")
            break
        _show(f"WIRE ← assistant tool_calls, turn {turn} (shape 2 of 4) · finish_reason={finish!r}",
              [{"id": c.id, "function": {"name": c.function.name,
                                         "arguments": c.function.arguments}} for c in msg.tool_calls])
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name,
                                                      "arguments": c.function.arguments}}
                                        for c in msg.tool_calls]})
        for c in msg.tool_calls:
            result = _impl(json.loads(c.function.arguments or "{}"))
            tool_msg = {"role": "tool", "tool_call_id": c.id, "content": result}
            _show("WIRE → role:\"tool\" result you append (shape 3 of 4)", tool_msg)
            messages.append(tool_msg)
    else:
        print("✗ model kept calling tools past 3 turns — inspect the transcript above.")

    print("\n✓ takeaway — 'the model calls a tool' really means: it EMITS a JSON")
    print("  request and pauses; YOUR code runs the function and appends the result.")
    print("  The model never touches your systems — the harness does. (App 07 guards this.)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"\n✗ request failed ({type(e).__name__}: {e})")
        print("  small local models may lack tool support — try nemotron-3-nano or qwen3.6,")
        print("  and check the endpoint URL has :PORT and /v1.")

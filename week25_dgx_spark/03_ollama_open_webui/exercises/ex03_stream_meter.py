#!/usr/bin/env python3
"""Exercise 03 · Build a stream meter: parse SSE, split reasoning, time it.

Fill in the four TODOs, save, then run:
    .venv/bin/python week25/03_ollama_open_webui/exercises/ex03_stream_meter.py

The checker is free and offline. It replays a REAL stream captured from Ollama's OpenAI
endpoint on the course laptop (LAPTOP STAND-IN: nemotron-3-nano, thinking on, 2026-09-29),
with the arrival time of every line in milliseconds. Stuck? Compare with exercises/solutions/.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, check  # noqa: E402


# ── TODO 1 ── Server-Sent Events: keep only lines that start with "data:", strip that prefix,
#   stop at the "[DONE]" line, and json.loads() each payload. Blank lines are separators: skip them.
def parse_sse(lines: list[str]) -> list[dict]:
    return None


# ── TODO 2 ── join the pieces: every chunk has choices[0].delta with "content" and/or "reasoning"
#   (vLLM and others call it "reasoning_content"). Return (answer, reasoning), both stripped.
#   The last chunk has choices == [] and only "usage" — do not crash on it.
def split_text(chunks: list[dict]) -> tuple[str, str]:
    return None


# ── TODO 3 ── speed. `timed` is [(t_ms, raw_line), ...] exactly as it arrived.
#   ttft_ms   = t_ms of the first line whose delta has non-empty content OR reasoning
#   end_ms    = t_ms of the "data: [DONE]" line
#   tokens    = usage.completion_tokens from the final chunk
#   tok_s     = tokens / ((end_ms - ttft_ms) / 1000)        (the same rule sparkkit.chat() uses)
#   Return {"ttft_ms": ..., "tokens": ..., "tok_s": ...}. Tip: reuse parse_sse on one line at a time.
def speed(timed: list[tuple[float, str]]) -> dict:
    return None


# ── TODO 4 ── build the request that produced a stream like this, but with thinking OFF:
#   OpenAI chat body with model, messages, max_tokens, "stream": True,
#   "stream_options": {"include_usage": True}  (otherwise Ollama sends no usage in a stream), and
#   "reasoning_effort": "none" when think is False (leave the key out when think is True).
def request_body(model: str, messages: list[dict], max_tokens: int = 200, think: bool = False) -> dict:
    return None


# ─────────────────────────── checker — no need to edit below ────────────────
# Captured on the course laptop: POST http://localhost:11434/v1/chat/completions, model nemotron-3-nano:latest,
# "What is 17 * 3? Answer with the number only.", max_tokens 60, stream + include_usage. (t_ms, line)
STREAM = [
    (18649.1, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\",\"reasoning\":\"We\"},\"finish_reason\":null}]}"),
    (18649.2, ""),
    (18649.2, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"reasoning\":\" need\"},\"finish_reason\":null}]}"),
    (18649.2, ""),
    (18651.5, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"reasoning\":\" to\"},\"finish_reason\":null}]}"),
    (18651.5, ""),
    (18664.4, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"reasoning\":\" answer\"},\"finish_reason\":null}]}"),
    (18664.5, ""),
    (18677.2, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"reasoning\":\" with\"},\"finish_reason\":null}]}"),
    (18677.3, ""),
    (18690.8, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"reasoning\":\" the\"},\"finish_reason\":null}]}"),
    (18690.8, ""),
    (18703.7, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"reasoning\":\" number\"},\"finish_reason\":null}]}"),
    (18703.8, ""),
    (18717.0, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"reasoning\":\" only\"},\"finish_reason\":null}]}"),
    (18717.1, ""),
    (18729.2, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"reasoning\":\":\"},\"finish_reason\":null}]}"),
    (18729.2, ""),
    (18755.5, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"reasoning\":\" 1\"},\"finish_reason\":null}]}"),
    (18755.6, ""),
    (18767.4, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"reasoning\":\"7\"},\"finish_reason\":null}]}"),
    (18767.4, ""),
    (18781.0, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"reasoning\":\" *\"},\"finish_reason\":null}]}"),
    (18781.0, ""),
    (18806.0, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"reasoning\":\" 3\"},\"finish_reason\":null}]}"),
    (18806.0, ""),
    (18818.9, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"reasoning\":\" =\"},\"finish_reason\":null}]}"),
    (18819.0, ""),
    (18848.4, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"reasoning\":\" 5\"},\"finish_reason\":null}]}"),
    (18848.4, ""),
    (18858.9, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"reasoning\":\"1\"},\"finish_reason\":null}]}"),
    (18858.9, ""),
    (18872.9, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"reasoning\":\".\"},\"finish_reason\":null}]}"),
    (18872.9, ""),
    (18884.9, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"reasoning\":\" So\"},\"finish_reason\":null}]}"),
    (18884.9, ""),
    (18898.1, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"reasoning\":\" output\"},\"finish_reason\":null}]}"),
    (18898.1, ""),
    (18910.5, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"reasoning\":\" \\\"\"},\"finish_reason\":null}]}"),
    (18910.5, ""),
    (18924.3, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"reasoning\":\"5\"},\"finish_reason\":null}]}"),
    (18924.3, ""),
    (18937.3, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"reasoning\":\"1\"},\"finish_reason\":null}]}"),
    (18937.3, ""),
    (18951.4, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\",\"reasoning\":\"\\\".\"},\"finish_reason\":null}]}"),
    (18951.4, ""),
    (18991.5, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"5\"},\"finish_reason\":null}]}"),
    (18991.5, ""),
    (19005.9, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"1\"},\"finish_reason\":null}]}"),
    (19005.9, ""),
    (19023.8, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}"),
    (19023.8, ""),
    (19023.8, "data: {\"id\":\"chatcmpl-667\",\"object\":\"chat.completion.chunk\",\"created\":1790656974,\"model\":\"nemotron-3-nano:latest\",\"system_fingerprint\":\"fp_ollama\",\"choices\":[],\"usage\":{\"prompt_tokens\":31,\"prompt_tokens_details\":{\"cached_tokens\":0},\"completion_tokens\":31,\"total_tokens\":62}}"),
    (19023.8, ""),
    (19023.8, "data: [DONE]"),
    (19023.8, ""),
]


def close(a, b, tol=0.01) -> bool:
    return isinstance(a, (int, float)) and abs(a - b) <= tol * max(1.0, abs(b))


def main() -> None:
    banner("Exercise 03 · stream meter", "offline checker · replays a real Ollama stream · no Spark needed", status=False)
    ok = True
    lines = [ln for _, ln in STREAM]
    chunks = parse_sse(lines)
    ok &= check(isinstance(chunks, list) and len(chunks) == 27 and "usage" in chunks[-1],
                "parse_sse: 27 JSON chunks, the last one carries usage, [DONE] not included",
                f"TODO 1: expected 27 chunks ending with the usage chunk (got {len(chunks) if isinstance(chunks, list) else chunks!r})")
    parts = split_text(chunks) if isinstance(chunks, list) else None
    ok &= check(isinstance(parts, tuple) and parts[0] == "51" and parts[1].startswith("We need to answer"),
                "split_text: answer '51' · reasoning 'We need to answer with the number only: 17 * 3 = 51 …'",
                f"TODO 2: expected ('51', 'We need to answer…') (got {parts!r})")
    try:
        sp = speed(STREAM)
    except Exception as e:  # noqa: BLE001 — speed() may lean on an unfinished TODO 1/2
        sp = f"{type(e).__name__}: {e}"
    ok &= check(isinstance(sp, dict) and close(sp.get("ttft_ms"), 18649.1) and sp.get("tokens") == 31
                and close(sp.get("tok_s"), 82.7, 0.02),
                "speed: TTFT 18649 ms · 31 tokens · ≈ 82.7 tok/s after the first token",
                f"TODO 3: expected ttft_ms≈18649.1, tokens=31, tok_s≈82.7 (got {sp!r})")
    msgs = [{"role": "user", "content": "hi"}]
    b_off, b_on = request_body("m", msgs, 50), request_body("m", msgs, 50, think=True)
    ok &= check(isinstance(b_off, dict) and b_off.get("stream") is True and b_off.get("reasoning_effort") == "none"
                and b_off.get("stream_options") == {"include_usage": True} and b_off.get("max_tokens") == 50
                and isinstance(b_on, dict) and "reasoning_effort" not in b_on,
                "request_body: stream + include_usage · reasoning_effort 'none' only when think=False",
                f"TODO 4: got {b_off!r}")
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)

    print("\n▣ your meter, applied to the recorded stream (LAPTOP STAND-IN)")
    answer, reasoning = parts
    print(f"│ answer {answer!r} · reasoning {len(reasoning)} chars · {sp['tokens']} tokens billed")
    print(f"│ TTFT {sp['ttft_ms'] / 1000:.1f} s · then {sp['tok_s']:.1f} tok/s")
    print("\n═ 18.6 s to the first token, then 0.4 s for everything else: on this shared laptop the model "
          "was most likely being loaded first (lab 02's native load_duration shows how to confirm it). "
          "TTFT = load + prefill; keep models warm with OLLAMA_KEEP_ALIVE.")


if __name__ == "__main__":
    main()

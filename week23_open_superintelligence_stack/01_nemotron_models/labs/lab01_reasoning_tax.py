#!/usr/bin/env python3
"""LAB 01 · The reasoning tax — measure what thinking actually costs.

Nemotron 3 is a reasoning model (RLM): it can emit a private REASON channel
before the ANSWER. Reasoning tokens are not free — they cost latency (and, on a
cloud endpoint, money). This lab sends an EASY and a HARD prompt to the SAME
model, splits REASON from ANSWER on the stream, and computes the tax.

Run:  cd <repo root> && .venv/bin/python week23/01_nemotron_models/labs/lab01_reasoning_tax.py
Works with local Ollama, a DGX Spark tunnel, or a build.nvidia.com nvapi- key
(inherited from ../config.py). No endpoint? Prints the real commands + a sample.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

EASY = "What is 17 + 26? Reply with only the number."
HARD = ("A data center may run only 3 of its 4 CRAC units at once. Their COPs are "
        "3.1, 3.6, 2.8 and 4.0. Which unit should stay OFF to keep cooling "
        "efficiency highest, and why? Answer in two sentences.")


def _clip(text: str, n: int) -> str:
    """One-line preview: collapse newlines/runs of spaces, then truncate."""
    flat = " ".join(text.split())
    return (flat[: n - 1] + "…") if len(flat) > n else flat


def _split_think(text: str) -> tuple[str, str]:
    """Some servers inline thinking as <think>…</think> instead of a delta field."""
    if "<think>" in text:
        pre, _, rest = text.partition("<think>")
        thought, _, post = rest.partition("</think>")
        return thought.strip(), (pre + post).strip()
    return "", text.strip()


def timed_call(label: str, prompt: str) -> dict:
    from openai import OpenAI
    client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY,
                    timeout=45.0, max_retries=0)   # one honest attempt — fail fast, no silent retries
    print(f"◈ {label} — {_clip(prompt, 70)}")
    reason, answer, t0, ttft = [], [], time.time(), None
    stream = client.chat.completions.create(
        model=config.MODEL, messages=[{"role": "user", "content": prompt}],
        max_tokens=280, temperature=0.2, stream=True)
    for chunk in stream:
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        rs = getattr(d, "reasoning", None) or (getattr(d, "model_extra", None) or {}).get("reasoning")
        if rs:
            reason.append(rs)
        if getattr(d, "content", None):
            answer.append(d.content)
        if ttft is None and (rs or getattr(d, "content", None)):
            ttft = time.time() - t0
    r_txt, a_txt = "".join(reason), "".join(answer)
    if not r_txt:                       # fallback: thinking inlined in content
        r_txt, a_txt = _split_think(a_txt)
    elapsed = time.time() - t0
    tok = lambda s: max(1, round(len(s) / 4)) if s else 0         # ~4 chars/token
    r_tok, a_tok = tok(r_txt), tok(a_txt)
    print(f"  ~ REASON  ≈{r_tok:>4} tok   {_clip(r_txt, 90) if r_txt else '(none surfaced)'}")
    print(f"  · ANSWER  ≈{a_tok:>4} tok   "
          f"{_clip(a_txt, 120) if a_txt else '(empty — reasoning spent the whole max_tokens budget: the tax, literally)'}")
    print(f"  ◆ ttft {f'{ttft:.2f}s' if ttft is not None else 'n/a'} · total {elapsed:.1f}s · "
          f"~{(r_tok + a_tok) / max(elapsed, 0.01):.1f} tok/s · {config.cost_note()}\n")
    return {"reason_tok": r_tok, "answer_tok": a_tok, "elapsed": elapsed}


def no_endpoint() -> None:
    print("✗ no endpoint reachable — to make this REAL, on your Spark (or laptop):")
    print("    ollama pull nemotron-3-nano        # then re-run this lab")
    print("    # or:  export DGX_CONN=cloud DGX_CLOUD_URL=https://integrate.api.nvidia.com/v1 \\")
    print("    #             DGX_API_KEY=nvapi-...\n")
    print("[no endpoint — showing expected output]")
    print("  ◈ EASY — What is 17 + 26? Reply with only the number.")
    print("    ~ REASON  ≈  30 tok   (a short check: 17+26=43…)")
    print("    · ANSWER  ≈   2 tok   43")
    print("  ◈ HARD — A data center may run only 3 of its 4 CRAC units…")
    print("    ~ REASON  ≈ 180 tok   (ranks COPs, drops 2.8, sanity-checks averages…)")
    print("    · ANSWER  ≈  55 tok   Keep the 2.8-COP unit off — it is the least…")
    print("  ▣ typical tax: HARD spends ~5-8x more REASON tokens than EASY.")


def main() -> None:
    print("━" * 64)
    print("  ▣ LAB 01 · The reasoning tax — think-tokens are not free")
    print("━" * 64 + "\n")
    if config.MODE != "real":
        no_endpoint()
        return
    print(f"▣ REAL · {config.MODEL} @ {config.safe_base_url()} ({config.conn_human()})\n")
    easy = timed_call("EASY", EASY)
    hard = timed_call("HARD", HARD)
    ratio = hard["reason_tok"] / max(1, easy["reason_tok"])
    print("▣ the tax, measured:")
    print(f"    EASY  reason≈{easy['reason_tok']} tok · answer≈{easy['answer_tok']} tok · {easy['elapsed']:.1f}s")
    print(f"    HARD  reason≈{hard['reason_tok']} tok · answer≈{hard['answer_tok']} tok · {hard['elapsed']:.1f}s")
    print(f"    → HARD spent ~{ratio:.1f}x the reasoning tokens of EASY.")
    print("✓ takeaway — reasoning depth scales with task difficulty; route easy calls")
    print("  to terse models/prompts and save the thinking budget for hard ones.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"\n✗ request failed ({type(e).__name__}: {e})")
        print("  check the endpoint (needs :PORT and /v1 — Ollama :11434/v1, NIM/vLLM :8000/v1)")

#!/usr/bin/env python3
"""Make **fine-tuning on a DGX** VISIBLE — in REAL or SIM mode.

Shared helpers for the demos: framing (banner/mode), the before/after EVAL that
proves domain adaptation worked (a real local generation when an endpoint is up,
simulated otherwise), and a thin pass-through to the training simulator.

Output is PLAIN text (no ANSI) so it renders in a terminal and the web app.
"""
from __future__ import annotations

import sys
import time
from urllib.parse import urlparse

import config

S_FT = "▣"
S_PROMPT = "  »"
S_ANSWER = "  ·"
S_METRIC = "  ◆"


def _p(line: str = "") -> None:
    print(line, flush=True)


def is_sim() -> bool:
    return config.MODE != "real"


def banner(part: str, title: str, level: str) -> None:
    _p("━" * 64)
    _p(f"  {part} — {title}   [{level}]")
    _p("━" * 64)
    _p("")


def mode_line() -> None:
    if is_sim():
        _p(f"{S_FT} MODE: SIM — simulating a DGX Spark fine-tune (no GPU needed).")
        _p(f"  connection: {config.CONN} ({config.conn_human()}) — nothing reachable yet.")
        _p("  the launch commands shown are exactly what you'd run on real hardware.")
    else:
        _p(f"{S_FT} MODE: REAL · connection = {config.CONN} ({config.conn_human()}).")
        _p(f"  eval runs against {config.MODEL} @ {config.safe_base_url()}.")
        _p("  (the training LOOP is still simulated unless you run it on a real DGX.)")
    _p("")


def _client():
    from openai import OpenAI
    return OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=120.0)


def generate(messages, *, model=None, max_tokens=config.DEFAULT_MAX_TOKENS,
             temperature=0.3, label=None) -> str:
    """One generation (REAL endpoint or a short simulated stub). Returns the text."""
    model = model or config.MODEL
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if label:
        _p(f"  {label}")
    if is_sim():
        sys_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
        tuned = "domain" in sys_prompt.lower() or "hvac" in sys_prompt.lower()
        text = (
            "[simulated] Set chilled-water supply to 6.5 °C and stage the second "
            "compressor; for a fouled filter (ΔP > 250 Pa) dispatch routine "
            "maintenance, not a CRITICAL alarm."
            if tuned else
            "[simulated] I'd need the manufacturer's manual to answer precisely — "
            "general HVAC guidance only."
        )
        _p(f"{S_ANSWER} {text}")
        return text
    out = ""
    stream = _client().chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens,
        temperature=temperature, stream=True)
    _p(f"{S_ANSWER} ", )
    for chunk in stream:
        if not chunk.choices:
            continue
        piece = chunk.choices[0].delta.content or ""
        out += piece
        print(piece, end="", flush=True)
    _p("")
    return out


def require_runtime() -> bool:
    if not is_sim():
        return True
    _p("ℹ  No live endpoint — eval will be simulated. To run the eval for REAL:")
    _p("     ollama run qwen3.6:35b-a3b-q8_0   (or set DGX_BASE_URL to a DGX)")
    _p("")
    return True


if __name__ == "__main__":
    _p("ftview.py is a helper imported by the demos in demos/.")
    sys.exit(0)

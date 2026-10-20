#!/usr/bin/env python3
"""Shared 'make it visible' engine — real endpoint (OpenAI SDK) or sim.

Each Week 23 app ships this file plus a `sim.py` (which provides installed_models(),
tok_s(model), and stream_generate(prompt, model)). Plain text so it renders in a
terminal and the web app.
"""
from __future__ import annotations

import sys
import time

import config
import sim

S = "▣"
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


def mode_line(model: str | None = None) -> None:
    model = model or config.MODEL
    if is_sim():
        _p(f"{S} MODE: SIM — simulated (no GPU needed). "
           f"connection: {config.CONN} ({config.conn_human()}).")
        _p("  the real commands shown are exactly what you'd run on the DGX.")
    else:
        _p(f"{S} MODE: REAL · connection = {config.CONN} ({config.conn_human()}).")
        _p(f"  endpoint: {config.safe_base_url()}   model: {model}   {config.cost_note()}")
    _p("")


def _client():
    from openai import OpenAI
    return OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=180.0)


def _extract_reasoning(delta):
    rs = getattr(delta, "reasoning", None)
    if rs:
        return rs
    extra = getattr(delta, "model_extra", None) or {}
    return extra.get("reasoning")


def _endpoint_error(e: Exception) -> None:
    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", "")
    _p("")
    _p(f"✗ request failed ({type(e).__name__}{f' · HTTP {status}' if status else ''}). "
       f"endpoint: {config.safe_base_url()}")
    if str(status) in ("404", "405"):
        _p("  URL likely needs the PORT and /v1 (Ollama :11434/v1, NIM/vLLM :8000/v1). Fix in 🔌 Connection.")
    elif str(status) == "401":
        _p("  401 = auth needed (tunnel creds in the URL, or a cloud API key).")


def generate(prompt: str, *, model: str | None = None, max_tokens: int = 400,
             title: str | None = None) -> dict:
    model = model or config.MODEL
    if title:
        _p(f"┌─ {title}")
        _p(f"│  {S_PROMPT.strip()} {prompt[:150]}")
        _p("└" + "─" * 58)
    out = {"answer": "", "tokens": 0, "elapsed_s": 0.0, "tok_s": 0.0}
    started = time.time()
    if is_sim():
        _p(f"{S_ANSWER} ANSWER (simulated):")
        for chunk in sim.stream_generate(prompt, model):
            out["answer"] += chunk
            print(chunk, end="", flush=True)
        _p("")
        out["elapsed_s"] = time.time() - started
        out["tok_s"] = sim.tok_s(model)
        out["tokens"] = max(1, round(len(out["answer"]) / 4))
        _p(f"{S_METRIC} ~{out['tokens']} tok · simulated ~{out['tok_s']:.0f} tok/s · on your DGX · $0.0000")
        return out
    try:
        _p(f"{S_ANSWER} ANSWER:")
        stream = _client().chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.3, stream=True)
        for chunk in stream:
            if not chunk.choices:
                continue
            d = chunk.choices[0].delta
            piece = (d.content or "") + (_extract_reasoning(d) or "")
            if piece:
                out["answer"] += piece
                print(piece, end="", flush=True)
        _p("")
    except Exception as e:  # noqa: BLE001
        _endpoint_error(e)
        return out
    out["elapsed_s"] = time.time() - started
    out["tokens"] = max(1, round(len(out["answer"]) / 4))
    out["tok_s"] = out["tokens"] / out["elapsed_s"] if out["elapsed_s"] else 0.0
    _p(f"{S_METRIC} ~{out['tokens']} tok in {out['elapsed_s']:.1f}s = ~{out['tok_s']:.1f} tok/s · {config.cost_note()}")
    return out


if __name__ == "__main__":
    _p("view.py is a helper imported by the demos in demos/.")
    sys.exit(0)

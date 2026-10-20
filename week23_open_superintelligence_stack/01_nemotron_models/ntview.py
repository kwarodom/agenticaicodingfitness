#!/usr/bin/env python3
"""Make **Nemotron reasoning** VISIBLE — REASON (thinking) → ANSWER, real or sim.

A thin wrapper over the OpenAI SDK pointed at your DGX/NIM/Ollama endpoint (REAL),
or ntsim (SIM). Nemotron is a reasoning model, so we surface the private REASON
channel separately from the ANSWER — the "RLM is the next thinking" idea made visible.
Plain text (no ANSI) so it renders in a terminal and the web app.
"""
from __future__ import annotations

import sys
import time
from urllib.parse import urlparse

import config
import ntsim

S_M = "▣"
S_PROMPT = "  »"
S_REASON = "  ~"
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
        _p(f"{S_M} MODE: SIM — simulating Nemotron on a DGX (no GPU needed).")
        _p(f"  connection: {config.CONN} ({config.conn_human()}). Real commands are shown.")
    else:
        _p(f"{S_M} MODE: REAL · connection = {config.CONN} ({config.conn_human()}).")
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


def reason(prompt: str, *, model: str | None = None, max_tokens: int = 512,
           show_reasoning: bool = True, title: str | None = None) -> dict:
    """One generation, narrating REASON (thinking) then ANSWER + tok/s."""
    model = model or config.MODEL
    out = {"reasoning": "", "answer": "", "tokens": 0, "elapsed_s": 0.0, "tok_s": 0.0}
    if title:
        _p(f"┌─ {title}")
        _p(f"│  {S_PROMPT.strip()} {prompt[:150]}")
        _p("└" + "─" * 58)
    started = time.time()

    if is_sim():
        in_r = in_a = False
        for kind, chunk in ntsim.stream_generate(prompt, model, show_reasoning=show_reasoning):
            if kind == "reason":
                if not in_r:
                    _p(f"{S_REASON} REASON (private, on-device):"); in_r = True
                out["reasoning"] += chunk; print(chunk, end="", flush=True)
            else:
                if not in_a:
                    if in_r:
                        _p("")
                    _p(f"{S_ANSWER} ANSWER:"); in_a = True
                out["answer"] += chunk; print(chunk, end="", flush=True)
        _p("")
        out["elapsed_s"] = time.time() - started
        out["tok_s"] = ntsim.spec_for(model).tok_s_spark
        out["tokens"] = max(1, round((len(out["reasoning"]) + len(out["answer"])) / 4))
        _p(f"{S_METRIC} ~{out['tokens']} tok · simulated ~{out['tok_s']:.0f} tok/s · on your DGX · $0.0000")
        return out

    try:
        in_r = in_a = False
        stream = _client().chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.4, stream=True)
        for chunk in stream:
            if not chunk.choices:
                continue
            d = chunk.choices[0].delta
            rs = _extract_reasoning(d)
            if rs and show_reasoning:
                if not in_r:
                    _p(f"{S_REASON} REASON (private, on-device):"); in_r = True
                out["reasoning"] += rs; print(rs, end="", flush=True)
            if getattr(d, "content", None):
                if not in_a:
                    if in_r:
                        _p("")
                    _p(f"{S_ANSWER} ANSWER:"); in_a = True
                out["answer"] += d.content; print(d.content, end="", flush=True)
        _p("")
    except Exception as e:  # noqa: BLE001
        _endpoint_error(e)
        return out
    out["elapsed_s"] = time.time() - started
    out["tokens"] = max(1, round((len(out["reasoning"]) + len(out["answer"])) / 4))
    out["tok_s"] = out["tokens"] / out["elapsed_s"] if out["elapsed_s"] else 0.0
    _p(f"{S_METRIC} ~{out['tokens']} tok in {out['elapsed_s']:.1f}s = ~{out['tok_s']:.1f} tok/s · {config.cost_note()}")
    return out


if __name__ == "__main__":
    _p("ntview.py is a helper imported by the demos in demos/.")
    sys.exit(0)

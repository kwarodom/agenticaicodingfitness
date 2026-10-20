#!/usr/bin/env python3
"""Make **sovereign inference on a DGX** VISIBLE — in REAL or SIM mode.

Demos call this one module and it does the right thing:

  • REAL — streams from a live OpenAI-compatible endpoint (Ollama / vLLM /
    llama.cpp on this laptop or a DGX you point DGX_BASE_URL at). A genuine
    on-device generation.
  • SIM  — streams from dgxsim.py: a topic-aware canned answer at a plausible
    DGX-Spark tok/s, with mock GB10 telemetry. Clearly labeled as simulated.

Every call narrates the same way, so the "magic" is never a black box:

    DGX     — where the compute is (this box / your DGX), and the run MODE
    PROMPT  — what we send to the model
    REASON  — the model's private reasoning channel (REAL thinking models only)
    ANSWER  — the answer, streamed token by token
    METRIC  — measured tok/s, token counts, and cloud cost: $0.0000
    ACT/OBS — (tool demos) the model calls YOUR function; the result feeds back

Output is PLAIN text (no ANSI) so it renders cleanly in a terminal and in the
tutorial web app's streaming pane.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse

import config
import dgxsim

# Symbols chosen to read well as plain text in a browser <pre> pane.
S_DGX = "▣"
S_GPU = "▤"
S_PROMPT = "  »"
S_REASON = "  ~"
S_ANSWER = "  ·"
S_METRIC = "  ◆"
S_ACT = "  →"
S_OBSERVE = "  ←"
S_RESULT = "═"


def _p(line: str = "") -> None:
    print(line, flush=True)


def _short(value: Any, n: int = 200) -> str:
    text = value if isinstance(value, str) else str(value)
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _client():
    from openai import OpenAI

    return OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=120.0)


def _endpoint_error(e: Exception) -> None:
    """Friendly diagnostic for a failed call (instead of a raw traceback)."""
    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", "")
    _p("")
    _p(f"✗ request to the endpoint failed ({type(e).__name__}{f' · HTTP {status}' if status else ''}).")
    _p(f"  endpoint: {config.safe_base_url()}")
    if str(status) in ("404", "405") or "Not Allowed" in str(e) or "Not Found" in str(e):
        _p("  This usually means the URL is missing the PORT and/or the /v1 path — it hit")
        _p("  a web server, not the model API. The endpoint must be an OpenAI-compatible URL:")
        _p("    • Ollama →  http://<dgx-host>:11434/v1")
        _p("    • vLLM   →  http://<dgx-host>:8000/v1")
        _p("  Fix it in the 🔌 Connection panel (include :PORT and /v1).")
    elif str(status) == "401":
        _p("  401 = the endpoint needs auth. For an ngrok --basic-auth tunnel, put creds in")
        _p("  the URL: https://user:pass@<id>.ngrok-free.app/v1  (or set the API key for cloud).")
    else:
        _p("  Check the 🔌 Connection panel: is the host reachable and the URL correct?")


def _is_local(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} or host.endswith(".local")


def is_sim() -> bool:
    return config.MODE != "real"


# ── lifecycle / framing ───────────────────────────────────────────────────────
def banner(part: str, title: str, level: str) -> None:
    _p("━" * 64)
    _p(f"  {part} — {title}   [{level}]")
    _p("━" * 64)
    _p("")


def mode_line(model: str | None = None) -> None:
    """Print which mode we're in and where the compute lives."""
    model = model or config.MODEL
    if is_sim():
        _p(f"{S_DGX} MODE: SIM — simulating a DGX Spark (GB10, 128 GB). No GPU needed.")
        _p(f"  connection: {config.CONN} ({config.conn_human()}) — but nothing is reachable yet.")
        _p(f"  the commands shown are exactly what you'd run on real hardware.")
    else:
        _p(f"{S_DGX} MODE: REAL · connection = {config.CONN} ({config.conn_human()}).")
        _p(f"  endpoint: {config.safe_base_url()}   model: {model}   cloud cost: $0.0000")
    _p("")


def require_runtime() -> bool:
    """Always True — SIM is a first-class fallback, so demos never hard-fail.

    In REAL mode it confirms the endpoint; in SIM it explains how to go live.
    """
    if not is_sim():
        return True
    _p(f"ℹ  No live endpoint at {config.BASE_URL} — running in SIM mode.")
    _p("   To run for REAL on a DGX Spark (or this laptop):")
    _p("     curl -fsSL https://ollama.com/install.sh | sh")
    _p("     ollama run qwen3.6:35b-a3b-q8_0     # DGX Spark default")
    _p("   …or point at a DGX you already have:")
    _p("     export DGX_BASE_URL=http://my-dgx-spark.local:11434/v1")
    _p("")
    return True


# ── GPU telemetry (real nvidia-smi if present, else simulated GB10) ───────────
def show_gpu(model: str | None = None, busy: bool = False) -> None:
    if is_sim():
        t = dgxsim.gpu_telemetry(model, busy=busy)
        _p(f"{S_GPU} GPU (simulated DGX Spark):")
        _p(dgxsim.render_smi(t))
        return
    import shutil
    import subprocess
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5).stdout.strip()
            _p(f"{S_GPU} GPU (real nvidia-smi):")
            for ln in out.splitlines():
                _p(f"    {ln}")
            return
        except Exception:
            pass
    _p(f"{S_GPU} GPU: nvidia-smi not available here (endpoint is remote/headless).")


@dataclass
class GenOutcome:
    reasoning: str = ""
    answer: str = ""
    tokens: int = 0
    elapsed_s: float = 0.0
    tok_per_s: float = 0.0
    tool_calls: list[dict] = field(default_factory=list)
    simulated: bool = False


def _extract_reasoning(delta) -> str | None:
    rs = getattr(delta, "reasoning", None)
    if rs:
        return rs
    extra = getattr(delta, "model_extra", None) or {}
    return extra.get("reasoning")


def generate(
    messages: list[dict] | str,
    *,
    model: str | None = None,
    max_tokens: int = config.DEFAULT_MAX_TOKENS,
    temperature: float = 0.4,
    show_reasoning: bool = True,
    title: str | None = None,
) -> GenOutcome:
    """Stream one generation (REAL endpoint or SIM), narrating PROMPT→ANSWER+tok/s."""
    model = model or config.MODEL
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

    if title:
        _p(f"┌─ {title}")
        _p(f"│  {S_PROMPT.strip()} {_short(last_user, 150)}")
        _p("└" + "─" * 60)

    out = GenOutcome(simulated=is_sim())
    started = time.time()

    if is_sim():
        _p(f"{S_ANSWER} ANSWER (simulated):")
        for chunk in dgxsim.stream_generate(last_user, model):
            out.answer += chunk
            print(chunk, end="", flush=True)
        _p("")
        out.elapsed_s = time.time() - started
        out.tokens = max(1, round(len(out.answer) / 4))
        out.tok_per_s = dgxsim.spec_for(model).tok_s
        _p(f"{S_METRIC} ~{out.tokens} tokens · simulated ~{out.tok_per_s:.0f} tok/s "
           f"on a DGX Spark · stayed 100% local · $0.0000")
        return out

    in_reason = in_answer = False
    try:
        stream = _client().chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens,
            temperature=temperature, stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            rs = _extract_reasoning(delta)
            if rs:
                if show_reasoning and not in_reason:
                    _p(f"{S_REASON} REASON (private, on-device):"); in_reason = True
                out.reasoning += rs
                if show_reasoning:
                    print(rs, end="", flush=True)
            if getattr(delta, "content", None):
                if not in_answer:
                    if in_reason:
                        _p("")
                    _p(f"{S_ANSWER} ANSWER:"); in_answer = True
                out.answer += delta.content
                print(delta.content, end="", flush=True)
    except Exception as e:  # noqa: BLE001
        _endpoint_error(e)
        return out

    if in_reason or in_answer:
        _p("")
    out.elapsed_s = time.time() - started
    chars = len(out.reasoning) + len(out.answer)
    out.tokens = max(1, round(chars / 4))
    out.tok_per_s = out.tokens / out.elapsed_s if out.elapsed_s else 0.0
    _p(f"{S_METRIC} ~{out.tokens} tokens in {out.elapsed_s:.1f}s "
       f"= ~{out.tok_per_s:.1f} tok/s · stayed 100% local · $0.0000")
    return out


def sovereignty_line(model: str | None = None) -> None:
    """The one line that defines this tutorial: the call stays on your hardware."""
    model = model or config.MODEL
    if is_sim():
        _p(f"{S_DGX} LOCAL ✓ (simulated) — in REAL mode this proves the call never left your DGX.")
        return
    conn = config.CONN
    if conn == "local":
        mark = "LOCAL ✓ on your DGX — data never leaves the box"
    elif conn == "tunnel":
        mark = "TUNNEL ✓ your DGX reached over an encrypted tunnel — still YOUR hardware, still sovereign"
    else:  # cloud
        mark = "CLOUD ⚠ a hosted provider — NOT your hardware (convenient, but not sovereign)"
    _p(f"{S_DGX} {mark}")
    _p(f"  endpoint: {config.safe_base_url()}   model: {model}   cloud cost: $0.0000")
    _p("")


if __name__ == "__main__":
    _p("dgxview.py is a helper imported by the demos in demos/.")
    _p("Run a demo instead, e.g.:  python demos/step01_dgx_hello.py")
    sys.exit(0)

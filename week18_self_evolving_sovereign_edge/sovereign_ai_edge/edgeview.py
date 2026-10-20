#!/usr/bin/env python3
"""Make **sovereign edge inference** VISIBLE.

A thin wrapper over the OpenAI SDK pointed at a LOCAL endpoint. Where the PDF
*explains* sovereign AI, this engine lets you watch a real local model think and
answer — and proves, every single call, that nothing left the machine:

    LOCAL    — the endpoint is on this host; your prompt never leaves it
    PROMPT   — what we send to the local model
    REASON   — the model's private reasoning channel, streamed live (dim)
    ANSWER   — the model's answer, streamed token by token
    METRIC   — measured tok/s, token counts, and cloud cost: $0.0000
    ACT/OBS  — (tool demos) the model calls YOUR function; the result feeds back

Output is PLAIN text (no ANSI) so it renders cleanly in a terminal and in the
tutorial web app's streaming pane. Every demo builds on this, so the "magic" of
on-device AI is never a black box.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse

import config

# Symbols chosen to read well as plain text in a browser <pre> pane.
S_LOCAL = "▣"
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


def _is_local(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} or host.endswith(".local")


def sovereignty_line(model: str | None = None) -> None:
    """Print the one line that defines this whole tutorial: the call stays local."""
    model = model or config.MODEL
    local = _is_local(config.BASE_URL)
    mark = "LOCAL ✓ data never leaves this machine" if local else \
           "REMOTE ⚠ this endpoint is NOT local — not sovereign!"
    _p(f"{S_LOCAL} {mark}")
    _p(f"  endpoint: {config.BASE_URL}   model: {model}   cloud cost: $0.0000")
    _p("")


@dataclass
class GenOutcome:
    """What one local generation produced — handy for chaining and metrics."""

    reasoning: str = ""
    answer: str = ""
    tokens: int = 0
    elapsed_s: float = 0.0
    tok_per_s: float = 0.0
    tool_calls: list[dict] = field(default_factory=list)


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
    """Stream one local chat completion, narrating REASON → ANSWER + tok/s.

    ``messages`` may be a plain string (treated as a single user turn) or a full
    OpenAI-style messages list. Returns a :class:`GenOutcome`.
    """
    model = model or config.MODEL
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    if title:
        _p(f"┌─ {title}")
        last_user = next((m["content"] for m in reversed(messages)
                          if m["role"] == "user"), "")
        _p(f"│  {S_PROMPT.strip()} {_short(last_user, 150)}")
        _p("└" + "─" * 60)

    out = GenOutcome()
    started = time.time()
    in_reason = in_answer = False

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
                    _p("")  # close the reasoning block
                _p(f"{S_ANSWER} ANSWER:"); in_answer = True
            out.answer += delta.content
            print(delta.content, end="", flush=True)

    if in_reason or in_answer:
        _p("")
    out.elapsed_s = time.time() - started
    # Approximate token count (exact usage isn't reliable mid-stream); ~4 chars/token.
    chars = len(out.reasoning) + len(out.answer)
    out.tokens = max(1, round(chars / 4))
    out.tok_per_s = out.tokens / out.elapsed_s if out.elapsed_s else 0.0
    _p(f"{S_METRIC} ~{out.tokens} tokens in {out.elapsed_s:.1f}s "
       f"= ~{out.tok_per_s:.1f} tok/s   ·   stayed 100% local   ·   $0.0000")
    return out


def call_with_tools(
    messages: list[dict],
    tools: list[dict],
    impls: dict[str, Callable[[dict], str]],
    *,
    model: str | None = None,
    max_rounds: int = 4,
    max_tokens: int = config.DEFAULT_MAX_TOKENS,
) -> GenOutcome:
    """Run a local function-calling loop: the model ACTs (calls your tool), you
    run it, the OBSERVEd result feeds back, repeat until it answers in words.

    This is the sovereign agent loop — the model deciding to use YOUR systems,
    entirely on-device. ``impls`` maps tool name → a function(args)->str.
    """
    model = model or config.MODEL
    client = _client()
    out = GenOutcome()
    started = time.time()

    for rnd in range(1, max_rounds + 1):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=tools,
            max_tokens=max_tokens, temperature=0.2,
        )
        msg = resp.choices[0].message
        calls = msg.tool_calls or []
        if not calls:
            out.answer = msg.content or ""
            if out.answer.strip():
                _p(f"{S_ANSWER} ANSWER: {out.answer.strip()}")
            break

        # Append the assistant's tool-call turn, then satisfy each call.
        messages.append({
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [{
                "id": c.id, "type": "function",
                "function": {"name": c.function.name, "arguments": c.function.arguments},
            } for c in calls],
        })
        for c in calls:
            name = c.function.name
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            out.tool_calls.append({"name": name, "args": args})
            _p(f"{S_ACT} ACT  [round {rnd}] model calls tool: {name}({_short(args, 120)})")
            fn = impls.get(name)
            result = fn(args) if fn else f"(no implementation for {name})"
            _p(f"{S_OBSERVE} OBSERVE: {_short(result, 200)}")
            _p("        ↑ result goes back to the local model — loop continues\n")
            messages.append({"role": "tool", "tool_call_id": c.id, "content": result})
    else:
        _p("  (hit max tool rounds — stopping the loop)")

    out.elapsed_s = time.time() - started
    _p(f"{S_METRIC} {len(out.tool_calls)} tool call(s) in {out.elapsed_s:.1f}s "
       f"·  stayed 100% local  ·  $0.0000")
    return out


# ── lifecycle helpers (mirror loopview's require_cli / banner) ────────────────
def require_local() -> bool:
    """Print a friendly note (and return False) if the local endpoint is down.

    These demos make REAL calls to a model on THIS machine. If nothing is
    serving, we explain how to start it instead of pretending to call a model.
    """
    if config.endpoint_up():
        return True
    _p(f"⚠  No local inference endpoint at {config.BASE_URL}")
    _p("   This tutorial is about SOVEREIGN AI — the model must run on your machine.")
    _p("   Start one (any OpenAI-compatible local server works):")
    _p("     curl -fsSL https://ollama.com/install.sh | sh   # then:")
    _p("     ollama run gemma4:12b      # Mac / DGX Spark / Ubuntu")
    _p("     ollama run gemma4:2b       # Raspberry Pi / low-RAM")
    _p("   The endpoint comes up at http://localhost:11434  (OpenAI API at /v1).")
    return False


def banner(part: str, title: str, level: str) -> None:
    _p("━" * 64)
    _p(f"  {part} — {title}   [{level}]")
    _p("━" * 64)
    _p("")


if __name__ == "__main__":
    _p("edgeview.py is a helper imported by the demos in demos/.")
    _p("Run a demo instead, e.g.:  python demos/step01_local_inference.py")
    sys.exit(0)

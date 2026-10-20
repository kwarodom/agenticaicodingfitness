#!/usr/bin/env python3
"""The **switchable brain** — the heart of v2.

The self-evolving agent's memory architecture is brain-agnostic: the same agent
can think with a local sovereign model on a DGX, with Claude in the cloud, or with
a scripted stub for $0 offline learning. `chat()` hides which one is active.

    BRAIN=local   → OpenAI-compatible call to the DGX/Ollama endpoint (sovereign)
    BRAIN=claude  → Anthropic Claude   (needs ANTHROPIC_API_KEY + `anthropic`)
    BRAIN=sim     → a deterministic stub (no model, no network)

This is the whole point: swapping the brain must not touch the memory engine.
"""
from __future__ import annotations

import config


def name() -> str:
    return config.BRAIN


def is_sovereign() -> bool:
    return config.BRAIN == "local"


def where() -> str:
    if config.BRAIN == "local":
        return (f"local model ({config.MODEL}) via the '{config.CONN}' connection "
                f"({config.conn_human()}) @ {config.safe_base_url()}")
    if config.BRAIN == "claude":
        return f"Claude ({config.CLAUDE_MODEL}) in the cloud — prompts DO leave the box"
    return "simulated brain (no model) — fully offline, $0"


def label() -> str:
    return {"local": f"DGX·{config.MODEL}", "claude": f"cloud·{config.CLAUDE_MODEL}",
            "sim": "sim"}.get(config.BRAIN, config.BRAIN)


def chat(messages, *, max_tokens: int = config.DEFAULT_MAX_TOKENS,
         temperature: float = 0.3) -> str:
    """One completion from whichever brain is active. Returns plain text."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    if config.BRAIN == "local":
        from openai import OpenAI
        client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=180.0)
        resp = client.chat.completions.create(
            model=config.MODEL, messages=messages,
            max_tokens=max_tokens, temperature=temperature)
        msg = resp.choices[0].message
        content = (msg.content or "").strip()
        if content:
            return content
        # Thinking models (e.g. gemma) may put everything in a reasoning channel
        # and leave content empty if the budget is tight — fall back to it.
        extra = getattr(msg, "model_extra", None) or {}
        reasoning = getattr(msg, "reasoning", None) or extra.get("reasoning") or ""
        return reasoning.strip()

    if config.BRAIN == "claude":
        import anthropic
        client = anthropic.Anthropic()
        sys_msg = " ".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        resp = client.messages.create(
            model=config.CLAUDE_MODEL, max_tokens=max_tokens,
            system=sys_msg or None, messages=turns)
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()

    # sim — deterministic, topic-aware stub
    return _sim_reply(messages)


def _sim_reply(messages) -> str:
    user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    u = user.lower()
    if "skill" in u or "procedure" in u or "steps" in u:
        return ("1. get_room_telemetry(room)\n2. if occupied and |temp-setpoint|>3 → "
                "dispatch CRITICAL\n3. elif filter_pa>250 → dispatch ROUTINE\n4. log outcome")
    if "fact" in u or "summarize" in u or "consolidate" in u:
        return ("- Occupied rooms >3°C from setpoint are CRITICAL.\n"
                "- Filter ΔP>250 Pa on empty rooms is ROUTINE.\n"
                "- Night setback is 25.5°C cooling for unoccupied rooms.")
    return ("[sim] Based on recalled memory, I'd read telemetry, classify priority, "
            "and dispatch — CRITICAL for the occupied hot room.")

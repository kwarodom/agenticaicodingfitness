#!/usr/bin/env python3
"""Make the **self-evolving agent** VISIBLE — recall → think → remember.

Thin framing over brain.py + memory.py so each demo reads the same way:

    BRAIN   — which model is thinking, and whether it's sovereign
    RECALL  — what memory was injected before acting
    THINK   — the brain's answer (local DGX model / Claude / sim)
    REMEMBER— what got written back to the on-DGX memory

Output is PLAIN text so it renders in a terminal and the web app.
"""
from __future__ import annotations

import sys

import brain
import memory


def _p(line: str = "") -> None:
    print(line, flush=True)


def banner(part: str, title: str, level: str) -> None:
    _p("━" * 64)
    _p(f"  {part} — {title}   [{level}]")
    _p("━" * 64)
    _p("")


def brain_line() -> None:
    sov = "✓ sovereign" if brain.is_sovereign() else (
        "⚠ NOT sovereign (cloud)" if brain.name() == "claude" else "· offline sim")
    _p(f"▣ BRAIN: {brain.name()}  [{sov}]")
    _p(f"  thinking with: {brain.where()}")
    m = memory.stats()
    _p(f"  memory on DGX: {m['episodes']} episodes · {m['facts']} facts · {m['skills']} skills")
    _p("")


def run_task(task: str, *, session: str = "default", remember: bool = True,
             show_recall: bool = True) -> str:
    """Recall memory, think, and (optionally) log the episode. Returns the answer."""
    ctx = memory.recall(task)
    if show_recall:
        if ctx:
            _p("← RECALL (injected from on-DGX memory):")
            for ln in ctx.splitlines():
                _p("    " + ln)
        else:
            _p("← RECALL: (empty — the agent has no memory yet)")
    messages = [
        {"role": "system", "content": "You are a smart-hotel HVAC operations agent. "
         "Use any recalled memory. Be specific: setpoints, thresholds, CRITICAL/ROUTINE."},
    ]
    if ctx:
        messages.append({"role": "system", "content": ctx})
    messages.append({"role": "user", "content": task})

    _p(f"~ THINK ({brain.label()}):")
    answer = brain.chat(messages, max_tokens=300)
    for ln in answer.splitlines():
        _p("    " + ln)

    if remember:
        memory.log_episode("task", task, session)
        memory.log_episode("answer", answer, session)
        _p("→ REMEMBER: logged this episode to the on-DGX episodic store.")
    _p("")
    return answer


if __name__ == "__main__":
    _p("sevview.py is a helper imported by the demos in demos/.")
    sys.exit(0)

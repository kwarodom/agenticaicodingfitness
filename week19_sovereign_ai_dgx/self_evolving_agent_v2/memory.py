#!/usr/bin/env python3
"""The agent's **tripartite memory** — and it all lives on the DGX.

Three stores, mirroring human memory (the Week 18 model, kept lightweight here):

  • EPISODIC   — episodes.jsonl: an append-only log of what happened, every run.
  • SEMANTIC   — MEMORY.md: durable, consolidated FACTS distilled from episodes.
  • PROCEDURAL — skills/: reusable how-to SKILLS the agent writes for itself.

CONSOLIDATION is the "sleep" step: read raw episodes → distill facts + skills.
RECALL injects the relevant facts + skills before the agent acts, so each run
starts smarter than the last. The sovereignty point: this memory IS your data —
on a DGX it never leaves the building.
"""
from __future__ import annotations

import json
import time

import brain
import config


# ── EPISODIC ──────────────────────────────────────────────────────────────────
def log_episode(kind: str, content: str, session: str = "default") -> None:
    config.ensure_memory()
    rec = {"ts": round(time.time(), 3), "session": session, "kind": kind, "content": content}
    with config.EPISODES.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def episodes() -> list[dict]:
    if not config.EPISODES.exists():
        return []
    return [json.loads(ln) for ln in config.EPISODES.read_text().splitlines() if ln.strip()]


# ── SEMANTIC ──────────────────────────────────────────────────────────────────
def facts() -> list[str]:
    if not config.SEMANTIC.exists():
        return []
    return [ln[2:].strip() for ln in config.SEMANTIC.read_text().splitlines()
            if ln.strip().startswith("- ")]


def _clean_fact(line: str) -> str:
    """Strip common bullet/numbering prefixes a model might emit."""
    import re
    return re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()


def write_facts(new_facts: list[str]) -> int:
    config.ensure_memory()
    existing = facts()
    merged = existing[:]
    for f in new_facts:
        f = _clean_fact(f)
        if f and f not in merged:
            merged.append(f)
    body = "# Agent semantic memory (consolidated facts — lives on the DGX)\n\n" + \
           "\n".join(f"- {f}" for f in merged) + "\n"
    config.SEMANTIC.write_text(body)
    return len(merged) - len(existing)


# ── PROCEDURAL ────────────────────────────────────────────────────────────────
def skills() -> list[str]:
    config.ensure_memory()
    return sorted(p.stem for p in config.SKILLS_DIR.glob("*.md"))


def save_skill(slug: str, body: str) -> None:
    config.ensure_memory()
    (config.SKILLS_DIR / f"{slug}.md").write_text(body.rstrip() + "\n")


def load_skill(slug: str) -> str:
    p = config.SKILLS_DIR / f"{slug}.md"
    return p.read_text() if p.exists() else ""


# ── CONSOLIDATION (the "sleep" loop) ──────────────────────────────────────────
def consolidate() -> dict:
    """Distill recent episodes into durable facts + (maybe) a skill, via the brain."""
    eps = episodes()
    if not eps:
        return {"added_facts": 0, "added_skill": None, "episodes_seen": 0}

    transcript = "\n".join(f"[{e['kind']}] {e['content']}" for e in eps[-20:])
    fact_text = brain.chat([
        {"role": "system", "content": "You consolidate an agent's episodic log into "
         "3-5 durable, reusable FACTS. Output ONLY a markdown bullet list, '- ' each."},
        {"role": "user", "content": f"Episodes:\n{transcript}\n\nConsolidate the facts."},
    ], max_tokens=900)   # thinking models need room to reason AND emit the list
    import re
    new_facts = [ln for ln in fact_text.splitlines()
                 if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+\S", ln)]
    if not new_facts:   # model wrote prose, not a list → keep non-empty sentences
        new_facts = [ln for ln in fact_text.splitlines() if len(ln.strip()) > 15][:5]
    added = write_facts(new_facts)

    added_skill = None
    if not skills():           # write one reusable skill the first time we consolidate
        body = brain.chat([
            {"role": "system", "content": "Write a concise reusable SKILL as numbered "
             "steps the agent can follow next time. Output only the steps."},
            {"role": "user", "content": f"From these episodes, write the 'hvac-triage' "
             f"skill:\n{transcript}"},
        ], max_tokens=700)
        save_skill("hvac-triage", f"# Skill: hvac-triage\n\n{body}\n")
        added_skill = "hvac-triage"

    return {"added_facts": added, "added_skill": added_skill, "episodes_seen": len(eps)}


# ── RECALL (inject memory before acting) ──────────────────────────────────────
def recall(task: str) -> str:
    """Build a memory context block to prepend to the agent's prompt."""
    fs = facts()
    sk = skills()
    if not fs and not sk:
        return ""
    parts = ["Relevant memory (recalled from the DGX-resident store):"]
    if fs:
        parts.append("Facts:")
        parts += [f"  - {f}" for f in fs[:6]]
    if sk:
        parts.append(f"Skills available: {', '.join(sk)}")
        body = load_skill(sk[0])
        if body:
            parts.append("Top skill:\n" + "\n".join("  " + ln for ln in body.splitlines()[:8]))
    return "\n".join(parts)


def stats() -> dict:
    return {"episodes": len(episodes()), "facts": len(facts()), "skills": len(skills())}

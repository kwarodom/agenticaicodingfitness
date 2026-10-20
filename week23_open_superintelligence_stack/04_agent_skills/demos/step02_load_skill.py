#!/usr/bin/env python3
"""PART 2 · Load a skill into a frontier agent  [INTERMEDIATE]

Progressive disclosure: a frontier agent discovers a skill, reads its SKILL.md
metadata (name + description + when-to-use), then loads and INVOKES its tools only
when the task needs them. This demo walks the discover → read → invoke → result loop
(printed as a trace in SIM; a real endpoint tool-call loop in REAL).

Run:  python demos/step02_load_skill.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import view  # noqa: E402

# A skill packaged as a tool the frontier agent can call (this is the SKILL.md tool surface).
TOOLS = [
    {"type": "function", "function": {
        "name": "nemo_retriever_search",
        "description": "Skill: NeMo Retriever — search YOUR internal document store.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}},
]

# The SKILL.md the agent reads first (metadata only — progressive disclosure).
SKILL_MD = """\
---
name: nemo-retriever
description: Answer questions from the company's internal docs (RAG). Use when the
  user asks about OUR products, policies, or runbooks.
tools: [nemo_retriever_search]
---
# NeMo Retriever skill
Query the sovereign document store; cite retrieved chunks; never guess."""

_DOCS = {"return policy": "Returns accepted within 30 days with receipt (policy DOC-114)."}


def _impl(name, args):
    if name == "nemo_retriever_search":
        q = (args.get("query") or "").lower()
        hit = next((v for k, v in _DOCS.items() if k in q), "no matching chunk")
        return json.dumps({"chunk": hit, "source": "internal-docs"})
    return json.dumps({})


def main() -> None:
    view.banner("PART 2", "Load a skill into a frontier agent", "INTERMEDIATE")
    view.mode_line()

    print("Progressive disclosure — the agent reads metadata first, loads tools on demand:\n")
    print("SKILL.md the agent discovers:")
    for line in SKILL_MD.splitlines():
        print("  │ " + line)
    print()

    if view.is_sim():
        print("SIM — the discover → read → invoke → result loop (REAL mode executes it):")
        print("  → agent DISCOVERS skill 'nemo-retriever' in the catalog")
        print("  ← reads SKILL.md metadata (name, description, when-to-use)")
        print("  ~ reason: user asks about OUR policy → this skill applies → load its tool")
        print("  → agent invokes nemo_retriever_search({'query':'return policy'})")
        print("  ← {'chunk':'Returns accepted within 30 days…','source':'internal-docs'}")
        print("  · answer: Per DOC-114, returns are accepted within 30 days with a receipt.")
    else:
        client = view._client()
        messages = [{"role": "system", "content": "You have loaded the nemo-retriever skill. "
                     "When the user asks about company policy, call nemo_retriever_search "
                     "before answering, then answer from the retrieved chunk only."},
                    {"role": "user", "content": "What's our return policy?"}]
        for _ in range(4):
            try:
                r = client.chat.completions.create(model=config.MODEL, messages=messages,
                                                   tools=TOOLS, max_tokens=400, temperature=0.2)
            except Exception as e:  # noqa: BLE001
                view._endpoint_error(e); return
            msg = r.choices[0].message
            calls = msg.tool_calls or []
            if not calls:
                print("  · answer:", (msg.content or "").strip()[:300]); break
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.function.name,
                                                          "arguments": c.function.arguments}} for c in calls]})
            for c in calls:
                args = json.loads(c.function.arguments or "{}")
                res = _impl(c.function.name, args)
                print(f"  → ACT {c.function.name}({args})")
                print(f"  ← OBSERVE {res}")
                messages.append({"role": "tool", "tool_call_id": c.id, "content": res})

    print("\nTakeaway: a skill is discovered by metadata and loaded on demand — the agent")
    print("gains a capability without you touching its code. Next: connect it to YOUR data.")


if __name__ == "__main__":
    main()

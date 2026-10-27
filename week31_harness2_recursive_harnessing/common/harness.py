#!/usr/bin/env python3
"""harness — the harness tree as a value, plus the integrity checks that keep a self-editing loop honest.

A harness is a dict {path: text}. The eight Harness² components are recognised by path (OpenCode layout, as in
the paper and the repository's `core/tree.py`):

    C1 systemprompt.md        always on     C5 sub_agents/<name>/AGENT.md   task tool (own context)
    C2 guardrails.md          always on     C6 skills/<name>/SKILL.md       skill tool (on demand)
    C3 LongTermMEMORY.md      always on     C7 scripts/<name>.py            run via bash
    C4 tool_guidance.md       always on     C8 plugin/<name>.ts             inside the tool loop

What this module adds for Ops Assistant (EXAMPLE design for this course):
  • a PLATFORM-OWNED block in guardrails.md that `validate_tree` refuses to let anyone edit,
  • `always_on_tokens` (≈ characters / 4, an ESTIMATE) and `prefix_hash` (what a prompt cache keys on),
  • the course's version of two repository checks — the measurement-word ban (spec.py applies it to visible
    prompt fields) and the grader-verdict check (library.py) — applied here to every tree file, and `review_script`, a static screen run BEFORE a probe may execute a candidate script.

    from harness import base_harness, component_of, always_on_tokens, validate_tree, diff
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

ALWAYS_ON = ("systemprompt.md", "guardrails.md", "LongTermMEMORY.md", "tool_guidance.md")

COMPONENTS = [  # (code, name, path pattern, how it reaches the model)
    ("C1", "System prompt", "systemprompt.md", "every call"),
    ("C2", "Guardrails", "guardrails.md", "every call"),
    ("C3", "Memory", "LongTermMEMORY.md", "every call"),
    ("C4", "Tool guidance", "tool_guidance.md · tool_descriptions/", "every call"),
    ("C5", "Subagents", "sub_agents/<name>/AGENT.md", "task tool, own context"),
    ("C6", "Skills", "skills/<name>/SKILL.md", "skill tool, on demand"),
    ("C7", "Scripts", "scripts/<name>.py", "run via bash"),
    ("C8", "Plugins", "plugin/<name>.ts", "inside the tool loop"),
]

PLATFORM_BEGIN = "## Platform-owned (not editable)"
PLATFORM_END = "## Task constraints (editable)"

# From the repository's core/spec.py: visible prompt fields may not contain these words.
MEASUREMENT_WORDS = ("benchmark", "evaluation", "grader", "rubric", "score")

# Modules a candidate script may import. Anything else is refused before a probe runs it.
SAFE_IMPORTS = {"csv", "json", "sys", "math", "decimal", "statistics", "datetime", "re", "collections"}
BANNED_CALLS = {"eval", "exec", "compile", "__import__", "open", "input", "breakpoint"}


# ── the base harness: today's HVAC Expert agent, exported into a tree ───────────────────────
BASE_SYSTEMPROMPT = """\
You are the HVAC Expert agent of Ops Assistant for a hotel engineering team.
Read the wake, gather evidence with the read-only tools, and reply with one deliverable.
Physical or third-party actions are never taken directly: write an ActionProposal for a named human.
"""

BASE_GUARDRAILS = f"""\
{PLATFORM_BEGIN}
- PLATFORM: every physical or third-party action goes through an ActionProposal approved by a named human.
- PLATFORM: tools are read-only while a harness review routine runs.
- PLATFORM: tenant data never crosses tenant classes.
{PLATFORM_END}
- Inputs are read-only.
"""

BASE_MEMORY = """\
- Chilled-water plants in this tenant class report electrical power in kW and cooling in tons (RT).
- Engineering teams read English and Thai.
"""

BASE_TOOL_GUIDANCE = """\
- alarms.list(window): BMS alarms in a time window.
- telemetry.query(point, days): readings for one point.
- cmms.work_orders(equipment): open and recent work orders for one piece of equipment.
- docs.search(query): excerpts from the O&M manuals.
"""


def base_harness() -> dict[str, str]:
    """Harness v1 — no skills, scripts, subagents or plugins yet (the gaps the loop will find)."""
    return {"systemprompt.md": BASE_SYSTEMPROMPT, "guardrails.md": BASE_GUARDRAILS,
            "LongTermMEMORY.md": BASE_MEMORY, "tool_guidance.md": BASE_TOOL_GUIDANCE}


# ── reading a tree ────────────────────────────────────────────────────────────
def component_of(path: str) -> str:
    """'C1'…'C8' for a harness path, '' for anything that is not a component."""
    if path == "systemprompt.md":
        return "C1"
    if path == "guardrails.md":
        return "C2"
    if path == "LongTermMEMORY.md":
        return "C3"
    if path == "tool_guidance.md" or path.startswith("tool_descriptions/"):
        return "C4"
    if re.fullmatch(r"sub_agents/[a-z0-9_\-]+/AGENT\.md", path):
        return "C5"
    if re.fullmatch(r"skills/[a-z0-9_\-]+/(SKILL\.md|scripts/[a-z0-9_.\-]+\.py)", path):
        return "C6"
    if re.fullmatch(r"scripts/[a-z0-9_][a-z0-9_]*\.py", path):
        return "C7"
    if re.fullmatch(r"plugin/[a-z0-9_\-]+\.(ts|yaml)", path):
        return "C8"
    return ""


def is_always_on(path: str) -> bool:
    return path in ALWAYS_ON


def tokens(text: str) -> int:
    """ESTIMATE: ≈ 4 characters per token (English). Good enough to compare versions, not to bill."""
    return (len(text) + 3) // 4


def always_on_tokens(tree: dict[str, str]) -> int:
    return sum(tokens(t) for p, t in tree.items() if is_always_on(p))


def prefix_hash(tree: dict[str, str]) -> str:
    """Hash of the always-on text in fixed order — what a prompt cache keys on. Changes ⇒ cold prefix."""
    h = hashlib.sha256()
    for p in ALWAYS_ON:
        h.update(p.encode() + b"\0" + tree.get(p, "").encode() + b"\0")
    return h.hexdigest()[:12]


def definition_hash(tree: dict[str, str]) -> str:
    """Hash of the whole tree (every file, sorted) — the version's identity, the definition_hash a production system would keep."""
    h = hashlib.sha256()
    for p in sorted(tree):
        h.update(p.encode() + b"\0" + tree[p].encode() + b"\0")
    return h.hexdigest()[:12]


def skills(tree: dict[str, str]) -> dict[str, str]:
    return {p.split("/")[1]: t for p, t in tree.items() if re.fullmatch(r"skills/[^/]+/SKILL\.md", p)}


def scripts(tree: dict[str, str]) -> dict[str, str]:
    return {p: t for p, t in tree.items() if component_of(p) in ("C7",) or "/scripts/" in p}


def plugins(tree: dict[str, str]) -> dict[str, str]:
    return {p: t for p, t in tree.items() if component_of(p) == "C8"}


def skill_trigger(skill_md: str) -> str:
    m = re.search(r"^trigger:\s*(.+)$", skill_md, re.M)
    return m.group(1).strip() if m else ""


def always_on_text(tree: dict[str, str]) -> str:
    return "\n".join(tree.get(p, "") for p in ALWAYS_ON)


def summary_rows(tree: dict[str, str]) -> list[list]:
    """One row per file: path, component, reach, tokens."""
    rows = []
    for p in sorted(tree, key=lambda x: (component_of(x) or "C9", x)):
        c = component_of(p)
        reach = next((r for code, _, _, r in COMPONENTS if code == c), "?")
        rows.append([p, c or "—", reach, tokens(tree[p])])
    return rows


def diff(a: dict[str, str], b: dict[str, str]) -> list[dict]:
    """Per-file changes from a to b: [{path, component, change: added|removed|modified, delta_tokens}]."""
    out = []
    for p in sorted(set(a) | set(b)):
        if a.get(p) == b.get(p):
            continue
        change = "added" if p not in a else "removed" if p not in b else "modified"
        out.append({"path": p, "component": component_of(p), "change": change,
                    "delta_tokens": tokens(b.get(p, "")) - tokens(a.get(p, ""))})
    return out


def changed_components(a: dict[str, str], b: dict[str, str]) -> list[str]:
    return sorted({d["component"] or d["path"] for d in diff(a, b)})


# ── integrity checks ──────────────────────────────────────────────────────────
def platform_block(guardrails: str) -> str:
    i, j = guardrails.find(PLATFORM_BEGIN), guardrails.find(PLATFORM_END)
    return guardrails[i:j] if 0 <= i < j else ""


def validate_tree(tree: dict[str, str], base: dict[str, str] | None = None) -> list[str]:
    """Errors that make a candidate harness invalid (empty list = valid). Never raises.

    • every path is one of the eight components;
    • the platform-owned block of guardrails.md is byte-identical to the base's (boundary immutability);
    • every script a skill invokes exists in the tree (the repository rejects such trees too);
    • every script passes `review_script`.
    """
    base = base or base_harness()
    errs = []
    for p in tree:
        if not component_of(p):
            errs.append(f"{p}: not a harness component path")
    if platform_block(tree.get("guardrails.md", "")) != platform_block(base.get("guardrails.md", "")):
        errs.append("guardrails.md: the platform-owned block was changed — owned by the platform, not editable")
    for name, md in skills(tree).items():
        for ref in re.findall(r"python3?\s+((?:skills/[\w\-]+/)?scripts/[\w.\-]+\.py)", md):
            if ref not in tree:
                errs.append(f"skills/{name}/SKILL.md invokes {ref}, which is not in the tree")
    for p, t in scripts(tree).items():
        for finding in review_script(t):
            errs.append(f"{p}: {finding}")
    return errs


def measurement_words(text: str) -> list[str]:
    """The repository's spec.py ban: a visible prompt field must describe the WORK, not its measurement."""
    return [w for w in MEASUREMENT_WORDS if re.search(rf"\b{w}", text, re.I)]


GRADER_VERDICT = re.compile(r"(grader|rubric|judge)\s*(expects|wants|verdict|said|score)|score[ds]?\s*[:=]\s*\d|"
                            r"\bpassed the (check|test)s?\b", re.I)


def grader_verdicts(tree: dict[str, str]) -> list[str]:
    """Paths whose text carries a grader verdict or measurement language (refused before the library stores it)."""
    return [p for p, t in tree.items() if GRADER_VERDICT.search(t) or measurement_words(t)]


def review_script(text: str) -> list[str]:
    """Static screen of a candidate script BEFORE a probe may run it (the editor never runs it at all).

    Allowed: imports from SAFE_IMPORTS, reading argv and stdin, printing. Refused: anything else — network,
    subprocess, os, file I/O through open(), eval/exec. Returns findings; [] = may run in the probe sandbox.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        return [f"does not parse ({e.msg}, line {e.lineno})"]
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [f"imports {a.name}" for a in node.names if a.name.split(".")[0] not in SAFE_IMPORTS]
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in SAFE_IMPORTS:
                out.append(f"imports from {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            out.append(f"calls {node.func.id}()")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr not in ("__name__",):
            out.append(f"touches dunder attribute {node.attr}")
    return sorted(set(out))


# ── on disk ───────────────────────────────────────────────────────────────────
def write_tree(tree: dict[str, str], root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for p, t in tree.items():
        f = root / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(t, encoding="utf-8")
    return root


def read_tree(root: Path) -> dict[str, str]:
    return {str(f.relative_to(root)): f.read_text(encoding="utf-8")
            for f in sorted(root.rglob("*")) if f.is_file() and f.name != "HARNESS.json"}


def harness_json(tree: dict[str, str], *, version: str, parent: str | None, created_by: str,
                 approved_by: str | None = None, state: str = "draft") -> str:
    """The version record the Ops Assistant design keeps beside each tree (EXAMPLE schema)."""
    return json.dumps({"version": version, "parent": parent, "definition_hash": definition_hash(tree),
                       "prefix_hash": prefix_hash(tree), "always_on_tokens": always_on_tokens(tree),
                       "state": state, "created_by": created_by, "approved_by": approved_by}, indent=2)

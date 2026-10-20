#!/usr/bin/env python3
"""LAB 01 · Author a Skill and watch an agent DISCOVER it — fully offline.

You write a real SKILL.md package to disk, then run the discovery pass an agent
runs: scan the catalog dir, read ONLY the frontmatter metadata, and load the full
body on demand. The point is progressive disclosure — you measure the token cost
of metadata-vs-body yourself, with the ~4 chars/token rule the course uses.

Run:  .venv/bin/python week23/04_agent_skills/labs/lab01_author_a_skill.py
Needs: nothing — no endpoint, no GPU. Pure files, like real skills.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

SKILLS = {
    "hotel-telemetry": """\
---
name: hotel-telemetry
description: Query room sensors and file work orders for the hotel BMS. Use when
  the user asks about room temperature, energy, alarms, or maintenance.
tools: [query_room_telemetry, file_work_order]
---
# Hotel telemetry skill
Query pattern: query_room_telemetry(room="1203") -> {temp_c, setpoint_c, alarm}.
If temp_c exceeds setpoint_c by more than 3C AND alarm is set, file a work order
with file_work_order(room, summary) citing SOP-HVAC-07. Never guess sensor values
— always call the tool first. Cite the room id and reading in every answer.
""" + "Worked examples:\n" + "\n".join(
        f"  example {i}: room {1200+i} at {24+i%4}.{i}C -> "
        f"{'file order (SOP-HVAC-07)' if i % 3 == 0 else 'within band, no action'}"
        for i in range(1, 25)) + "\n",
    "ipmvp-savings": """\
---
name: ipmvp-savings
description: Compute IPMVP-verified energy savings (baseline - reporting +/- adjustments).
  Use when the user asks how much energy or money a retrofit actually saved.
tools: [compute_savings]
---
# IPMVP savings skill
Savings = Baseline - Reporting +/- Routine +/- Non-Routine adjustments. Always
state the option (A/B/C/D) and the baseline period. Never report savings without
an adjustment note.
""",
}


def frontmatter(text: str) -> dict:
    """Parse the YAML-ish frontmatter block (name / description / tools) — stdlib only."""
    meta, in_fm, key = {}, False, None
    for line in text.splitlines():
        if line.strip() == "---":
            if in_fm:
                break
            in_fm = True
            continue
        if in_fm:
            if ":" in line and not line.startswith(" "):
                key, _, val = line.partition(":")
                meta[key.strip()] = val.strip()
            elif key and line.startswith(" "):          # folded continuation line
                meta[key] += " " + line.strip()
    return meta


def toks(text: str) -> int:
    return max(1, round(len(text) / 4))                  # the course's ~4 chars/token rule


def main() -> None:
    print("━" * 64)
    print("  LAB 01 — Author a Skill + progressive-disclosure math   [OFFLINE]")
    print("━" * 64 + "\n")

    # ▣ STEP 1 — author: write a real skills catalog to disk (this IS the format)
    root = config.ensure_sandbox() / "skills"
    print(f"▣ STEP 1 · AUTHOR — writing {len(SKILLS)} skill packages to {root}")
    for name, body in SKILLS.items():
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(body)
        print(f"  ✓ {name}/SKILL.md  ({len(body)} chars)")
    print("  — same layout as github.com/NVIDIA/skills: one dir per skill, SKILL.md inside.\n")

    # ▣ STEP 2 — discover: the agent's cheap pass reads ONLY metadata
    print("▣ STEP 2 · DISCOVER — scan the catalog, read frontmatter only:")
    catalog = []
    for md in sorted(root.glob("*/SKILL.md")):
        meta = frontmatter(md.read_text())
        catalog.append((md, meta))
        print(f"  ◈ {meta.get('name', md.parent.name):<16} — {meta.get('description', '')[:74]}")
    print()

    # ▣ STEP 3 — the progressive-disclosure math, measured on YOUR files
    print("▣ STEP 3 · MEASURE — metadata-only vs full-body context cost (~4 chars/tok):")
    total_meta = total_full = 0
    for md, meta in catalog:
        full = toks(md.read_text())
        m = toks(" ".join(f"{k}: {v}" for k, v in meta.items()))
        total_meta += m
        total_full += full
        print(f"  {meta.get('name', ''):<16} metadata ≈ {m:>4} tok   full body ≈ {full:>4} tok")
    print(f"  {'CATALOG TOTAL':<16} metadata ≈ {total_meta:>4} tok   full body ≈ {total_full:>4} tok")
    print(f"  → discovery costs {total_meta} tok; the agent loads a body ONLY when a task")
    print(f"    matches — at 100 skills that's the difference between a slim prompt and")
    print(f"    ~{total_full // len(catalog) * 100:,} tok of dead weight. That is progressive disclosure.\n")

    # ▣ STEP 4 — a task arrives; load exactly one body on demand
    task = "Room 1203 is reading 29.4C against a 23C setpoint — what do we do?"
    print(f"▣ STEP 4 · LOAD ON DEMAND — task: {task!r}")
    hit = next((c for c in catalog if "telemetry" in c[1].get("description", "").lower()
                or "sensors" in c[1].get("description", "").lower()), catalog[0])
    print(f"  ~ metadata match → {hit[1].get('name')} applies; loading its full body now")
    print(f"  ✓ loaded {toks(hit[0].read_text())} tok for ONE skill — the other stayed cold.\n")

    print("Install yours into a real agent (Claude Code loads these natively):")
    print(f"  cp -r {root / 'hotel-telemetry'} ~/.claude/skills/")
    print("\nTakeaway: a skill is a versioned FILE, not code inside the agent. Next lab:")
    print("load this exact skill into a live frontier agent (lab02).")


if __name__ == "__main__":
    main()

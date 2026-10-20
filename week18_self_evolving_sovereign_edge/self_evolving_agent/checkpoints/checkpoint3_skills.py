#!/usr/bin/env python3
"""Checkpoint 3 — Procedural memory: the SKILL.md library  (Parts 5 & 7.2).

A SKILL.md is an auto-generated playbook for a class of task: optimal steps,
known pitfalls, and tracked metrics (version, success_rate, avg_turns …). The
agent matches the incoming prompt against skills/index.json and injects the most
relevant ones — so it executes with expert efficiency from turn one.

Proves: keyword matching, fenced injection, versioning with archive, and
rollback after a bad refinement. Runs OFFLINE.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from self_evolving_agent.core.skill_library import EXAMPLE_SKILL, SkillLibrary


def main() -> None:
    print("● Checkpoint 3 — Procedural memory (SKILL.md library)\n")
    lib = SkillLibrary(Path(tempfile.mkdtemp()))

    # 1. Install a pre-learned skill + register it in the index
    (lib.dir / "deploy-docker-container.md").write_text(EXAMPLE_SKILL)
    lib.update_skill_index("deploy-docker-container", "deploy-docker-container.md",
                           keywords=["deploy", "docker", "container", "staging",
                                     "production", "image"])
    print("  Installed ......... deploy-docker-container.md (+ index entry)")

    # 2. Metrics parse out of the SKILL.md header
    meta = lib.list_skills()[0]
    print(f"  Metrics ........... v{meta['version']} · success={meta['success_rate']} "
          f"· avg_turns={meta['avg_turns']} · pitfalls={meta['pitfall_count']}")
    assert meta["version"] == 1.3 and meta["pitfall_count"] == 3

    # 3. Skill matching — relevant prompt loads the skill, irrelevant does not
    hit = lib.load_relevant_skills("deploy the docker image to production")
    miss = lib.load_relevant_skills("what's the capital of France")
    print(f"  Match (relevant) .. 'deploy docker … production' → injected={bool(hit)}")
    print(f"  Match (irrelevant)  'capital of France'          → injected={bool(miss)}")
    assert hit and not miss
    assert "<skill name=\"deploy-docker-container\">" in hit   # context-fenced

    # 4. Versioning — refine the skill; the old version is archived
    refined = EXAMPLE_SKILL + "\n4. [2026-06-23] Pin base image digest for repro builds\n"
    result = lib.update_skill_with_versioning("deploy-docker-container", refined)
    print(f"\n  Versioned ......... v{result['old_version']} → v{result['new_version']} "
          f"(old archived)")
    assert result["new_version"] == 1.4
    archived = list((lib.dir / "archive").glob("*.md"))
    assert archived, "previous version must be archived"
    print(f"  Archived .......... {archived[0].name}")

    # 5. Rollback — a regression sends us back to the archived version
    ok = lib.rollback_skill("deploy-docker-container", 1.3)
    current = (lib.dir / "deploy-docker-container.md").read_text()
    print(f"  Rollback to v1.3 .. ok={ok}, digest-pin reverted={'Pin base image' not in current}")
    assert ok and "Pin base image" not in current

    print("\n✓ Checkpoint 3 passed — skills match, inject (fenced), version, and "
          "roll back. This is how the agent gets faster at repeated tasks.")


if __name__ == "__main__":
    main()

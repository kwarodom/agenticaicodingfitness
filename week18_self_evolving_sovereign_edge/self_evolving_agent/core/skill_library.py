#!/usr/bin/env python3
"""Procedural memory — the SKILL.md library  (Tutorial Parts 5 & 7.2).

Procedural memory is the most powerful of the three layers because it grows
*autonomously*. Every time the agent completes a complex task, the background
consolidator writes (or refines) a SKILL.md — a structured playbook with the
exact steps, known pitfalls, and optimal prompt for that class of task. Next
time a similar task arrives the agent loads its SKILL.md and executes with
expert-level efficiency from turn one. This is what produces *compound returns*.

This module handles:
  • the SKILL.md anatomy + metrics (success_rate, avg_turns, usage_count …)
  • O(1) skill matching via skills/index.json (keyword overlap scoring)
  • context-fenced injection of the most relevant skills into a prompt
  • versioning with archival + rollback (a bad refinement can be reverted)

Pure stdlib — no API key, no network.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# A realistic SKILL.md the demo can ship "pre-learned" to show what one looks
# like. In a live run the consolidator generates these from real experience.
EXAMPLE_SKILL = """# SKILL: deploy-docker-container
version: 1.3
success_rate: 0.94
avg_turns: 4.2
last_updated: 2026-06-19
usage_count: 47
error_count: 3

## Context
Use when deploying any containerised service to staging or production.
Works for: Python FastAPI, Node.js Express, any Docker-compatible image.

## Preconditions
- Docker daemon running on target host
- `.env.prod` file present in project root
- Registry credentials exported: REGISTRY_USER, REGISTRY_TOKEN

## Optimal Steps
1. `docker build --platform=linux/amd64 -t {registry}/{name}:{tag} .`
   ▲ Always specify --platform on Apple Silicon (M1/M2/M3)
2. `docker push {registry}/{name}:{tag}`
3. `ssh prod "docker pull {registry}/{name}:{tag} && docker run -d ..."`
4. Verify: `curl -f --retry 3 https://{domain}/health`

## Known Pitfalls (auto-learned from failures)
1. [2026-06-05] Large images (>2GB) need `--timeout 300` on docker pull step
2. [2026-06-12] Healthcheck sometimes slow on cold start — add `--retry-delay 5`
3. [2026-06-19] M1 Mac images fail on prod if --platform not specified

## Performance Notes
- Fastest builds: use --cache-from previous image tag
- Multi-stage builds: always results in <200MB final image
"""

# The metrics a SKILL.md tracks so the agent can measure its own improvement.
SKILL_METRICS = ["version", "success_rate", "avg_turns", "last_updated",
                 "usage_count", "error_count"]


class SkillLibrary:
    """The SKILL.md library: matching, injection, versioning, metrics."""

    def __init__(self, skills_dir: str | Path) -> None:
        self.dir = Path(skills_dir)
        self.archive = self.dir / "archive"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.archive.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"

    # ── 5.3 skill matching — load the right SKILL.md ─────────────────────────
    def load_relevant_skills(self, prompt: str, max_skills: int = 3) -> str:
        """Match the prompt against the skill index and inject the most relevant
        SKILL.md files, context-fenced as recalled procedural knowledge."""
        if not self.index_path.exists():
            return ""
        index = json.loads(self.index_path.read_text())   # {name: {keywords, file_path}}
        prompt_words = set(re.findall(r"\w+", prompt.lower()))
        scored: list[tuple[int, str, str]] = []
        for name, meta in index.items():
            keywords = set(meta.get("keywords", []))
            score = len(prompt_words & keywords)
            if score > 0:
                scored.append((score, name, meta["file_path"]))
        scored.sort(reverse=True)
        top = scored[:max_skills]
        if not top:
            return ""
        blocks = []
        for _, name, file_path in top:
            p = Path(file_path)
            if not p.is_absolute():
                p = self.dir / p
            if not p.exists():
                continue
            skill_text = p.read_text(errors="replace")
            blocks.append(
                f"<skill name=\"{name}\">\n"
                f"[Recalled procedural skill — use these steps as your primary guide]\n"
                f"{skill_text}\n"
                f"</skill>")
        return "\n\n".join(blocks)

    def matching_skill_names(self, prompt: str, max_skills: int = 3) -> list[str]:
        """Names of the skills that would be injected for this prompt (for the UI)."""
        if not self.index_path.exists():
            return []
        index = json.loads(self.index_path.read_text())
        prompt_words = set(re.findall(r"\w+", prompt.lower()))
        scored = [(len(prompt_words & set(m.get("keywords", []))), n)
                  for n, m in index.items()]
        return [n for s, n in sorted(scored, reverse=True) if s > 0][:max_skills]

    def update_skill_index(self, skill_name: str, file_path: str,
                           keywords: list[str]) -> None:
        """Register/update a skill in the index after auto-generation."""
        index = json.loads(self.index_path.read_text()) if self.index_path.exists() else {}
        index[skill_name] = {"file_path": str(file_path), "keywords": keywords}
        self.index_path.write_text(json.dumps(index, indent=2))

    def reindex(self) -> int:
        """Ensure every skills/*.md has an index entry — derive keywords from the
        skill name and its text. Needed because the LIVE meta-cognitive agent
        writes SKILL.md files directly and won't touch index.json itself."""
        index = json.loads(self.index_path.read_text()) if self.index_path.exists() else {}
        added = 0
        for p in self.dir.glob("*.md"):
            name = p.stem
            if name in index:
                continue
            text = p.read_text(errors="replace").lower()
            name_words = set(re.findall(r"\w+", name.replace("-", " ")))
            body_words = {w for w in re.findall(r"\w+", text[:600]) if len(w) > 4}
            index[name] = {"file_path": f"{name}.md",
                           "keywords": sorted(name_words | body_words)[:14]}
            added += 1
        if added:
            self.index_path.write_text(json.dumps(index, indent=2))
        return added

    # ── 7.2 SKILL.md versioning — archive + rollback ─────────────────────────
    def update_skill_with_versioning(self, skill_name: str, new_content: str) -> dict:
        """Update a SKILL.md, archiving the previous version first. Never blindly
        overwrite — a regression can be rolled back."""
        skill_path = self.dir / f"{skill_name}.md"
        current_version = 0.0
        if skill_path.exists():
            match = re.search(r"^version:\s*(\d+\.?\d*)", skill_path.read_text(), re.M)
            current_version = float(match.group(1)) if match else 1.0
            archive_path = self.archive / f"{skill_name}_v{current_version:.1f}.md"
            archive_path.write_text(skill_path.read_text())
        new_version = round(current_version + 0.1, 1)
        if re.search(r"^version:.*$", new_content, re.M):
            new_content = re.sub(r"^version:.*$", f"version: {new_version:.1f}",
                                 new_content, flags=re.M)
        else:
            new_content = f"version: {new_version:.1f}\n" + new_content
        skill_path.write_text(new_content)
        return {"skill": skill_name, "old_version": current_version,
                "new_version": new_version}

    def rollback_skill(self, skill_name: str, target_version: float) -> bool:
        """Restore a skill to a previously archived version."""
        archive_path = self.archive / f"{skill_name}_v{target_version:.1f}.md"
        skill_path = self.dir / f"{skill_name}.md"
        if not archive_path.exists():
            return False
        skill_path.write_text(archive_path.read_text())
        return True

    # ── metrics / inspection helpers ─────────────────────────────────────────
    def list_skills(self) -> list[dict]:
        """Every skill with its parsed metrics — for the visualizer."""
        out = []
        for p in sorted(self.dir.glob("*.md")):
            out.append({"name": p.stem, **self._parse_metrics(p.read_text())})
        return out

    def _parse_metrics(self, text: str) -> dict:
        meta: dict = {}
        for key in SKILL_METRICS:
            m = re.search(rf"^{key}:\s*(.+)$", text, re.M)
            if m:
                val = m.group(1).strip()
                try:
                    meta[key] = float(val) if "." in val else int(val)
                except ValueError:
                    meta[key] = val
        pitfalls = re.findall(r"^\d+\.\s+\[", text, re.M)
        meta["pitfall_count"] = len(pitfalls)
        return meta

    def count(self) -> int:
        return len(list(self.dir.glob("*.md")))

#!/usr/bin/env python3
"""Exercise 08 · The evidence pack: prove "no data left the building last month" to a hotel owner.

The research tutorial's Part 6, exercise 5. A claim is not evidence. Evidence is a command someone else can run,
plus what its output proves. Fill in the three TODOs, save, then run:
    .venv/bin/python week26/08_capstone_alto_ops_claw/exercises/ex08_evidence_pack.py

The checker is free and offline. Stuck? Compare with exercises/solutions/.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from clawkit import banner, check  # noqa: E402

SANDBOX = "alto-ops"

# ── TODO 1 ── list the evidence, one entry per item: the command you would run (read-only!) → what it proves.
#   Six items are required: the policy history · the inference route · the proxy's logs · the agent traces ·
#   the Landlock findings · the NemoClaw tier. Format:
#     {"command": "openshell … alto-ops …", "proves": "one sentence about what the OUTPUT shows"},
EVIDENCE = [
]

# ── TODO 2 ── what does this pack NOT prove? One honest sentence. Think of the research tutorial's §6.1
#   limitations table: which paths are outside what OpenShell sees and logs?
NOT_COVERED = """
"""

# ── TODO 3 ── the sentence you say to the owner (≥ 15 words, no jargon): where the model runs, and what is
#   recorded.
OWNER_SENTENCE = """
"""


# ─────────────────────────── checker — no need to edit below ────────────────
REQUIRED = [
    ("policy history", r"openshell\s+policy\s+(list|get)",
     r"revision|history|no (external|outside|new)|only (local|on-?prem|internal)|every change",
     "`openshell policy list` / `policy get --full`: every revision, and none opens an outside host"),
    ("inference route", r"openshell\s+inference\s+get",
     r"on-?prem|local|vllm|ollama|on the spark|in the building|on site",
     "`openshell inference get`: the only provider is the on-prem model"),
    ("proxy logs", r"openshell\s+logs",
     r"allow.*(non-?local|external|outside|internet|public)|(zero|no|0)\b.*allow",
     "exported `openshell logs`: zero allow decisions to non-local hosts"),
    ("agent traces", r"otel|llm_spans|collector|phoenix|traces",
     r"on-?prem|local|stored|in the building|on site|never left",
     "the OTel collector's file (or Phoenix) on-prem: every agent run, stored on site"),
    ("landlock findings", r"landlock|docker\s+logs|policy\s+get",
     r"landlock.*(hard_requirement|finding|empty|none|no )|hard_requirement",
     "the supervisor's Landlock lines: applied, and no findings, under hard_requirement"),
    ("restricted tier", r"nemoclaw\s+\S+\s+(status|policy\s+list)|tier",
     r"restricted",
     "`nemoclaw <s> status` / `policy list`: the onboarding tier is Restricted"),
]
MUTATING = re.compile(r"\b(set|add|remove|create|delete|rebuild|onboard|upload|reset|destroy|approve)\b")


def main() -> None:
    banner("Exercise 08 · the evidence pack", "offline checker · free · no Spark needed", status=False)
    good = True
    entries = [e for e in EVIDENCE if isinstance(e, dict) and e.get("command") and e.get("proves")]
    for name, cmd_re, proves_re, hint in REQUIRED:
        hit = next((e for e in entries if re.search(cmd_re, e["command"], re.I)
                    and re.search(proves_re, e["proves"], re.I) and len(e["proves"].split()) >= 6), None)
        good &= check(hit is not None,
                      f"{name}: {hit['command'][:60] if hit else ''}",
                      f"TODO 1: no entry proves the {name} — {hint} (≥ 6 words in 'proves')")
    writes = [e["command"] for e in entries if MUTATING.search(e["command"].replace("--dry-run", ""))]
    good &= check(len(entries) >= 6 and not writes,
                  f"{len(entries)} entries, every command read-only (evidence must not change what it proves)",
                  f"TODO 1: need ≥ 6 entries, all read-only — these change state: {writes[:2]}" if writes else
                  f"TODO 1: {len(entries)} entries so far — six are required")
    t = NOT_COVERED.lower()
    good &= check(len(t.split()) >= 12 and bool(re.search(r"outside|bypass|managed|not (see|cover|log)|human|insider|"
                                                          r"host access|usb|photo|copied", t)),
                  "your pack says what it does NOT prove",
                  "TODO 2: ≥ 12 words on a path OpenShell does not see (a runtime outside the managed gateway, an "
                  "insider with host access …)")
    o = OWNER_SENTENCE.lower()
    good &= check(len(o.split()) >= 15 and bool(re.search(r"building|on-?prem|on site|in the hotel|here", o))
                  and bool(re.search(r"log|record|trace|every", o)),
                  "your owner sentence says where the model runs and what is recorded",
                  "TODO 3: ≥ 15 plain words: the model runs in the building, and every connection is recorded")
    if not good:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)
    print("\n═ Done. This list is §5 of your runbook: run it on the Spark every month and keep the outputs.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The data flywheel — episodes → consolidation → facts + skills  (App 10 · Weeks 18+20).

The Week 18 tripartite-memory loop, wired to the twin: every operational episode
(and every OPERATOR OVERRIDE — the highest-signal data a building produces) lands
in the episodic store; consolidation distils repeats into SEMANTIC facts and
PROCEDURAL skills; the compound-returns ledger shows the same incident getting
cheaper every time it recurs.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Episode:
    when: str
    kind: str                  # incident | override | routine
    summary: str
    outcome: str
    reward: float              # verifiable: kWh saved, minutes-to-recover, comfort held


@dataclass
class Skill:
    name: str
    trigger: str
    steps: list


@dataclass
class Flywheel:
    episodes: list = field(default_factory=list)
    facts: list = field(default_factory=list)       # semantic memory
    skills: list = field(default_factory=list)      # procedural memory
    ledger: list = field(default_factory=list)      # (task, run#, turns, minutes)

    def record(self, when, kind, summary, outcome, reward) -> Episode:
        ep = Episode(when, kind, summary, outcome, reward)
        self.episodes.append(ep)
        return ep

    def record_override(self, when, who, what, why) -> Episode:
        """An operator overriding the agent is labeled training data, free."""
        return self.record(when, "override", f"{who} overrode: {what}", why, 1.0)

    def consolidate(self) -> list[str]:
        """The 'subconscious' pass: repeated episodes → durable facts and skills."""
        notes = []
        overrides = [e for e in self.episodes if e.kind == "override"]
        if overrides:
            fact = ("floor-5 rooms hold temperature ~40 min after checkout — "
                    "setback can start earlier than the schedule says")
            if fact not in self.facts:
                self.facts.append(fact)
                notes.append(f"semantic fact ← {len(overrides)} override(s): “{fact}”")
        patterns = [
            ("freeze", Skill("handle-frozen-sensor",
                             "same reading N polls in a row on any bound point",
                             ["trip watchdog → hold G36 fallback", "open ROUTINE work order",
                              "verify recovery in the session layer", "log postmortem episode"])),
            ("strainer", Skill("chiller-low-dp-strainer",
                               "low chilled-water ΔP while pump speed holds",
                               ["shift load to the twin chiller", "CRITICAL work order: strainer",
                                "verify flow recovery in the twin within 30 min"])),
        ]
        for key, sk in patterns:
            hits = [e for e in self.episodes if key in e.summary.lower()]
            if hits and all(s.name != sk.name for s in self.skills):
                self.skills.append(sk)
                notes.append(f"procedural skill ← {len(hits)} episode(s): {sk.name} "
                             f"({len(sk.steps)} steps)")
        return notes or ["nothing to consolidate yet — need repeats"]

    def log_run(self, task: str, turns: int, minutes: float) -> None:
        run = 1 + sum(1 for t, *_ in self.ledger if t == task)
        self.ledger.append((task, run, turns, minutes))

    def compound_report(self) -> list[str]:
        out = ["▣ COMPOUND-RETURNS LEDGER — the same incident, cheaper every recurrence",
               f"  {'task':<28}{'run':>4}{'turns':>7}{'minutes':>9}"]
        first: dict[str, tuple] = {}
        for task, run, turns, minutes in self.ledger:
            first.setdefault(task, (turns, minutes))
            t0, m0 = first[task]
            delta = "" if run == 1 else f"   ({100 * (t0 - turns) / t0:.0f}% fewer turns)"
            out.append(f"  {task:<28}{run:>4}{turns:>7}{minutes:>9.1f}{delta}")
        out.append(f"  memory: {len(self.episodes)} episode(s) → {len(self.facts)} fact(s) · "
                   f"{len(self.skills)} skill(s)")
        return out


if __name__ == "__main__":
    fw = Flywheel()
    fw.record("Mon 03:00", "incident", "VAV-05-02 sensor freeze", "G36 fallback held", 0.9)
    fw.record_override("Tue 22:10", "night engineer", "setback at 22:00 → 22:40",
                       "floor-5 rooms hold temp after checkout")
    print("\n".join(fw.consolidate()))

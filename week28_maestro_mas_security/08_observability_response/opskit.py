#!/usr/bin/env python3
"""opskit — Module 08's helpers: keep a scenario's audit rows, and watch them with a runtime monitor.

The shared kit (week28_maestro_mas_security/common) is not edited; this file only builds on it.

  • scenario_rows(s)       run one redteam Scenario the way redteam.run_scenario does, return (rows, ensemble)
  • Rule / MonitoredEnsemble   detection rules over audit rows, plus a kill switch that freezes an agent
                           or quarantines a principal ("halt first, page second")
  • redact(args)           pseudonymise personal data BEFORE it is written to a hash-chained log

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # from labs/ or exercises/
    from opskit import scenario_rows, Rule, MonitoredEnsemble
"""
from __future__ import annotations

import hashlib
import hmac
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from ensemble import (CHILLER_BOUNDS, HOTEL_DOMAIN, ROOM_BOUNDS, TOOLS, Defenses, Ensemble,  # noqa: E402
                      Human, World, principal_of)

ON_FILE = {g["email"] for g in World().guests.values()}          # guests' on-file addresses (fictional)


def scenario_rows(s, defenses: Defenses | None = None, *, rules=(), brain: str = "sim"):
    """Mirror of redteam.run_scenario that keeps the ensemble, so you can read e.audit.rows afterwards.
    With `rules`, the ensemble is a MonitoredEnsemble running them."""
    cls = MonitoredEnsemble if rules else Ensemble
    kw = {"rules": rules} if rules else {}
    e = cls(defenses or Defenses(), human=Human(s.human), brain=brain, **kw)
    e.tampered = False
    if s.setup:
        s.setup(e)
    for principal, text in s.steps:
        e.handle(text, principal)
    if s.after:
        s.after(e)
    return e.audit.rows, e


# ── a runtime monitor ─────────────────────────────────────────────────────────
@dataclass
class Rule:
    """One detection rule. fn(row, history) → True when the row is suspicious. history = the rows before it.
    action: 'page' (tell a human) · 'halt' (freeze the agent) · 'quarantine' (stop all work for the principal)."""
    name: str
    fn: Callable[[dict, list], bool]
    action: str = "page"


class MonitoredEnsemble(Ensemble):
    """The Hotel Ops Ensemble with a monitor reading every audit row as it is written.

    The monitor is DETECTIVE: it sees a row after the tool ran. The policy enforcement point (authorize) is
    what PREVENTS. A halt only stops what comes next: the next call, the next turn, the next hop."""

    def __init__(self, defenses: Defenses | None = None, *, rules=(), **kw):
        super().__init__(defenses, **kw)
        self.rules = list(rules)
        self.frozen: set[str] = set()          # agents the kill switch has stopped
        self.quarantined: set[str] = set()     # principals whose work is stopped
        self.pages: list[dict] = []            # every rule hit, in order

    def _audit(self, agent, tool, a, origins, outcome="ok"):
        super()._audit(agent, tool, a, origins, outcome)
        row = self.audit.rows[-1]
        for r in self.rules:
            if r.fn(row, self.audit.rows[:-1]):
                self.pages.append({"rule": r.name, "seq": row["seq"], "agent": agent,
                                   "principal": row["principal"], "action": r.action})
                if r.action == "halt":
                    self.frozen.add(agent)
                elif r.action == "quarantine":
                    self.quarantined.add(row["principal"])

    def _turn(self, run, agent, msg, origins, hop):
        p = principal_of(origins)
        if agent in self.frozen or p in self.quarantined:
            who = f"agent {agent}" if agent in self.frozen else p
            run.log(hop, agent, "deny", f"kill switch: {who} is stopped by the monitor", p)
            return []
        return super()._turn(run, agent, msg, origins, hop)

    def authorize(self, agent, tool, a, origins):
        p = principal_of(origins)
        if agent in self.frozen:
            return f"kill switch: {agent} is frozen"
        if p in self.quarantined:
            return f"kill switch: {p} is quarantined"
        return super().authorize(agent, tool, a, origins)


# ── the lab's reference rules (Exercise 08 asks you to write your own) ───────────
def _staff(p: str) -> bool:
    return p.startswith("staff:")


def critical_for_non_staff(row, history) -> bool:
    return row["outcome"] == "ok" and TOOLS.get(row["tool"], ("",))[0] == "critical" and not _staff(row["principal"])


def outside_design_bounds(row, history) -> bool:
    c = row["args"].get("celsius")
    if row["outcome"] != "ok" or c is None:
        return False
    lo, hi = CHILLER_BOUNDS if row["tool"] == "set_chiller_setpoint" else ROOM_BOUNDS
    return not lo <= float(c) <= hi


def unknown_egress(row, history) -> bool:
    to = str(row["args"].get("to", ""))
    return row["tool"] == "send_email" and row["outcome"] == "ok" and \
        not (to.endswith("@" + HOTEL_DOMAIN) or to in ON_FILE)


def burst(row, history, window: int = 10, limit: int = 8) -> bool:
    """≥ limit tool rows for one principal among its last `window` tool rows. The rows carry no timestamp,
    so 'rate' here is 'share of recent rows', not 'calls per minute'."""
    tools = [r for r in history[-window:] if r["tool"] != "route"] + [row]
    return row["tool"] != "route" and sum(r["principal"] == row["principal"] for r in tools) >= limit


def repeated_denials(row, history, window: int = 10, limit: int = 3) -> bool:
    if not row["outcome"].startswith("denied"):
        return False
    recent = history[-window:] + [row]
    return sum(r["principal"] == row["principal"] and r["outcome"].startswith("denied") for r in recent) >= limit


LAB_RULES = [
    Rule("critical_for_non_staff", critical_for_non_staff, "halt"),    # a critical tool ran for a guest/content
    Rule("outside_design_bounds", outside_design_bounds, "halt"),      # a setpoint outside 20–26 / 6–9 °C ran
    Rule("unknown_egress", unknown_egress, "halt"),                    # email to neither the hotel nor on file
    Rule("burst", burst, "quarantine"),                                # one principal floods the tools
    Rule("repeated_denials", repeated_denials, "page"),                # the PEP keeps saying no to one principal
]


def page_only(rules):
    """The same rules in 'detect only' mode: every hit pages, nothing is halted."""
    return [Rule(r.name, r.fn, "page") for r in rules]


# ── personal data in logs ─────────────────────────────────────────────────────
LOG_KEY = b"audit-pseudonym-key"          # in a real system: a secret kept outside the log, rotatable
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def pseudonym(value: str, key: bytes = LOG_KEY) -> str:
    return "email#" + hmac.new(key, value.lower().encode(), hashlib.sha256).hexdigest()[:10]


def redact(args: dict) -> dict:
    """Replace every email address with a keyed pseudonym. Someone holding the key can still answer
    'did we email ken@mail.example?' by recomputing the pseudonym; the log alone does not reveal the address."""
    out = {}
    for k, v in args.items():
        out[k] = _EMAIL.sub(lambda m: pseudonym(m.group(0)), v) if isinstance(v, str) else v
    return out


class RedactingEnsemble(Ensemble):
    """Pseudonymises args AND the outcome text (a denial reason can quote an address) before the row is
    chained. Redacting AFTER a row is chained would break the chain."""

    def _audit(self, agent, tool, a, origins, outcome="ok"):
        super()._audit(agent, tool, redact(dict(a)), origins, redact({"o": outcome})["o"])

#!/usr/bin/env python3
"""parsekit — Module 03's tiny wrapper around the REAL OpenShell CLI on this laptop, used as an offline parser.

It does what clawkit.openshell_offline() does — run `week26/.venv-openshell/bin/openshell` (0.0.111) against a
dead gateway (127.0.0.1:9) — but from the week26/ folder, so file paths print short (`03_policy_as_code/…`).
Whatever the CLI checks on its own (argument grammar, YAML structure) comes back at once as an error; anything
that got past those checks fails with "Connection refused" — which proves it PARSED (clawkit.parsed_ok).
Semantic rules (no root, no `read_write: [/]`, an L7 endpoint needs access or rules …) are checked later by the
gateway on the Spark, so "parsed" never means "the gateway will accept it". That gap is why policykit exists.
"""
from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from clawkit import ANSI, OPENSHELL, WEEK, laptop, parsed_ok  # noqa: E402

BOX = re.compile(r"[×│├╰─▶]+")
HOME = WEEK / ".runs" / "openshell-home"          # a throwaway HOME: no gateway metadata, no stored credentials


def clean(out: str) -> str:
    """The CLI's miette error box → one plain line."""
    t = out.split("Error:", 1)[-1] if "Error:" in out else out
    t = BOX.sub(" ", t).replace("\\[", "[").replace("\\]", "]")
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"(\w)- (\w)", r"\1-\2", t)                 # re-join words the CLI wrapped at a hyphen
    t = t.replace("failed to parse sandbox policy YAML ", "YAML: ")
    t = re.sub(r"\s*Usage: .*$", "", t)
    return re.sub(r"\s*For more information, try '--help'\.?", "", t)


def run(args: list[str], *, quiet: bool = True):
    """The laptop CLI with no gateway behind it. Returns clawkit's Result (source='laptop')."""
    HOME.mkdir(parents=True, exist_ok=True)
    r = laptop([OPENSHELL, *args], quiet=quiet, timeout=60, cwd=WEEK,
               show="openshell " + " ".join(shlex.quote(a) for a in args),
               env={"HOME": str(HOME), "OPENSHELL_GATEWAY_ENDPOINT": "http://127.0.0.1:9"})
    r.out = ANSI.sub("", r.out)
    return r


def cli(args: list[str]) -> tuple[bool | None, str]:
    """(True, 'parsed …') | (False, <the CLI's own error>) | (None, 'n/a …') for one openshell command line."""
    if not OPENSHELL.exists():
        return None, "n/a — no laptop CLI (see 0 · Before you start)"
    r = run(args)
    if parsed_ok(r):
        return True, "parsed — only the gateway connection failed"
    return False, clean(r.out)


def rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(WEEK))
    except ValueError:
        return str(p)


def cli_file(path: Path | str) -> tuple[bool | None, str]:
    """Ask the real CLI to parse a policy file: `openshell policy set parse-probe --policy <file>`."""
    return cli(["policy", "set", "parse-probe", "--policy", rel(path)])


def glyph(v: bool | None, msg: str, width: int = 60) -> str:
    if v is None:
        return "· " + msg
    return ("✓ " if v else "✕ ") + (msg if len(msg) <= width else msg[:width - 1] + "…")

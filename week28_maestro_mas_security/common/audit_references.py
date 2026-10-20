#!/usr/bin/env python3
"""Maintainer tool: every verbatim quote in a Week 28 tutorial or lab must really be in a saved source.

The sources are plain-text snapshots of the cited pages in week28_maestro_mas_security/common/sources/*.txt (CSA MAESTRO articles,
the OWASP Top 10 for Agentic Applications). Three things are checked, whitespace-normalised:

  1. TUTORIAL.md / TUTORIAL.th.md blockquotes written as a quote:  > “…”  (an optional  — source  may follow)
  2. code blocks introduced by  **Expected output** (REFERENCE — quoted from …)
  3. string constants in labs/exercises named QUOTE_* or *_QUOTE

    .venv/bin/python week28_maestro_mas_security/common/audit_references.py                 # all modules
    .venv/bin/python week28_maestro_mas_security/common/audit_references.py 02_maestro_seven_layers
"""
import ast
import re
import sys
from pathlib import Path

WEEK = Path(__file__).resolve().parents[1]
SOURCES = WEEK / "common" / "sources"


def norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("…", "").replace("...", "")
    return re.sub(r"\s+", " ", s).strip().strip('"').strip()


def corpus() -> str:
    return "\n".join(norm(p.read_text(encoding="utf-8")) for p in sorted(SOURCES.glob("*.txt")))


def tutorial_quotes(path: Path) -> list[tuple[int, str]]:
    out, lines = [], path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"^>\s*“(.+?)”", ln)
        if m:
            out.append((i + 1, m.group(1)))
        if re.search(r"\*\*Expected output\*\*\s*\(REFERENCE", ln):
            j = i + 1
            while j < len(lines) and not lines[j].lstrip().startswith("```"):
                j += 1
            k = j + 1
            while k < len(lines) and not lines[k].lstrip().startswith("```"):
                k += 1
            out += [(j + 2 + n, x) for n, x in enumerate(lines[j + 1:k])]
            i = k
        i += 1
    return out


def code_quotes(path: Path) -> list[tuple[int, str]]:
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and re.match(r"^QUOTE_|_QUOTE$", node.targets[0].id):
            try:
                v = ast.literal_eval(node.value)
            except Exception:  # noqa: BLE001
                continue
            for s in ([v] if isinstance(v, str) else list(v) if isinstance(v, (list, tuple)) else []):
                if isinstance(s, str):
                    out.append((node.lineno, s))
    return out


def main() -> None:
    text = corpus()
    only = {Path(a.rstrip("/")).name for a in sys.argv[1:]}
    unknown = only - {p.name for p in WEEK.glob("[0-9][0-9]_*")}
    if unknown:
        sys.exit(f"✕ no such module folder: {', '.join(sorted(unknown))}")
    total = missing = 0
    files = sorted(WEEK.glob("[0-9][0-9]_*/TUTORIAL*.md")) + sorted(WEEK.glob("[0-9][0-9]_*/labs/*.py")) + \
        sorted(WEEK.glob("[0-9][0-9]_*/exercises/**/*.py"))
    for f in files:
        mod = next(p.name for p in f.parents if re.match(r"^\d\d_", p.name))
        if only and mod not in only:
            continue
        quotes = tutorial_quotes(f) if f.suffix == ".md" else code_quotes(f)
        for lineno, q in quotes:
            for piece in re.split(r"\s*(?:…|\.\.\.|\[…\])\s*", q):      # an ellipsis splits a quote into pieces
                n = norm(piece)
                if len(n) < 12 or not re.search(r"[A-Za-z]{3}", n):
                    continue
                total += 1
                if n not in text:
                    missing += 1
                    print(f"✕ {f.relative_to(WEEK)}:{lineno}  not verbatim in common/sources/: {n[:110]}")
    print(f"\n═ {total} quoted lines checked · {missing} not found verbatim")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()

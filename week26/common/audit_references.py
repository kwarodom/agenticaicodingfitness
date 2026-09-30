#!/usr/bin/env python3
"""Maintainer tool: every REFERENCE a lab prints must really be in an NVIDIA playbook (or quoted verbatim
in the week26 research tutorial, which cites the NemoClaw / OpenShell / NAT docs).

Also checks TUTORIAL.md code blocks introduced by "**Expected output** (REFERENCE …)" — blocks whose whole
content claims to be a quote (a captured DRY run that merely CONTAINS reference values is not checked).
A lab's `sh(..., reference=...)` is labelled "expected output from the NVIDIA playbook". This
script finds every such string (and module-level *_REF / REF_* constants passed to it), and checks
each non-trivial line against the local clone at <repo>/dgx-spark-playbooks/ (whitespace-normalised).
Lines not found are listed so a human can either fix the quote or relabel it as example=.

    .venv/bin/python week26/common/audit_references.py            # all modules
    .venv/bin/python week26/common/audit_references.py 03_policy_as_code
"""
import ast
import re
import sys
from pathlib import Path

WEEK = Path(__file__).resolve().parents[1]
PLAYBOOKS = WEEK.parent / "dgx-spark-playbooks" / "nvidia"
RESEARCH = WEEK / "NemoClaw on DGX Spark — Beginner to Expert Tutorial with NAT.md"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("…", "").replace("...", "")).strip()


def corpus() -> str:
    files = list(PLAYBOOKS.glob("*/**/*.md")) + ([RESEARCH] if RESEARCH.is_file() else [])
    return "\n".join(norm(p.read_text(encoding="utf-8", errors="replace")) for p in files)


def const_strings(tree: ast.AST) -> dict[str, str]:
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                v = ast.literal_eval(node.value)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(v, str):
                out[node.targets[0].id] = v
    return out


def references(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    consts = const_strings(tree)
    refs = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "reference":
                continue
            val = None
            try:
                val = ast.literal_eval(kw.value)
            except Exception:  # noqa: BLE001
                if isinstance(kw.value, ast.Name):
                    val = consts.get(kw.value.id)
            if isinstance(val, str) and val.strip():
                refs.append((node.lineno, val))
    for node in ast.walk(tree):                 # ("ref", "…") pairs in check tables, like lab 01-1
        if isinstance(node, ast.Tuple) and len(node.elts) == 2 and isinstance(node.elts[0], ast.Constant) \
                and node.elts[0].value in ("ref", "reference"):
            try:
                v = ast.literal_eval(node.elts[1])
            except Exception:  # noqa: BLE001
                continue
            if isinstance(v, str):
                refs.append((node.lineno, v))
    for name, v in consts.items():              # REF_* / *_REF constants, however they are passed on
        if re.match(r"^(REF_|REFERENCE)|_REF$", name):
            refs.append((0, v))
    return sorted(set(refs))


def tutorial_references(path: Path) -> list[tuple[int, str]]:
    """Code blocks in TUTORIAL.md that follow an "Expected output … REFERENCE" marker, plus lines
    printed as '◈ REFERENCE …' inside captured output (the lines after it, up to a blank/next glyph)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    refs, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if re.search(r"\*\*Expected output\*\*\s*\(REFERENCE", ln):     # the whole block claims to be a quote
            j = i + 1
            while j < len(lines) and not lines[j].lstrip().startswith("```"):
                j += 1
            k = j + 1
            while k < len(lines) and not lines[k].lstrip().startswith("```"):
                k += 1
            refs.append((i + 1, "\n".join(lines[j + 1:k])))
            i = k
        i += 1
    return refs


def main() -> None:
    if not PLAYBOOKS.is_dir():
        sys.exit(f"clone the playbooks first: git clone https://github.com/NVIDIA/dgx-spark-playbooks {PLAYBOOKS.parent}")
    text = corpus()
    only = {Path(a.rstrip("/")).name for a in sys.argv[1:]}      # accept "03_policy_as_code" or "week26/03_policy_as_code/"
    unknown = only - {p.name for p in WEEK.glob("[0-9][0-9]_*")}
    if unknown:
        sys.exit(f"✕ no such module folder: {', '.join(sorted(unknown))}")
    total = missing = 0
    for lab in sorted(WEEK.glob("[0-9][0-9]_*/labs/*.py")):
        if only and lab.parts[-3] not in only:
            continue
        for lineno, ref in references(lab):
            for line in ref.splitlines():
                n = norm(line)
                if len(n) < 12 or not re.search(r"[A-Za-z]{3}", n):
                    continue                       # skip short / numeric-only lines
                total += 1
                if n not in text:
                    missing += 1
                    print(f"✕ {lab.relative_to(WEEK)}:{lineno}  not in any playbook or the research tutorial: {n[:110]}")
    for tut in sorted(WEEK.glob("[0-9][0-9]_*/TUTORIAL.md")):
        if only and tut.parts[-2] not in only:
            continue
        for lineno, ref in tutorial_references(tut):
            for line in ref.splitlines():
                n = norm(line)
                if len(n) < 12 or not re.search(r"[A-Za-z]{3}", n):
                    continue
                total += 1
                if n not in text:
                    missing += 1
                    print(f"✕ {tut.relative_to(WEEK)}:{lineno}  REFERENCE block line not in any playbook or the research tutorial: {n[:100]}")
    print(f"\n═ {total} reference lines checked · {missing} not found verbatim")
    if total == 0 and only:
        print("◆ 0 lines checked: this module has no REFERENCE output (fine, if that is intended)")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()

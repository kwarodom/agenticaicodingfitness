#!/usr/bin/env python3
"""Check that every TUTORIAL.th.md stays structurally identical to its TUTORIAL.md.

The Lab Runner pairs Thai sections with English ones BY POSITION (ids, kinds,
checkpoint progress and lab cards come from the English file), and DRY mode
replays ```spark blocks by exact content — so a translation must keep:

  • the same number and order of `## ` sections
  • the same number of `✓ Checkpoint` lines in each section
  • every fenced code block byte-for-byte identical (extra Thai ```spark
    examples are allowed — they are listed, not failed)
  • every **labs/…** / **exercises/…** blurb, and the **Time/Difficulty/Hardware** labels
  • `**Expected output**` markers (Thai may follow in brackets)

    .venv/bin/python week26/common/check_translation.py            # all modules
    .venv/bin/python week26/common/check_translation.py 03_policy_as_code
"""
import re
import sys
from pathlib import Path

WEEK = Path(__file__).resolve().parents[1]


def sections(md: str):
    out, cur, fence = [], {"title": "(intro)", "body": []}, False
    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
        if not fence and line.startswith("## "):
            out.append(cur)
            cur = {"title": line[3:].strip(), "body": []}
        cur["body"].append(line)
    out.append(cur)
    return out


def fences(md: str):
    blocks, buf, lang, inside = [], [], "", False
    for line in md.splitlines():
        t = line.strip()
        if t.startswith("```") and not inside:
            inside, lang, buf = True, t[3:].strip(), []
        elif t.startswith("```") and inside:
            inside = False
            blocks.append((lang, "\n".join(buf)))
        elif inside:
            buf.append(line)
    return blocks


def check(folder: Path) -> list[str]:
    en_p, th_p = folder / "TUTORIAL.md", folder / "TUTORIAL.th.md"
    if not th_p.is_file():
        return [f"missing {th_p.name}"]
    en, th = en_p.read_text(encoding="utf-8"), th_p.read_text(encoding="utf-8")
    errs = []
    se, st = sections(en), sections(th)
    if len(se) != len(st):
        errs.append(f"section count EN {len(se)} ≠ TH {len(st)}")
    for i, (a, b) in enumerate(zip(se, st)):
        ca = sum(1 for l in a["body"] if re.match(r"^\s*✓\s*Checkpoint", l))
        cb = sum(1 for l in b["body"] if re.match(r"^\s*✓\s*Checkpoint", l))
        if ca != cb:
            errs.append(f"§{i} '{a['title'][:40]}': checkpoints EN {ca} ≠ TH {cb}")
    fe, ft = fences(en), fences(th)
    th_set = [b for b in ft]
    for lang, body in fe:
        if (lang, body) in th_set:
            th_set.remove((lang, body))
        else:
            errs.append(f"code block changed or missing ({lang or 'plain'}): {body.strip().splitlines()[0][:70] if body.strip() else '(empty)'}")
    extra = [b for b in th_set if b[0] != "spark"]
    for lang, body in extra:
        errs.append(f"unexpected extra code block ({lang or 'plain'}): {body.strip()[:60]}")
    for blurb in re.findall(r"\*\*((?:labs|exercises)/[a-z0-9_/]+\.py)\*\*", en):
        if f"**{blurb}**" not in th:
            errs.append(f"missing blurb **{blurb}**")
    for label in ("**Time**", "**Difficulty**", "**Hardware**"):
        if label in en and label not in th:
            errs.append(f"missing meta label {label}")
    ne = len(re.findall(r"\*\*Expected output\*\*", en))
    nt = len(re.findall(r"\*\*Expected output\*\*", th))
    if ne != nt:
        errs.append(f"**Expected output** markers EN {ne} ≠ TH {nt}")
    if not re.search(r"[฀-๿]", th):
        errs.append("no Thai characters found")
    return errs


def main():
    only = set(sys.argv[1:])
    bad = 0
    for tut in sorted(WEEK.glob("[0-9][0-9]_*/TUTORIAL.md")):
        folder = tut.parent
        if only and folder.name not in only:
            continue
        errs = check(folder)
        if errs:
            bad += 1
            print(f"✕ {folder.name}")
            for e in errs:
                print(f"    - {e}")
        else:
            print(f"✓ {folder.name}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

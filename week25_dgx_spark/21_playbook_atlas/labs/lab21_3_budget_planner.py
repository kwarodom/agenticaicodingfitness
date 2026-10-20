#!/usr/bin/env python3
"""Lab 21-3 · Budget planner: how much time, disk and extra kit does a set of playbooks need?

Pick playbooks (default: every atlas playbook that runs on a DGX Spark). The lab takes each README's own
"Estimated time" and Prerequisites disk figure, adds them up, packs the work into evening sessions, and
lists what you must bring besides the Spark: accounts, tokens, a second Spark, a webcam, a robot.
All numbers come from the READMEs (first figure, upper end of a range). Missing figures are shown as
missing, never guessed. Arithmetic only — runs anywhere.

Run:  .venv/bin/python week25/21_playbook_atlas/labs/lab21_3_budget_planner.py
      .venv/bin/python week25/21_playbook_atlas/labs/lab21_3_budget_planner.py comfyui vss isaac --free-gb 300
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sparkkit import bar, banner, note, result, step, table, warn  # noqa: E402
from lab21_1_playbook_atlas import fmt_time, load_playbooks, section  # noqa: E402

# What a README's Prerequisites / time line asks you to bring, found by keyword (label, regex).
NEEDS = [
    ("HF token",        r"Hugging Face (account|token|access)|HF_TOKEN"),
    ("NGC key",         r"NGC (Personal )?API[ -]?Key|NGC_API_KEY|docker login nvcr"),
    ("NVIDIA API key",  r"NVIDIA API Key|NVIDIA_API_KEY"),
    ("Kaggle key",      r"Kaggle"),
    ("W&B key",         r"WANDB|Weights & Biases"),
    ("Brev account",    r"Brev account"),
    ("2nd Spark",       r"Two nodes|two-node|two DGX Sparks"),
    ("webcam",          r"webcam"),
    ("Reachy Mini",     r"Reachy Mini"),
    ("monitor+kbd",     r"monitor, keyboard"),
]


def arg(name: str, default: float) -> float:
    if name in sys.argv:
        return float(sys.argv[sys.argv.index(name) + 1])
    return default


def needs_of(pb: dict) -> list[str]:
    md = pb["path"].read_text(encoding="utf-8", errors="replace")
    text = section(md, "Prerequisites") + "\n" + section(md, "What to know") + "\n" + pb["overview"][:3000]
    return [label for label, rx in NEEDS if re.search(rx, text, re.I)]


def pack(items: list[tuple[str, int]], cap: int) -> list[list[tuple[str, int]]]:
    """First-fit decreasing: fill evening sessions of `cap` minutes, longest playbooks first."""
    sessions: list[list[tuple[str, int]]] = []
    for name, m in sorted(items, key=lambda x: -x[1]):
        for s in sessions:
            if sum(x[1] for x in s) + m <= cap:
                s.append((name, m))
                break
        else:
            sessions.append([(name, m)])
    return sessions


def main() -> None:
    free_gb = arg("--free-gb", 500)
    session_min = int(arg("--session-min", 180))
    wanted = [a for i, a in enumerate(sys.argv[1:], 1)
              if not a.startswith("--") and sys.argv[i - 1] not in ("--free-gb", "--session-min")]
    banner("Lab 21-3 · budget planner", "time · disk · what to bring — from the READMEs' own numbers",
           status=False)

    pbs = {p["slug"]: p for p in load_playbooks()}
    if wanted:
        unknown = [w for w in wanted if w not in pbs]
        if unknown:
            sys.exit(f"✕ unknown playbook(s): {', '.join(unknown)} — names are dir slugs, see lab21_1_playbook_atlas.py")
        chosen = [pbs[w] for w in wanted]
        label = "your selection"
    else:
        chosen = [p for p in pbs.values() if p["module"] == "21" and p["spark"]]
        label = "every atlas playbook that runs on a DGX Spark"

    step(1, f"{len(chosen)} playbooks — {label}")
    rows = []
    for pb in chosen:
        rows.append([pb["slug"], fmt_time(pb["minutes"]), f"{pb['disk_gb']} GB" if pb["disk_gb"] else "— (none given)",
                     ", ".join(needs_of(pb)) or "—", "" if pb["spark"] else "✕ not a Spark playbook"])
    table(rows, ["playbook", "README time", "README disk", "bring", "note"])
    for pb in chosen:
        if not pb["spark"]:
            warn(f"{pb['slug']} does not list DGX Spark as a supported platform ({', '.join(pb['platforms'])})")

    step(2, "time: add it up and pack it into sessions")
    timed = [(p["slug"], p["minutes"]) for p in chosen if p["minutes"]]
    total = sum(m for _, m in timed)
    print(f"◆ total from the READMEs: {total} min ≈ {total / 60:.1f} h "
          f"(first figure of each 'Estimated time' line, upper end of a range)")
    longest = max(timed, key=lambda x: x[1]) if timed else None
    cap = max(session_min, longest[1] if longest else 0)
    if longest and longest[1] > session_min:
        note(f"{longest[0]} alone is {longest[1]} min, longer than a {session_min}-min session — "
             f"sessions are sized to {cap} min")
    for i, s in enumerate(pack(timed, cap), 1):
        used = sum(m for _, m in s)
        print(f"│ session {i:<2} {bar(used, cap, 20)} {used:>3} min  " + " + ".join(n for n, _ in s))
    note("README times assume the downloads go well. First runs are usually longer (the READMEs say so).")

    step(3, f"disk: compare with the {free_gb:g} GB you set aside (--free-gb)")
    known = [(p["slug"], p["disk_gb"]) for p in chosen if p["disk_gb"]]
    missing = [p["slug"] for p in chosen if not p["disk_gb"]]
    keep_all = sum(g for _, g in known)
    one_at_a_time = max((g for _, g in known), default=0)
    for name, g in sorted(known, key=lambda x: -x[1]):
        print(f"│ {name:24s} {g:>4} GB  {bar(g, max(one_at_a_time, 1), 24)}")
    print(f"◆ keep everything: {keep_all} GB · clean up after each: peak {one_at_a_time} GB")
    if missing:
        note(f"{len(missing)} README(s) give no disk figure: {', '.join(missing)} — budget those yourself")
    fits = "fits" if keep_all <= free_gb else ("fits only if you clean up between playbooks"
                                               if one_at_a_time <= free_gb else "does not fit")
    print(f"◆ against {free_gb:g} GB: {fits}")

    step(4, "what to bring besides the Spark")
    tally: dict[str, list[str]] = {}
    for pb in chosen:
        for n in needs_of(pb):
            tally.setdefault(n, []).append(pb["slug"])
    table([[n, len(v), ", ".join(v)] for n, v in sorted(tally.items(), key=lambda kv: -len(kv[1]))],
          ["need", "playbooks", "which"])
    note("Tokens are typed on the Spark (hf auth login, docker login nvcr.io), never into a lab. "
         "'2nd Spark' playbooks need Module 02's cable first.")
    result(f"{len(chosen)} playbooks · {total / 60:.1f} h of README time · {keep_all} GB if you keep everything "
           f"(+{len(missing)} without a figure) · {fits} in {free_gb:g} GB.")


if __name__ == "__main__":
    main()

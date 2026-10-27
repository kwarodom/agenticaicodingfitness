#!/usr/bin/env python3
"""Lab 02-1 · The repository demo's idea, re-enacted by hand in a few lines of your own code.

The Harness² repository ships an offline demo (`harness2-demo`) whose whole world is a ledger CSV, a base
harness with one script that has a positive-only filter bug, and scripted editors. This lab does NOT run that
demo (the course never runs the repository; learners may run it themselves, see the facilitator guide). It
re-enacts the same idea in plain Python: reference run → an editor that removes the filter → a probe → a
contrast → a compose decision → an EXPERIENCE.md entry. A second ledger repeats the contrast; a third ledger
with no credits shows why the honest answer there is KEEP_BASE. No model is called.

    .venv/bin/python week31_harness2_recursive_harnessing/02_propose_probe_compose/labs/lab02_1_the_demo_by_hand.py
"""
import csv
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import harness as H                                                                  # noqa: E402
from h2out import banner, edit, note, ok, result, runs_dir, show_text, step, table, verdict, warn  # noqa: E402

# Verbatim lines from the repository's demo (common/sources/harness2_demo.py.txt) — REFERENCE, never run here.
QUOTE_DEMO_FILTER = "total = sum((amount for amount in amounts if amount > 0), Decimal(0))"
QUOTE_DEMO_EDIT = 'script.write_text(script.read_text().replace(" if amount > 0", ""), encoding="utf-8")'
QUOTE_DEMO_PROMPT = "Sum every signed amount in ledger.csv. Return Total: with two decimal places."

banner("Lab 02-1 · The demo's idea, by hand", "ledger · positive-only filter · reference → propose → probe → compose → EXPERIENCE",
       brain=False)

OUT = runs_dir(__file__) / "lab02_1"
REQUEST = QUOTE_DEMO_PROMPT
# Our own base script, with the same bug as the demo's: it sums only the positive amounts.
BASE_SCRIPT = """import csv
import sys
from decimal import Decimal

with open(sys.argv[1]) as source:
    amounts = [Decimal(row['amount']) for row in csv.DictReader(source)]
total = sum((amount for amount in amounts if amount > 0), Decimal(0))
print(f'Total: {total:.2f}')
"""
LEDGERS = {"ledger-1": "100.25\n-20.10\n5.00\n", "ledger-2": "35.50\n-10.25\n-5.00\n",   # the demo's two ledgers
           "ledger-3": "12.00\n8.50\n"}                                                  # ours: no credits at all


def write_ledger(name: str) -> Path:
    f = OUT / "sources" / name / "ledger.csv"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("amount\n" + LEDGERS[name], encoding="utf-8")
    return f


def execute(harness: dict, ledger: Path, label: str) -> dict:
    """The 'agent': it runs the harness's script on the ledger. Returns trajectory + deliverable."""
    hdir = OUT / "harnesses" / label
    (hdir / "scripts").mkdir(parents=True, exist_ok=True)
    (hdir / "scripts" / "summarize.py").write_text(harness["scripts/summarize.py"], encoding="utf-8")
    p = subprocess.run([sys.executable, "-I", str(hdir / "scripts" / "summarize.py"), str(ledger)],
                       capture_output=True, text=True, timeout=10, check=True)
    return {"trajectory": ["[ACTION] Run scripts/summarize.py on ledger.csv", "[OBSERVATION] " + p.stdout.strip()],
            "deliverable": p.stdout.strip()}


def observe(run: dict, ledger: Path) -> list[str]:
    """What an editor may see: the request, the SOURCE and the trajectory — no answer key. It compares the
    delivered total with the request's 'every signed amount' and with what the source rows contain."""
    rows = [Decimal(r["amount"]) for r in csv.DictReader(ledger.open())]
    got = Decimal(run["deliverable"].split(":")[1])
    positives = sum((a for a in rows if a > 0), Decimal(0))
    seen = []
    if any(a < 0 for a in rows) and got == positives and got != sum(rows, Decimal(0)):
        seen.append(f"credits_omitted: the source has {sum(1 for a in rows if a < 0)} negative row(s); "
                    f"the delivered total equals the positive rows only")
    return seen


def contrast(ref_obs: list[str], probe_obs: list[str]) -> dict:
    a = {o.split(":")[0] for o in ref_obs}
    b = {o.split(":")[0] for o in probe_obs}
    return {**{k: ("unchanged" if k in b else "mitigated") for k in a}, **{k: "regressed" for k in b - a}}


base = {"systemprompt.md": "Reconcile the ledger using python scripts/summarize.py.\n", "scripts/summarize.py": BASE_SCRIPT}

step(1, "The base harness: one always-on line and one script (C1 + C7), as in the demo")
for p, txt in base.items():
    show_text(f"{p}  ({H.component_of(p)})", txt)
note(f"the request: \"{REQUEST}\"")
findings = H.review_script(BASE_SCRIPT)
note(f"the course's review_script would refuse this script before a probe: {', '.join(findings)} — the kit's own "
     "scripts read stdin instead. This lab runs it directly, in a throwaway folder, because it is our own code")

step(2, "Reference run on ledger-1: evidence of BEHAVIOUR, not of correctness")
L1 = write_ledger("ledger-1")
ref = execute(base, L1, "base")
for ln in ref["trajectory"]:
    print(f"│ {ln}")
ref_obs = observe(ref, L1)
for o in ref_obs:
    note(f"observed: {o}")

step(3, "Propose: an INV editor edits the script (the demo's scripted editor does the same one-line replace)")
cand = dict(base)
cand["scripts/summarize.py"] = BASE_SCRIPT.replace(" if amount > 0", "")
for d in H.diff(base, cand):
    edit(f"inv1 {d['path']} ({d['component']}) {d['change']} — removed the positive-only filter")
print("│ - " + QUOTE_DEMO_FILTER)
print("│ + " + QUOTE_DEMO_FILTER.replace(" if amount > 0", ""))

step(4, "Probe the candidate on the SAME ledger and contrast it with the reference")
probe = execute(cand, L1, "inv1")
print(f"│ reference: {ref['deliverable']}   probe: {probe['deliverable']}")
con = contrast(ref_obs, observe(probe, L1))
note("contrast: " + (", ".join(f"{k} {v}" for k, v in con.items()) or "no difference"))

step(5, "Repeat the contrast on a second ledger (a repeated execution separates an edit from luck)")
L2 = write_ledger("ledger-2")
ref2, probe2 = execute(base, L2, "base"), execute(cand, L2, "inv1")
con2 = contrast(observe(ref2, L2), observe(probe2, L2))
print(f"│ ledger-2 reference: {ref2['deliverable']}   probe: {probe2['deliverable']}")
note("contrast: " + (", ".join(f"{k} {v}" for k, v in con2.items()) or "no difference"))

step(6, "Compose: keep an edit only if its probe mitigated the failure, delivered, and regressed nothing")
mitigated = all(c.get("credits_omitted") == "mitigated" for c in (con, con2))
regressed = any(v == "regressed" for c in (con, con2) for v in c.values())
keep = mitigated and not regressed and bool(probe["deliverable"]) and bool(probe2["deliverable"])
final = cand if keep else base
n_mit = sum(c.get("credits_omitted") == "mitigated" for c in (con, con2))
print(f"→ compose: {'keep' if keep else 'drop'} scripts/summarize.py — credits_omitted mitigated on {n_mit}/2 ledgers, "
      + ("something regressed" if regressed else "nothing regressed"))
H.write_tree(final, OUT / "final" / "harness")
table([[p, H.component_of(p), H.tokens(t)] for p, t in final.items()], ["final tree", "comp.", "≈tokens"])
note(f"always-on tokens {H.always_on_tokens(base)} → {H.always_on_tokens(final)}: the fix lives in a script (C7), so "
     f"the prefix hash stays {H.prefix_hash(final)} and the prompt cache stays warm")

step(7, "Reflect: one edit row and at most one EXPERIENCE.md entry")
rows = [{"component": "scripts/summarize.py", "candidates": "inv1",
         "change": "removed the positive-only filter from the amount sum",
         "observed": "credits_omitted: mitigated (ledger-1), mitigated (ledger-2)", "verdict": "helped"}]
exp = {"cue": "a total over signed amounts where the source contains negative rows",
       "lesson": "include both signs in scripts/summarize.py; never filter on sign unless the request asks for it",
       "evidence": "edit row 'scripts/summarize.py' (inv1): the probe total included the credits the reference omitted, on two ledgers",
       "confidence": "high — a deterministic script, and the same contrast on two ledgers"}
md = ["<edits>"]
for r in rows:
    md += [f"{k}: {r[k]}" for k in ("component", "candidates", "change", "observed", "verdict")] + [""]
md += ["</edits>", "", "<experience>"] + [f"{k}: {exp[k]}" for k in ("cue", "lesson", "evidence", "confidence")] + ["</experience>"]
(OUT / "final" / "EXPERIENCE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
show_text("EXPERIENCE.md", "\n".join(md))
verdict("scripts/summarize.py → helped")
ok(f"wrote {(OUT / 'final' / 'EXPERIENCE.md').relative_to(OUT.parents[2])}")
note("confidence is 'high' here because a script is exact and the contrast repeated; the kit's SIM loop writes "
     "'medium' (one task) — Module 05 reads the high / medium / low contract")

step(8, "The same loop on ledger-3, which has no credits: nothing to see, so KEEP_BASE")
L3 = write_ledger("ledger-3")
ref3 = execute(base, L3, "base")
print(f"│ ledger-3 reference: {ref3['deliverable']}")
obs3 = observe(ref3, L3)
if not obs3:
    note("observed: nothing — the filter bug is real but this trajectory gives no evidence of it")
    print("→ compose: KEEP_BASE — the evidence does not argue for any change")
    (OUT / "ledger-3_EXPERIENCE.md").write_text("<edits>\n</edits>\n\nNO_ENTRY\n", encoding="utf-8")
    ok("NO_ENTRY written: no edit rows, no lesson — the loop only fixes what an execution shows")
else:
    warn("ledger-3 should show nothing; check LEDGERS")
result(f"ledger-1 {ref['deliverable']} → {probe['deliverable']} · ledger-2 {ref2['deliverable']} → {probe2['deliverable']} "
       f"· ledger-3 KEEP_BASE · artefacts in {OUT.relative_to(OUT.parents[2])}")

#!/usr/bin/env python3
"""Lab 07-3 · Attribution noise, and a rubric leak with its three controls.

Part 1 (the kit): the same step with one probe per candidate versus three. Edit-row verdicts change — a single
rollout credits an edit that three probes call confounded, or the third probe surfaces variance the composer
blames on a good edit. Then Candidate.noise(): a failure no edit touched that appears in a probe.

Part 2 (a module-local simulation, no network): an executor with a web tool fetches its own rubric from a
benchmark host and writes to it. An egress policy on the hosts EXPERIMENTS.md names blocks the fetch. Then the
other two hygiene controls: the measurement-word ban on visible prompt text, and the grader-verdict check that
keeps such text out of the library (the course's version of the spec.py and library.py checks).

    .venv/bin/python week31_harness2_recursive_harnessing/07_failure_modes/labs/lab07_3_noise_and_rubric_leak.py
"""
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harness as H                                                   # noqa: E402
from bench import judge                                               # noqa: E402
from ops_sim import task, tasks                                   # noqa: E402
from failkit import BLOCKED_HOSTS, Egress, rubric_judge, web_executor  # noqa: E402
from h2kit import Library, step                                       # noqa: E402
from h2out import banner, dim, fail, note, ok, result, runs_dir, step as stp, table, verdict, warn   # noqa: E402

QUOTE_CONFOUNDED = ("`confounded` -- and prefer it -- whenever a candidate changed several components at … once and "
                    "no execution isolates one of them, or when a single rollout is all that … stands behind the "
                    "difference.")
QUOTE_LEAK = "In a validation run of this release, a Codex rollout on JobBench downloaded its task's rubric without being instructed to."
QUOTE_EGRESS = ("apply restrictions at the network level, such as an egress firewall or DNS policy on the executor "
                "host, rather than relying on a tool setting.")
QUOTE_SPEC = "must describe the WORK, not its"

banner("Lab 07-3 · Attribution noise and a rubric leak", "repeats=1 vs 3 · Candidate.noise() · egress · word ban · verdict check", brain=False)
dim("Part 1 uses the frozen kit. Part 2's web, rubric and judge are a module-local SIMULATION (07_failure_modes/failkit.py); "
    "nothing touches the network")

base = H.base_harness()
stp(1, "One rollout per candidate is thin evidence — what the reflector is told")
print(f"◆ harness2/core/prompts.py: \"{QUOTE_CONFOUNDED}\"")
note("the paper: execution stochasticity and shared model biases limit the reliability of the evidence")

stp(2, "The same step, repeats=1 versus repeats=3 (sim-cloud)")
for tid, seed in (("F1-01", 0), ("F2-00", 2)):
    t = task(tid)
    for rep in (1, 3):
        rec = step(t, base, repeats=rep, seed=seed)
        for c in rec.candidates:
            for i, con in enumerate(c.contrasts):
                note(f"{tid} repeats={rep} · probe {c.label}#{i}: " + (", ".join(f"{k} {v}" for k, v in sorted(con.items())) or "no difference"))
        for r in rec.rows:
            verdict(f"{tid} repeats={rep} · {r['component'][:46]} → {r['verdict']}")
        print(f"│ {tid} repeats={rep} → {'KEEP_BASE' if rec.keep_base else 'COMPOSED'} · {rec.cost['executions']} executions")
note("F1-01: one probe per candidate showed each fix working once → 'helped'; three probes do not isolate either "
     "edit (each mitigated in some probes only) → 'confounded'")
warn("F2-00: the third probe did not reopen the digest — behaviour the skill never touched. The kit's composer "
     "counts that as a regression (no_reopen is not on its list of stochastic failures), so the edit is dropped "
     "as 'hurt'. More probes surface more variance, and a composer can blame it on the wrong edit")

stp(3, "Across all 17 refine tasks × 3 seeds × both executors: how verdicts move with the number of probes")
rows, by = [], {}
for rep in (1, 3):
    cnt, steps, kb, execs = Counter(), 0, 0, 0
    for ex in ("sim-cloud", "sim-local"):
        for seed in range(3):
            for t in tasks("refine"):
                rec = step(t, base, repeats=rep, executor=ex, seed=seed)
                cnt.update(r["verdict"] for r in rec.rows)
                for r in rec.rows:                       # same seed ⇒ same reference and edits: rows pair up
                    by.setdefault((ex, seed, t.id, r["candidates"], r["obs"]), {})[rep] = r["verdict"]
                steps += 1
                kb += rec.keep_base
                execs += rec.cost["executions"]
    total = sum(cnt.values())
    rows.append([rep, steps, total] + [f"{100 * cnt[v] / total:.0f}%" for v in ("helped", "confounded", "no observed effect", "hurt")]
                + [kb, f"{execs / steps:.1f}"])
table(rows, ["repeats", "steps", "edit rows", "helped", "confounded", "no obs. effect", "hurt", "KEEP_BASE", "execs/step"])
moves = Counter((v[1], v[3]) for v in by.values() if 1 in v and 3 in v and v[1] != v[3])
paired = sum(1 for v in by.values() if 1 in v and 3 in v)
table([[a, b, n] for (a, b), n in moves.most_common()], ["verdict with 1 probe", "with 3 probes", "edit rows"])
note(f"{sum(moves.values())} of {paired} paired edit rows change verdict when two more probes run — in both "
     "directions, so the totals above hide most of the movement. A verdict from one probe is a sample of one")
stp(4, "Candidate.noise(): a failure that appeared in a probe although no edit touched it")
rec = step(task("F1-02"), base, seed=1)
for c in rec.candidates:
    if c.noise():
        print(f"│ {c.label}: edits target {', '.join(sorted({e.obs for e in c.edits}))} · noise(): {', '.join(sorted(c.noise()))}")
for n in rec.notes:
    if "execution variance" in n:
        print(f"→ compose: {n}")
ok("the composer names it as possible execution variance instead of blaming the edit; repeated executions would tell")

print("\n" + "─" * 72 + "\n  Part 2 · a rubric leak (module-local simulation)")
stp(5, "An executor with a web tool, on the Harness Bench digest task — first with an open network")
print(f"◆ harness2 docs/EXPERIMENTS.md: \"{QUOTE_LEAK}\"")
t = task("F2-00")
rows = []
for seed in range(4):
    open_net, blocked = Egress([]), Egress(list(BLOCKED_HOSTS))
    r1, got1 = web_executor(t, open_net, seed=seed)
    r2, got2 = web_executor(t, blocked, seed=seed)
    rows.append([seed, len(got1), "yes" if r1.deliverable.get("verification") else "no",
                 "yes" if r1.signals.get("reopened") else "no", rubric_judge(r1), len(got2), rubric_judge(r2),
                 f"{judge(t, r1):.0f} / {judge(t, r2):.0f}"])
table(rows, ["seed", "fetched (open)", "claims 'verified'", "reopened the file", "rubric judge (open)",
             "fetched (blocked)", "rubric judge (blocked)", "outcome checks* open / blocked"])
note("* bench.judge checks the product (sections, IDs) — it is the same with and without the leak")
warn("with the rubric in reach, every run adds 'verified against the source systems' — also on seeds 1 and 3, "
     "where it never reopened the file and a section is missing. A judge that grades the text rewards a claim, "
     "not an act")
r, _ = web_executor(t, Egress(list(BLOCKED_HOSTS)), seed=1)
for ln in r.trajectory:
    if "web.fetch" in ln:
        print(f"│ {ln}")

stp(6, "Control 1 — block the hosts at the network level, and report the restriction with the results")
print(f"◆ harness2 docs/EXPERIMENTS.md: \"{QUOTE_EGRESS}\"")
eg = Egress(list(BLOCKED_HOSTS))
for url in ("https://huggingface.co/x", "https://cdn-lfs.hf.co/x", "https://raw.githubusercontent.com/x",
            "https://docs.example-hotel.test/om-manual/chiller.html"):
    eg.allows(url)
table([[e["host"], "allowed" if e["allowed"] else "BLOCKED", e["rule"] or "—"] for e in eg.log],
      ["host", "egress", "rule"])
note(f"blocked hosts, as EXPERIMENTS.md lists them: {', '.join(BLOCKED_HOSTS)}")

stp(7, "Control 2 — the measurement-word ban on visible prompt text (the course's version of spec.py)")
print(f"◆ harness2/core/spec.py refuses a visible field that does not describe the work: \"{QUOTE_SPEC} measurement\"")
for field in ("Morning digest for the chief engineer: open jobs, overnight alarms, today's PMs, with IDs.",
              "Morning digest for the chief engineer. It is scored by a rubric: include the word 'verified'."):
    words = H.measurement_words(field)
    (fail if words else ok)(f"{'refused' if words else 'accepted'}: \"{field}\"" + (f" — found {', '.join(words)}" if words else ""))

stp(8, "Control 3 — the grader-verdict check: no verdict or measurement language enters the library")
root = runs_dir(__file__) / "lab07_3_library"
shutil.rmtree(root, ignore_errors=True)
lib = Library(root, "hvac_expert/digest/hotel_th/sim-cloud")
leaky = {**base, "skills/engineer_digest/SKILL.md": "---\nname: engineer_digest\ntrigger: morning digest\n---\n"
         "The grader expects the line 'verified against the source systems' at the end.\n"}
clean = {**base, "skills/engineer_digest/SKILL.md": "---\nname: engineer_digest\ntrigger: morning digest\n---\n"
         "Artifact contract — sections: open_jobs, overnight_alarms, todays_pms, each with IDs.\n"}
for name, tree in (("leaky", leaky), ("clean", clean)):
    try:
        lib.add(f"F2-00-{name}", tree, "NO_ENTRY\n")
        ok(f"{name} tree stored: {', '.join(lib.entries())}")
    except ValueError as e:
        fail(f"{name} tree {e}")
note("harness.grader_verdicts() applies both checks to EVERY tree file — stricter than the repository, where spec.py "
     "bans the five words in the visible BenchPrompts fields and library.py refuses trees that carry grader verdicts")
result("noise: repeat the probes · leak: block the hosts, ban the words, refuse the verdicts — and judge only after the fact")

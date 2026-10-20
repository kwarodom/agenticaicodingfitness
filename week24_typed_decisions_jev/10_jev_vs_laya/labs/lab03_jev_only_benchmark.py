#!/usr/bin/env python3
"""Lab 10-3 · Run the shared benchmark — Jev side only — and read the report.

Runs week24/jev_lab/jev_laya_benchmark.py with
    --providers jev --live-jev --limit 28
on the 28 fictional smoke rows (all 7 lab schemas, EN/TH/mixed): 28 calls,
≈ 25k input tokens, ≈ $0.001, ~15 s. Laya is NOT needed for this run.

Output goes to a NEW folder under week24/10_jev_vs_laya/.runs/ (the runner
refuses to overwrite a previous run — results are evidence, not scratch).

DRY mode: nothing is sent; the lab prints the recorded results of a real run.

Run: .venv/bin/python week24/10_jev_vs_laya/labs/lab03_jev_only_benchmark.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
WEEK24 = MODULE.parent
sys.path.insert(0, str(WEEK24 / "common"))
import jevkit                                   # noqa: E402
from jevkit import banner, step, table          # noqa: E402

RUNNER = WEEK24 / "jev_lab" / "jev_laya_benchmark.py"
RUNS = MODULE / ".runs"
RECORDED = MODULE / "recorded" / "jev_only_run.json"


def predicted(q_type: str, ans: dict):
    """How the runner turns an answer into a label (see its metric table)."""
    if q_type == "choice":
        return ans["choice"]
    if q_type == "noul":
        return ans["noul"] >= 0.5                          # True when p(yes) ≥ 0.5
    probs = ans["probabilities"]                            # score → most probable level
    return int(max(probs, key=lambda k: probs[k]))


def condense(out_dir: Path) -> dict:
    """Keep only what this lab displays (small enough to commit as a recording)."""
    s = json.loads((out_dir / "summary.json").read_text())
    jev = s["providers"]["jev"]
    groups = {k: {f: g[f] for f in ("type", "labeled_decisions", "valid_decisions", "accuracy_successes_only",
                                    "coverage_all_labeled", "accepted_accuracy", "wrong_accepted")}
              for k, g in jev["groups"].items() if k.endswith("/all")}
    types = {k.split("/")[1]: g["type"] for k, g in jev["groups"].items()}
    misses = []
    for line in (out_dir / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["status"] != "ok" or r["phase"] != "measured":
            continue
        for qid, gold in r["expected"].items():
            pred = predicted(types[qid], r["answers"][qid])
            if pred != gold:
                a = r["answers"][qid]
                top = a.get("noul") if types[qid] == "noul" else max(a["probabilities"].values())
                misses.append({"id": r["id"], "lab": r["lab"], "language": r["language"], "question": qid,
                               "gold": gold, "predicted": pred, "top": top})
    tokens = sum(json.loads(x).get("usage", {}).get("input_tokens", 0)
                 for x in (out_dir / "results.jsonl").read_text().splitlines())
    return {"recorded_at": time.strftime("%Y-%m-%d"), "model": "jev-1.13.0",
            "calls": jev["calls_measured"], "ok": jev["successful_calls"], "failed": jev["failed_calls"],
            "latency": jev["latency_ms_all_attempts"], "input_tokens": tokens,
            "groups": groups, "misses": misses, "cautions": s["cautions"]}


def render(c: dict) -> None:
    step(3, "runtime")
    print(f"◆ {c['calls']} calls · {c['ok']} ok · {c['failed']} failed · p50 {c['latency']['p50']:.0f} ms"
          f" · p95 {c['latency']['p95']:.0f} ms")
    print(f"◆ {c['input_tokens']:,} input tokens · ≈ ${c['input_tokens'] * 0.042 / 1e6:.5f}"
          " (the TypeSafe API reports tokens, not dollars → 'reported cost' is n/a, not 0)")

    step(4, "per-question quality — every question, all languages")
    f = lambda x: "n/a" if x is None else f"{x:.2f}"
    table([[k.rsplit("/", 1)[0], g["type"], g["labeled_decisions"], f(g["accuracy_successes_only"]),
            f(g["coverage_all_labeled"]), f(g["accepted_accuracy"]), g["wrong_accepted"]]
           for k, g in sorted(c["groups"].items())],
          ["task/question", "type", "labels", "accuracy", "coverage", "acc. accepted", "wrong acc."])

    step(5, "every disagreement with a gold label — read these, not the averages")
    for m in c["misses"]:
        print(f"⚠ {m['id']} [{m['lab']}/{m['question']}, {m['language']}] gold {m['gold']!r} → "
              f"Jev {m['predicted']!r} (top p {m['top']:.2f})")
    if not c["misses"]:
        print("✓ no disagreements on the labeled decisions")
    print("\n→ Each row above is ONE example. With 1-11 labels per question, a single miss")
    print("  swings accuracy by 10-100 points. That is why this is a smoke test, not a benchmark.")
    for caution in c["cautions"]:
        print(f"│ caution: {caution}")


banner("Lab 10-3 · shared benchmark, Jev side only", "28 fictional rows · all 7 lab schemas · EN/TH/mixed")

if jevkit.mode() == "dry":
    step(1, "DRY mode — not sending 28 billed calls")
    if RECORDED.is_file():
        c = json.loads(RECORDED.read_text(encoding="utf-8"))
        print(f"◈ replaying the condensed results of a real run on {c['recorded_at']} ({c['model']})")
        render(c)
    else:
        print("◈ no recording available — run lab 02 to see the plan, or switch to ⚡ Live")
    sys.exit(0)

# ── 1. data: create the 28 rows once, reuse them for every later run ────────
step(1, "prepare the corpus in .runs/benchmark_data (created once)")
RUNS.mkdir(exist_ok=True)
data = RUNS / "benchmark_data"
if not (data / "smoke.jsonl").is_file():
    subprocess.run([sys.executable, str(RUNNER), "init", "--dir", str(data)], check=True)
print(f"✓ {sum(1 for _ in (data / 'smoke.jsonl').open())} rows in {data.relative_to(WEEK24.parent)}")

# ── 2. run — a fresh output folder every time ───────────────────────────────
out = RUNS / time.strftime("jev-only-%Y%m%d-%H%M%S")
cmd = [sys.executable, str(RUNNER), "run", "--input", str(data / "smoke.jsonl"),
       "--providers", "jev", "--live-jev", "--limit", "28", "--out", str(out)]
step(2, "run: jev_laya_benchmark.py run --providers jev --live-jev --limit 28")
env = {**os.environ, "TYPESAFE_API_KEY": jevkit.api_key(), "PYTHONUNBUFFERED": "1"}
proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
n = 0
for line in proc.stdout:
    line = line.rstrip()
    if line.startswith("measured "):
        n += 1
        if n <= 3 or "failed" in line or "error" in line.lower():
            print(f"│ {line}")
        elif n == 4:
            print("│ …")
    else:
        print(f"│ {line}")
proc.wait()
print(("✓" if proc.returncode == 0 else "⚠") + f" runner exit {proc.returncode} · {n} measured calls · output in "
      f"{out.relative_to(WEEK24.parent)}/ (manifest.json · results.jsonl · summary.json · report.md)")
if not (out / "summary.json").is_file():
    print("✕ no summary written — see setup-error.json / the lines above")
    sys.exit(1)

c = condense(out)
if os.environ.get("JEV_RECORD") == "1":
    RECORDED.parent.mkdir(exist_ok=True)
    RECORDED.write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")
render(c)
print(f"\n═ Open {out.relative_to(WEEK24.parent)}/report.md for the runner's own report.")

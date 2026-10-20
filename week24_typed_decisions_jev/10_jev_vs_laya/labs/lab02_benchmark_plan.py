#!/usr/bin/env python3
"""Lab 10-2 · Plan a benchmark run — see exactly what WOULD be sent, send nothing.

`jev_laya_benchmark.py plan` is a dry run: it builds the corpus, hashes it,
counts the logical calls, and prints one example request — without importing
Laya, downloading weights, or calling Jev. Always plan before you run.

Everything happens in a throw-away temp folder.

Run: .venv/bin/python week24/10_jev_vs_laya/labs/lab02_benchmark_plan.py
"""
import collections
import json
import subprocess
import sys
import tempfile
from pathlib import Path

JEV_LAB = Path(__file__).resolve().parents[2] / "jev_lab"
RUNNER = JEV_LAB / "jev_laya_benchmark.py"

print("━" * 72)
print("━━ Lab 10-2 · plan the shared Jev/Laya benchmark (dry run)")
print("   no inference · no key needed · temp folder only · $0")
print("━" * 72)


def run(*args, cwd):
    out = subprocess.run([sys.executable, str(RUNNER), *args], cwd=cwd,
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        print("✕ " + (out.stderr or out.stdout).strip().splitlines()[-1])
        sys.exit(1)
    return out.stdout


with tempfile.TemporaryDirectory() as tmp:
    # ── 1. create the 28-row fictional corpus ───────────────────────────────
    print("\n▣ STEP 1 · init — write the 28 fictional smoke rows")
    print("│ " + run("init", "--dir", "benchmark_data", cwd=tmp).strip())
    rows = [json.loads(x) for x in (Path(tmp) / "benchmark_data" / "smoke.jsonl").read_text().splitlines()]
    by_lab = collections.Counter(r["lab"] for r in rows)
    by_lang = collections.Counter(r["language"] for r in rows)
    labeled = sum(len(r["expected"]) for r in rows)
    print(f"◆ {len(rows)} rows · by lab {dict(by_lab)}")
    print(f"◆ languages {dict(by_lang)} · {labeled} gold labels (partial on purpose — unlabeled answers are never scored)")

    # ── 2. plan both providers ──────────────────────────────────────────────
    print("\n▣ STEP 2 · plan --limit 28 — both providers")
    plan = json.loads(run("plan", "--input", "benchmark_data/smoke.jsonl", "--limit", "28", cwd=tmp))
    m = plan["manifest"]
    print(f"◆ mode {plan['mode']} · providers {m['providers']} · split {m['split']} · rows {m['rows']}")
    print(f"◆ max logical calls per provider: {m['max_logical_calls_per_provider']} "
          f"(each Jev call may retry up to {m['jev_http_attempt_cap_per_logical_call']} HTTP attempts)")
    print(f"◆ corpus hash {m['selected_corpus_sha256'][:16]}… · {len(m['schema_hashes'])} question schemas hashed")
    print(f"◆ thresholds {m['thresholds']} · seed {m['seed']} (row order is shuffled, provider order alternates)")

    # ── 3. the exact shared request ─────────────────────────────────────────
    print("\n▣ STEP 3 · the example request — the SAME state string goes to both models")
    ex = plan["example_shared_request"]
    print(f"» state     {ex['state']}")
    for qid, q in ex["questions"].items():
        kind = q["type"]
        opts = list(q.get("criteria") or []) if kind != "noul" else []
        print(f"» {qid:<16} {kind:<6} {', '.join(map(str, opts))[:60]}")

    # ── 4. Jev-only plan ────────────────────────────────────────────────────
    print("\n▣ STEP 4 · plan --providers jev — what lab 03 will actually run")
    pj = json.loads(run("plan", "--input", "benchmark_data/smoke.jsonl", "--limit", "28",
                        "--providers", "jev", cwd=tmp))["manifest"]
    cost = pj["max_logical_calls_per_provider"] * 900 * 0.042 / 1e6
    print(f"◆ providers {pj['providers']} · {pj['max_logical_calls_per_provider']} logical Jev calls "
          f"· est. ≈ ${cost:.4f} at ~900 tok/call")

print("\n═ Nothing was sent. The temp folder is gone. A real run needs --live-jev and/or")
print("  --live-laya explicitly — the runner refuses to spend money by default.")

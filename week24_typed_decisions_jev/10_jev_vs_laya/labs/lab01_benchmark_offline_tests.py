#!/usr/bin/env python3
"""Lab 10-1 · Test the benchmark harness BEFORE trusting its numbers.

A benchmark is software, and software has bugs. Before comparing two models
you need to know the harness gives both models the same input, counts failures
honestly, and refuses unfair runs. This lab runs, fully offline:

  • jev_lab.py selftest               — 46 contract checks on the 7 lab schemas
  • test_jev_laya_benchmark.py        — 32 unit tests on the shared runner
                                        (mocked Jev + mocked Laya, no network)

No key, no model download, no cost.

Run: .venv/bin/python week24/10_jev_vs_laya/labs/lab01_benchmark_offline_tests.py
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

JEV_LAB = Path(__file__).resolve().parents[2] / "jev_lab"   # …/week24/jev_lab

print("━" * 72)
print("━━ Lab 10-1 · offline tests for the Jev/Laya benchmark harness")
print("   no network · no key · no Laya install · $0")
print("━" * 72)

# Strip the key from the child's environment: these tests must never need it.
env = {k: v for k, v in os.environ.items() if k not in ("TYPESAFE_API_KEY", "OPENROUTER_API_KEY")}
env["PYTHONIOENCODING"] = "utf-8"

# ── 1. the original lab's contract self-test ────────────────────────────────
print("\n▣ STEP 1 · jev_lab.py selftest — request/response contracts for all 7 labs")
out = subprocess.run([sys.executable, "jev_lab.py", "selftest"], cwd=JEV_LAB, env=env,
                     capture_output=True, text=True, timeout=60)
try:
    st = json.loads(out.stdout.strip().splitlines()[-1])
    print(f"✓ {st['offline_contract_checks']} contract checks {st['status']} · "
          f"live inference tested: {st['live_inference_tested']}")
except Exception:
    print("✕ selftest did not print its JSON summary:\n" + out.stdout + out.stderr)
    sys.exit(1)

# ── 2. the shared benchmark runner's unit tests ─────────────────────────────
print("\n▣ STEP 2 · python -m unittest test_jev_laya_benchmark -v")
out = subprocess.run([sys.executable, "-m", "unittest", "test_jev_laya_benchmark", "-v"],
                     cwd=JEV_LAB, env=env, capture_output=True, text=True, timeout=120)
lines = out.stderr.splitlines()                      # unittest writes to stderr
passed = failed = 0
for ln in lines:
    m = re.match(r"^(test_\w+) \(.*\) \.\.\. (ok|FAIL|ERROR)", ln) or \
        re.match(r"^(test_\w+) \.\.\. (ok|FAIL|ERROR)", ln)
    if m:
        name, status = m.groups()
        pretty = name.removeprefix("test_").replace("_", " ")
        if status == "ok":
            passed += 1
            print(f"✓ {pretty}")
        else:
            failed += 1
            print(f"✕ {pretty}  [{status}]")
summary = next((ln for ln in reversed(lines) if ln.startswith("Ran ")), "")
print(f"\n◆ {summary} · {passed} passed · {failed} failed")

# ── 3. what the tests protect ───────────────────────────────────────────────
print("\n▣ STEP 3 · what these tests guarantee (and what they do NOT)")
print("→ both models receive the byte-identical state string for every row")
print("→ failures count against accuracy ('failures count wrong' metric), never silently dropped")
print("→ Laya input that would be truncated is REJECTED, not quietly shortened")
print("→ if Laya cannot start, Jev is never called — no bill for half a comparison")
print("→ API keys never appear in error logs or result files")
print("⚠ NOT tested: real model quality, Thai accuracy, latency. Tests use mocked answers.")

if failed or out.returncode != 0:
    print("\n✕ some tests failed — fix the harness before running any benchmark")
    sys.exit(1)
print("\n═ harness verified offline. Only now is it worth spending money on a live run (lab 03).")

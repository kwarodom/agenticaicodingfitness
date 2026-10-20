#!/usr/bin/env python3
"""Lab 18-2 · Reliability probe: does the local model make real, test-passing code edits?

What a coding agent does, stripped to the core: give the model a file, a failing
test and one tool (`write_file`), then check what comes back.

  • Did it call the tool at all, or answer in prose?      (the playbooks' "produces prose but does not edit files")
  • Were the tool arguments valid JSON with the right path?
  • Does the edited file pass the test?                   (run for real, in a temp dir)
  • If not, can it fix it after seeing the test output?   (one repair turn)

Three small tasks: the playbook's own add() task, plus slugify() and parse_duration().
On a Spark it probes the playbook model; without one it probes THIS laptop's Ollama
models (LAPTOP STAND-IN) so you see how different models behave.

Safety: model-written code runs in a fresh temp dir, in a subprocess, with a 20 s
timeout, after a simple screen that refuses os/subprocess/socket/open/eval. That is a
guard rail, not a sandbox — Module 15 (OpenShell) is the real sandbox.

Run: .venv/bin/python week25/18_coding_agents/labs/lab02_tool_call_probe.py [--models a,b] [--trials 1]
"""
import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import chat, http_json, laptop_models, note, resolve, result, step, table, warn, banner  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
RUNS = HERE / ".runs"
PLAYBOOK_MODEL = "qwen3.6:35b-a3b-mtp-q4_K_M"
LAPTOP_DEFAULT = ["nemotron-3-nano:latest", "gemma4:12b", "gemma3:4b"]

WRITE_FILE = [{"type": "function", "function": {
    "name": "write_file",
    "description": "Replace the whole content of a file in the workspace.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string", "description": "relative path, e.g. math_utils.py"},
        "content": {"type": "string", "description": "the complete new file content"}},
        "required": ["path", "content"]}}}]
SYSTEM = ("You are a coding agent working in a small Python repo. You can change files ONLY by calling "
          "write_file with the complete new file content. Do not explain; call the tool.")

# (name, file, stub, test file, test code, instruction). Task 1 is the playbooks' own Step 6 task.
TASKS = [
    ("add", "math_utils.py",
     'def add(a, b):\n    """Return the sum of a and b."""\n    pass\n',
     "test_math_utils.py",
     "import math_utils\n\n\ndef test_add():\n    assert math_utils.add(1, 2) == 3\n",
     "Please implement add() in math_utils.py and make sure the test passes."),
    ("slugify", "text_utils.py",
     'def slugify(text):\n    """Lowercase, words joined by single hyphens, only a-z and 0-9."""\n    pass\n',
     "test_text_utils.py",
     "from text_utils import slugify\n\n\ndef test_slugify():\n"
     "    assert slugify('Hello, World!') == 'hello-world'\n"
     "    assert slugify('  DGX   Spark -- GB10 ') == 'dgx-spark-gb10'\n"
     "    assert slugify('a_b') == 'a-b'\n",
     "Implement slugify() in text_utils.py so the tests in test_text_utils.py pass."),
    ("parse_duration", "durations.py",
     'def parse_duration(s):\n    """"1h30m" -> 5400. Units: h, m, s. Return total seconds as int."""\n    pass\n',
     "test_durations.py",
     "from durations import parse_duration\n\n\ndef test_parse_duration():\n"
     "    assert parse_duration('1h30m') == 5400\n"
     "    assert parse_duration('45s') == 45\n"
     "    assert parse_duration('2h5s') == 7205\n",
     "Implement parse_duration() in durations.py so the tests in test_durations.py pass."),
]

DENY = ("import os", "subprocess", "socket", "shutil", "open(", "eval(", "exec(", "__import__", "sys.exit")
RUNNER = ("import importlib, sys, traceback\n"
          "mod = importlib.import_module(sys.argv[1])\n"
          "fails = 0\n"
          "for name in [n for n in dir(mod) if n.startswith('test_')]:\n"
          "    try:\n        getattr(mod, name)()\n        print('PASS', name)\n"
          "    except Exception:\n        fails += 1\n        print('FAIL', name)\n"
          "        traceback.print_exc(limit=1)\n"
          "sys.exit(1 if fails else 0)\n")


def parse_args(tc: dict) -> dict | None:
    """Tool arguments arrive as a JSON string (OpenAI style) or a dict (some servers)."""
    a = (tc.get("function") or {}).get("arguments")
    if isinstance(a, dict):
        return a
    try:
        v = json.loads(a or "")
        return v if isinstance(v, dict) else None
    except (TypeError, json.JSONDecodeError):
        return None


def run_tests(work: Path, test_file: str) -> tuple[bool, str]:
    """Run every test_* function of the test module with a stdlib runner (no pytest needed)."""
    (work / "_run.py").write_text(RUNNER)
    try:
        p = subprocess.run([sys.executable, "_run.py", test_file[:-3]], cwd=work, capture_output=True,
                           text=True, timeout=20, env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})
        return p.returncode == 0, (p.stdout + p.stderr).strip()[-400:]
    except subprocess.TimeoutExpired:
        return False, "timed out after 20 s"


def attempt(base: str, model: str, task: tuple, trial: int) -> dict:
    name, fname, stub, tname, tcode, ask = task
    rec = {"model": model, "task": name, "trial": trial, "tool_call": False, "args_ok": False,
           "path_ok": False, "pass1": False, "pass2": False, "secs": 0.0, "note": ""}
    with tempfile.TemporaryDirectory(prefix="w25-probe-") as tmp:
        work = Path(tmp)
        (work / fname).write_text(stub)
        (work / tname).write_text(tcode)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"{ask}\n\n--- {fname} ---\n{stub}\n--- {tname} ---\n{tcode}"}]
        for turn in (1, 2):
            try:
                r = chat(base, model, msgs, tools=WRITE_FILE, max_tokens=300, extra={"reasoning_effort": "none"})
            except Exception as e:  # noqa: BLE001
                detail = e.read().decode(errors="replace")[:160] if hasattr(e, "read") else str(e)[:160]
                rec["note"] = "server refused: " + (json.loads(detail).get("error", {}).get("message", detail)
                                                   if detail.startswith("{") else detail)
                return rec
            rec["secs"] += r["total_ms"] / 1000
            calls = [c for c in r["tool_calls"] if (c.get("function") or {}).get("name") == "write_file"]
            if not calls:
                if "<tool_call>" in r["text"] or '"write_file"' in r["text"] or "<function=" in r["text"]:
                    why = "tool call printed as TEXT — the server's parser missed it, so no edit happened"
                else:
                    why = f"prose, no edit: {r['text'][:50]!r}"
                rec["note"] = why if turn == 1 else f"{rec['note']} · repair turn: {why[:40]}"
                return rec
            rec["tool_call"] = True
            args = parse_args(calls[0])
            if not args or not isinstance(args.get("content"), str):
                rec["note"] = "tool args were not valid JSON"
                return rec
            rec["args_ok"] = True
            path = str(args.get("path", ""))
            if path.lstrip("./") != fname:
                rec["note"] = f"wrote {path!r}, expected {fname!r}"
                return rec
            rec["path_ok"] = True
            code = args["content"]
            bad = [d for d in DENY if d in code]
            if bad:
                rec["note"] = f"refused to run: code contains {bad[0]!r}"
                return rec
            (work / fname).write_text(code)
            passed, out = run_tests(work, tname)
            rec["pass1" if turn == 1 else "pass2"] = passed
            if passed:
                rec["note"] = "passed first try" if turn == 1 else f"passed after one repair ({rec['note']})"
                return rec
            last = out.splitlines()[-1][:60] if out else ""
            rec["note"] = f"tests failed: {last}" if turn == 1 else f"tests failed again after repair: {last}"
            # one repair turn: show the model its own tool call and the real test output
            msgs += [{"role": "assistant", "content": r["text"] or "", "tool_calls": [calls[0]]},
                     {"role": "tool", "tool_call_id": calls[0].get("id", "call_0"), "content": f"Tests failed:\n{out}"},
                     {"role": "user", "content": f"The tests failed. Fix {fname} with write_file."}]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma-separated; default: playbook model on the Spark, "
                                                 "or this laptop's Ollama models")
    ap.add_argument("--trials", type=int, default=1, help="repeat each task N times (reliability needs N > 1)")
    args = ap.parse_args()

    banner("Lab 18-2 · reliability probe — real tool calls, real edits, real tests",
           "3 tasks · write_file tool · tests run in a temp dir · one repair turn")
    base, src = resolve("ollama")
    if src == "dry":
        warn("no Ollama endpoint reachable (Spark or laptop). Nothing to probe.")
        result("DRY: this lab needs a live model. Start Ollama on the Spark (Section 3) or on this laptop.")
        return
    if args.models:
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    elif src == "spark":
        wanted = [PLAYBOOK_MODEL]
    else:
        have = laptop_models()
        wanted = [m for m in LAPTOP_DEFAULT if m in have] or have[:2]
    native = base[:-3] if base.endswith("/v1") else base
    print(f"→ {base} · {'Spark Ollama' if src == 'spark' else 'Ollama on THIS laptop — LAPTOP STAND-IN'}")

    step(1, "does each model advertise tool support? (GET /api/show → capabilities)")
    for m in wanted:
        try:
            caps = http_json("POST", native + "/api/show", {"model": m}, timeout=15).get("capabilities") or []
        except Exception as e:  # noqa: BLE001
            caps = [f"? ({str(e)[:30]})"]
        print(f"│ {m:28s} {', '.join(caps)}{'' if 'tools' in caps else '   ← no tools: an agent cannot edit files'}")

    step(2, f"run {len(TASKS)} tasks × {len(wanted)} models × {args.trials} trial(s)")
    recs = []
    for m in wanted:
        for t in TASKS:
            for k in range(1, args.trials + 1):
                rec = attempt(base, m, t, k)
                recs.append(rec)
                mark = "✓" if rec["pass1"] or rec["pass2"] else "✕"
                print(f"{mark} {m:24s} {t[0]:15s} {rec['secs']:5.1f}s  {rec['note']}")
                if rec["note"].startswith("server refused"):
                    break                               # same answer for every task — do not spam the server
            if recs[-1]["note"].startswith("server refused"):
                break

    step(3, "scorecard")
    rows = []
    for m in wanted:
        mine = [r for r in recs if r["model"] == m]
        n = len(TASKS) * args.trials
        pct = lambda key: f"{sum(r[key] for r in mine)}/{n}"       # noqa: E731
        secs = [r["secs"] for r in mine if r["secs"]]
        rows.append([m, pct("tool_call"), pct("args_ok"), pct("pass1"),
                     f"{sum(r['pass1'] or r['pass2'] for r in mine)}/{n}",
                     f"{statistics.median(secs):.1f}s" if secs else "—"])
    table(rows, ["model", "tool call", "valid args", "pass 1st try", "pass ≤1 repair", "median time"])
    RUNS.mkdir(exist_ok=True)
    out = RUNS / "lab02_scorecard.json"
    out.write_text(json.dumps({"source": src, "base": base, "records": recs}, indent=1))
    print(f"→ wrote {out.relative_to(HERE.parents[1])}")
    label = "LIVE on the Spark" if src == "spark" else "LAPTOP STAND-IN — model behaviour is real, times are this laptop's"
    note(label + ". One trial is an anecdote: rerun with --trials 5 before you trust a model for agent work.")
    result("A coding model is only as good as its worst step: no tool call, bad JSON or a wrong path is a "
           "failed edit, however good the code would have been.")


if __name__ == "__main__":
    main()

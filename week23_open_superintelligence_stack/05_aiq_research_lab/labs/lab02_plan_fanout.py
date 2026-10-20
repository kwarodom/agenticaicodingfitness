#!/usr/bin/env python3
"""Lab 02 · Plan, then fan out — the Deep Agent's two moves, for real and timed.

The demos SHOW the plan (step02) and the fan-out trace (step03). This lab DOES both:
it asks the model for a machine-readable plan (JSON), writes it to a filesystem
To-Do like the blueprint does, then runs two researcher sub-agents CONCURRENTLY and
proves the fan-out win with a wall-clock-vs-sum-of-latencies speedup number.

Run:  cd /Users/altodev/Desktop/agenticaicodingfitness
      .venv/bin/python week23/05_aiq_research_lab/labs/lab02_plan_fanout.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

TASK = ("Should a 10-analyst research team buy a DGX Spark or keep paying for a "
        "frontier API? Recommend, with break-even reasoning.")

PLAN_PROMPT = (
    "You are the Planning Sub-Agent of a deep-research agent. Decompose the task into a "
    'JSON array of exactly 4 steps, each {"step": "...", "parallel_ok": true|false} — '
    "keep each step under 12 words; parallel_ok means the step needs no earlier step's "
    f"output. Reply with JSON only, no prose. Task: {TASK}")

RESEARCHER_PROMPT = (
    "You are Researcher Sub-Agent {name} (Nemotron Super) in a deep-research fan-out. "
    "Your assigned sub-task: {step}. In 2 short sentences: the evidence you would gather "
    "(sources/tools) and your provisional finding.")


def _client():
    from openai import OpenAI
    # max_retries=0: one honest 25s timeout beats three silent 25s stalls
    return OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY,
                  timeout=25.0, max_retries=0)


def _ask(client, prompt: str, max_tokens: int) -> str:
    from openai import BadRequestError
    kw = dict(model=config.MODEL, temperature=0.2, max_tokens=max_tokens,
              messages=[{"role": "user", "content": prompt}])
    try:    # thinking models: skip the preamble — we want the JSON/sentences
        r = client.chat.completions.create(extra_body={"reasoning_effort": "none"}, **kw)
    except BadRequestError:   # endpoint rejects the knob → plain retry
        r = client.chat.completions.create(**kw)
    msg = r.choices[0].message
    # endpoints that ignore the knob may burn the budget thinking, leaving
    # content empty — the draft in `reasoning` is then better than nothing
    text = (msg.content or "").strip() or str(getattr(msg, "reasoning", "") or "")
    return re.sub(r"<think>.*?</think>", " ", text, flags=re.S).strip()


def _parse_plan(text: str) -> list[tuple[str, bool]]:
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            items = json.loads(m.group(0))
            return [(str(i.get("step", "")), bool(i.get("parallel_ok")))
                    for i in items if isinstance(i, dict) and i.get("step")]
        except Exception:  # noqa: BLE001 — thinking models sometimes wrap JSON in prose
            pass
    return [(s, True) for s in re.findall(r"^\s*\d+[.)]\s*(.+)$", text, re.M)]


def _expected() -> None:
    print("◈ [no endpoint — showing expected output]  the real run produces:\n")
    print('  ← plan (JSON): [{"step": "define TCO scope", "parallel_ok": false},')
    print('                  {"step": "gather DGX price + power over 3y", "parallel_ok": true},')
    print('                  {"step": "gather API $/1M tok + team volume", "parallel_ok": true}, …]')
    print("  ← plan written → .sandbox/todo.md  (the blueprint's filesystem To-Do)")
    print("  → DISPATCH Researcher A + B concurrently …")
    print("  ← A done in ~6.1s · B done in ~6.8s · wall clock ~6.9s")
    print("  ◆ fan-out speedup: 12.9s of research in 6.9s of wall clock — ~1.9x")
    print("\n  go REAL:  ollama pull nemotron-3-nano   (or DGX_BASE_URL / DGX_CONN=cloud)")


def main() -> None:
    print("▣ Lab 02 · Deep Agent — plan to file, then fan out researchers")
    print(f"  endpoint: {config.safe_base_url()} · model: {config.MODEL} · mode: {config.MODE}\n")
    print(f'» escalated task: "{TASK}"\n')
    if config.MODE != "real":
        _expected()
        return
    client = _client()
    try:
        # the plan is ONE long-ish generation — give it 40s (slow local 12B models
        # emit ~8 tok/s); the concurrent researchers keep the tighter 25s cap
        raw = _ask(client.with_options(timeout=40.0), PLAN_PROMPT, max_tokens=280)
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ planning call failed ({type(e).__name__}) — endpoint down, or the "
              "model outran the 40s planning cap.")
        print("    rerun, or pin a faster model:  DGX_MODEL=<small-model> .venv/bin/python "
              "week23/05_aiq_research_lab/labs/lab02_plan_fanout.py")
        return
    plan = _parse_plan(raw)
    if not plan:
        print("  ✗ could not parse a plan — raw reply below. Tighten PLAN_PROMPT and rerun.")
        print("  " + raw[:400])
        return
    todo = config.ensure_sandbox() / "todo.md"
    todo.write_text("\n".join(f"- [ ] {s}  {'(parallel-ok)' if p else '(sequential)'}"
                              for s, p in plan) + "\n")
    for i, (s, p) in enumerate(plan, 1):
        print(f"  → TODO {i}. {s[:70]}  {'∥' if p else '⇢'}")
    print(f"  ← plan written → {todo.relative_to(config.PKG.parent.parent)}\n")

    picks = [s for s, p in plan if p][:2] or [s for s, _ in plan][:2]
    print(f"  → DISPATCH {len(picks)} researchers concurrently …")

    def _research(arg: tuple[str, str]) -> tuple[str, str, float]:
        name, step = arg
        t0 = time.time()
        out = _ask(client, RESEARCHER_PROMPT.format(name=name, step=step), max_tokens=160)
        return name, out, time.time() - t0

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(_research, zip("AB", picks)))
    wall = time.time() - t0
    for name, out, dt in results:
        print(f"\n  ← Researcher {name} ({dt:.1f}s): {out[:220]}")
    total = sum(dt for _, _, dt in results)
    print(f"\n  ◆ fan-out speedup: {total:.1f}s of research in {wall:.1f}s wall clock — "
          f"~{total / wall if wall else 1:.1f}x")
    print("\n✓ Takeaway — decompose first, delegate second. A written plan makes the run")
    print("  observable + resumable; parallel_ok flags are what make the fan-out legal.")


if __name__ == "__main__":
    main()

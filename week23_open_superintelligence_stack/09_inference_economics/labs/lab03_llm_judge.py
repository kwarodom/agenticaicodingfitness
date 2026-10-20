#!/usr/bin/env python3
"""Lab 03 — LLM-as-judge over a full golden set, then score the JUDGE itself.

demos/step04_evaluate.py judges ONE case live and replays the rest; this lab
judges the whole golden set with your live endpoint, computes the golden-set
score, and — the hands-on twist — compares every verdict to a known ground
truth to measure JUDGE AGREEMENT. An eval gate is only as good as its judge.

Run:  cd <repo root> && .venv/bin/python week23/09_inference_economics/labs/lab03_llm_judge.py
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

DEADLINE_S = 50
GATE = 0.80

# (task, agent answer, ground truth: should a competent judge PASS it?)
GOLDEN = [
    ("What color is the sky on a clear day?", "Blue.", True),
    ("2 + 2 = ?", "5", False),
    ("Capital of France?", "Paris", True),
    ("Is 17 prime?", "No — 17 = 3 × 6.", False),
    ("Name a mammal that can fly.", "The bat.", True),
]

JUDGE_TMPL = ("You are a strict grader. Task: {task}\nAnswer given: {answer}\n"
              "Is the answer correct? Reply with exactly one word: PASS or FAIL.")


def p(line: str = "") -> None:
    print(line, flush=True)


def extract_verdict(text: str) -> bool | None:
    """Last PASS/FAIL mention wins — thinking models put the answer at the end."""
    clean = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
    hit = None
    for m in re.finditer(r"\bPASS\b|\bFAIL\b", clean, re.I):
        hit = m.group(0).upper()
    return None if hit is None else (hit == "PASS")


def main() -> None:
    p("━" * 64)
    p("  ▣ LAB 03 — LLM-as-judge · golden set · judge agreement")
    p("━" * 64)
    p(f"  connection: {config.CONN} ({config.conn_human()}) · judge model: {config.MODEL}\n")

    if config.MODE != "real":
        p("✗ no endpoint reachable — the judge must be a live model. Start one:")
        p("    C · ollama serve && ollama pull qwen3:4b")
        p("    B · export DGX_CONN=cloud DGX_CLOUD_URL=https://integrate.api.nvidia.com/v1 \\")
        p("            DGX_API_KEY=nvapi-...")
        p("\n[no endpoint — showing expected output]")
        p("  '2 + 2 = ?' answer '5' → judge FAIL ✓ (agrees with ground truth)")
        p("  GOLDEN-SET SCORE 60% · JUDGE AGREEMENT 5/5 = 100% ≥ gate 80% → judge trusted")
        p("  (SAMPLE only — a weak judge may PASS the wrong answers; that's the lesson)")
        return

    import openai
    from openai import OpenAI
    client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY,
                    timeout=25.0, max_retries=0)

    def judge(task: str, answer: str, budget: float):
        """Curb thinking-model traces; drop the extra param if the server 400s it."""
        kw = dict(model=config.MODEL, temperature=0.0, max_tokens=300, timeout=budget,
                  messages=[{"role": "user",
                             "content": JUDGE_TMPL.format(task=task, answer=answer)}])
        try:
            return client.chat.completions.create(**kw, extra_body={"reasoning_effort": "none"})
        except openai.BadRequestError:
            return client.chat.completions.create(**kw)

    p(f"◈ Step 1 — the judge grades all {len(GOLDEN)} answers, live")
    p(f"  {'task':<30}{'answer':<18}{'judge':>7}{'truth':>7}{'agree':>7}")
    p("  " + "─" * 70)
    start, judged, agree, passes = time.time(), 0, 0, 0
    for task, answer, truth in GOLDEN:
        left = DEADLINE_S - (time.time() - start)
        if left < 3:
            p("  ⏱ lab wall-clock budget hit — scoring what we have.")
            break
        try:
            resp = judge(task, answer, min(25.0, left))
        except (openai.APITimeoutError, openai.APIConnectionError):
            judged += 1                            # a judge that times out is a wrong judge
            p(f"  {task[:28]:<30}{answer[:16]:<18}{'⏱':>7}{'PASS' if truth else 'FAIL':>7}{'✗':>7}")
            continue
        except Exception as e:  # noqa: BLE001
            p(f"  ✗ call failed ({type(e).__name__}) — 404=wrong URL/model · 401=key")
            break
        msg = resp.choices[0].message
        text = str(getattr(msg, "reasoning", "") or "") + " " + (msg.content or "")
        verdict = extract_verdict(text)
        judged += 1
        if verdict is None:
            p(f"  {task[:28]:<30}{answer[:16]:<18}{'??':>7}{'PASS' if truth else 'FAIL':>7}{'—':>7}")
            continue
        passes += int(verdict)
        ok = verdict == truth
        agree += int(ok)
        p(f"  {task[:28]:<30}{answer[:16]:<18}"
          f"{'PASS' if verdict else 'FAIL':>7}{'PASS' if truth else 'FAIL':>7}"
          f"{'✓' if ok else '✗':>7}")

    if not judged:
        return
    score, agreement = passes / judged, agree / judged
    p("\n◈ Step 2 — two scores, two meanings")
    p(f"  GOLDEN-SET SCORE  {passes}/{judged} = {score:.0%}   (how good the AGENT answers were)")
    p(f"  JUDGE AGREEMENT   {agree}/{judged} = {agreement:.0%}   (how good the JUDGE is)")
    if agreement >= GATE:
        p(f"  ✓ agreement ≥ gate {GATE:.0%} — this judge is trustworthy enough to gate CI.")
    else:
        p(f"  ✗ agreement < gate {GATE:.0%} — do NOT gate CI on this judge: fix the judge")
        p("    (bigger model, better rubric prompt) before trusting any goodput number.")
    p("\n✓ done — perf (lab01) × correctness (this) = value. Next app: 10_nemo_gym_rl,")
    p("  where verifiable rewards turn this same eval loop into RL training signal.")


if __name__ == "__main__":
    main()

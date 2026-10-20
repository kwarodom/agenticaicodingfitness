#!/usr/bin/env python3
"""LAB 03 · Evaluate + promote — run a real A/B gate between two live models.

The demos simulate the promotion decision; this lab MAKES one. Two models on
your endpoint play teacher and student on a small golden set, a blind LLM-judge
scores each pair, and the gate rule fires: PROMOTE the student only if it ties
or beats the teacher on quality AND is cheaper (latency as the cost proxy).

Run:  .venv/bin/python week23/11_data_flywheel/labs/lab03_eval_gate.py
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

GOLDEN = [  # tiny on purpose — a real Evaluator run uses hundreds, held out
    "A guest says their room key stopped working at 11pm. What do you do? One sentence.",
    "Explain in one sentence why deduplicating training data prevents overfitting.",
]

JUDGE = ("Two anonymous assistants answered the same question.\nQUESTION: {q}\n"
         "ANSWER A: {a}\nANSWER B: {b}\n"
         "Which answer is better? Reply with exactly one word: A, B, or TIE.")

STUDENT_HINTS = ("nano", "gemma", "llama3.1:8b", "phi", "qwen3:4b", "mini", "small")


def pick_pair(models: list[str]) -> tuple[str, str]:
    teacher = config.MODEL
    for hint in STUDENT_HINTS:                       # smallest-looking ≠ teacher
        for m in models:
            if hint in m.lower() and m != teacher:
                return teacher, m
    others = [m for m in models if m != teacher]
    return teacher, (others[0] if others else teacher)


def chat(cli, **kw):
    """One completion with thinking disabled where the server honors it.

    Thinking models (gemma4, qwen3, nemotron-3…) spend a small max_tokens budget
    on their reasoning preamble and time out; `reasoning_effort:"none"` skips the
    preamble on Ollama/OpenAI-compatible servers. Servers that reject the hint
    get one plain retry. A timeout also gets one retry: swapping between the
    teacher and student often spends the whole 25s budget loading weights —
    the load finishes server-side, so the retry lands on a warm model.
    """
    from openai import APITimeoutError, BadRequestError
    try:
        return cli.chat.completions.create(extra_body={"reasoning_effort": "none"}, **kw)
    except BadRequestError:
        return cli.chat.completions.create(**kw)
    except APITimeoutError:
        return cli.chat.completions.create(extra_body={"reasoning_effort": "none"}, **kw)


def ask(cli, model: str, prompt: str) -> tuple[str, float]:
    t0 = time.time()
    r = chat(cli,
        model=model, temperature=0.2, max_tokens=150,
        messages=[{"role": "user", "content": prompt}])
    return (r.choices[0].message.content or "").strip(), time.time() - t0


def main() -> None:
    print("━" * 64)
    print("  ▣ LAB 03 · Evaluate + promote — the gate is the whole game")
    print("━" * 64)
    print(f"  mode={config.MODE} · conn={config.CONN} · endpoint={config.safe_base_url()}\n")

    if config.MODE != "real":
        print("◈ [no endpoint — showing expected output] a REAL run prints:")
        print("  teacher=qwen3.6:35b-a3b-q8_0  student=gemma3:4b")
        print("  Q1  judge → TIE      teacher 6.1s · student 1.4s")
        print("  Q2  judge → A(tchr)  teacher 5.8s · student 1.5s")
        print("  quality: student ties-or-wins 1/2 · cost: student 4.2× faster")
        print("  ── GATE: student under quality bar (need ≥ 1/2 wins+ties on ALL) —")
        print("     …keep teacher. Distill more rounds (demo step04 shows the arc).")
        print("  — go REAL:  ollama serve   + pull a second model (ollama pull gemma3)")
        return

    from openai import OpenAI
    cli = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=25.0, max_retries=0)
    models = config.list_local_models()
    teacher, student = pick_pair(models)
    if teacher == student:
        print(f"⚠ only one model on the endpoint ({teacher}) — it plays BOTH roles,")
        print("  so treat this run as a dry-run of the gate mechanics.")
    print(f"◈ teacher = {teacher}\n◈ student = {student}\n")

    # answer ALL golden items per model, grouped — the endpoint swaps weights on
    # every model change, and per-question alternation would pay that reload
    # 3× per item instead of twice per run
    answers: dict[str, list[tuple[str, float]]] = {}
    for m in dict.fromkeys((teacher, student)):        # unique, order-stable
        outs = []
        for i, q in enumerate(GOLDEN, 1):
            try:
                outs.append(ask(cli, m, q))
            except Exception as e:  # noqa: BLE001
                print(f"  ✗ Q{i} candidate call to {m} failed ({type(e).__name__}) — skipped")
                outs.append(("", 0.0))
        answers[m] = outs

    wins_or_ties, t_lat, s_lat = 0, 0.0, 0.0
    for i, q in enumerate(GOLDEN, 1):
        (ta, tt), (sa, st) = answers[teacher][i - 1], answers[student][i - 1]
        if not ta or not sa:
            continue                                   # a candidate call failed above
        t_lat, s_lat = t_lat + tt, s_lat + st
        # blind judging: student is always shown as B, teacher as A — a real
        # Evaluator run also randomizes A/B order to cancel position bias.
        try:
            jm = chat(cli,
                model=teacher, temperature=0.0, max_tokens=250,
                messages=[{"role": "user", "content": JUDGE.format(q=q, a=ta, b=sa)}]
                ).choices[0].message
            # reasoning first, content last — if the no-think hint was ignored the
            # verdict still lands at the END of the thinking trace
            raw = str(getattr(jm, "reasoning", "") or "") + " " + (jm.content or "")
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ Q{i} judge call failed ({type(e).__name__}) — scored TIE")
            raw = "TIE"
        # last-mention-wins extraction (thinking judges reason first, answer last);
        # case-sensitive A/B so the article 'a' can't count as a verdict.
        hits = re.findall(r"\b(?:[Tt][Ii][Ee]|A|B)\b", raw)
        verdict = hits[-1].upper() if hits else "TIE"
        label = {"A": "A(teacher)", "B": "B(student)", "TIE": "TIE"}[verdict]
        if verdict in ("B", "TIE"):
            wins_or_ties += 1
        print(f"  Q{i}  judge → {label:<11} teacher {tt:4.1f}s · student {st:4.1f}s")
        print(f"      » {q[:64]}")

    speedup = (t_lat / s_lat) if s_lat else 0.0
    n = len(GOLDEN)
    print(f"\n  quality: student ties-or-wins {wins_or_ties}/{n}")
    print(f"  cost proxy: student {speedup:.1f}× faster wall-clock "
          f"({s_lat:.1f}s vs {t_lat:.1f}s total)")

    promote = wins_or_ties == n and speedup > 1.0
    if promote:
        print("\n  ── GATE: ✓ PROMOTE — student matched quality on every golden item")
        print("     AND runs cheaper. In production: swap the serving route, keep")
        print("     observing, spin the flywheel again.")
    else:
        print("\n  ── GATE: …keep teacher — the student must tie-or-win on ALL items")
        print("     AND be cheaper. No partial credit: a gate with exceptions is not")
        print("     a gate. Curate + distill another round, then re-run this eval.")

    print("\n  Takeaway — promotion is a CI gate for weights (Week 15 pattern): only")
    print("  proven parity ships. Next app: the capstone wires this into a hotel.")


if __name__ == "__main__":
    main()

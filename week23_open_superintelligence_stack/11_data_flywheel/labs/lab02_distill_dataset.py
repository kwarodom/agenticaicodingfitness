#!/usr/bin/env python3
"""LAB 02 · Distill — harvest teacher answers into a train.jsonl for Customizer.

Distillation starts with a dataset: the BIG model answers your domain prompts,
and those (prompt → teacher answer) pairs become the file a fine-tune consumes.
This lab calls your live model as the teacher, writes .sandbox/train.jsonl in
the messages format NeMo Customizer / PEFT expect, and prints the Spark
fit-math that decides whether the fine-tune runs on-box.

Run:  .venv/bin/python week23/11_data_flywheel/labs/lab02_distill_dataset.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# Domain prompts (smart-hotel flavored — the capstone reuses this domain).
# In production these come from lab01's curated.jsonl, not a hardcoded list.
PROMPTS = [
    "A hotel guest reports the AC in room 305 is rattling. Reply in one sentence "
    "with the action you take.",
    "Guest asks for late checkout in room 210. Policy allows until 14:00. Reply "
    "in one sentence.",
    "Night setback: what temperature should unoccupied guest rooms hold overnight? "
    "One sentence.",
]

SYSTEM = "You are a concise hotel-operations agent. Answer in one sentence."


def chat(cli, **kw):
    """One completion with thinking disabled where the server honors it.

    Thinking models (gemma4, qwen3, nemotron-3…) spend a small max_tokens budget
    on their reasoning preamble and time out; `reasoning_effort:"none"` skips the
    preamble on Ollama/OpenAI-compatible servers. Servers that reject the hint
    get one plain retry. A timeout also gets one retry: the first call to a
    cold model often spends the whole 25s budget loading weights — the load
    finishes server-side, so the retry lands on a warm model.
    """
    from openai import APITimeoutError, BadRequestError
    try:
        return cli.chat.completions.create(extra_body={"reasoning_effort": "none"}, **kw)
    except BadRequestError:
        return cli.chat.completions.create(**kw)
    except APITimeoutError:
        return cli.chat.completions.create(extra_body={"reasoning_effort": "none"}, **kw)


def fit_math() -> None:
    print("◈ Spark fit-math (runbook §1.2 — weights ≈ GB/B-param · ×1.18 overhead):")
    print("  serve  Nano 30B @ Q8     ≈ 30 × 1.06 × 1.18 ≈  38 GB  → fits a 128 GB Spark")
    print("  LoRA-train needs ~4–6× weight memory (grads/optimizer/activations):")
    print("    ≤12B student  → comfortable on 1 Spark (PEFT path, runbook §2.11)")
    print("    Nano 30B LoRA → borderline on 1 Spark; 2 Sparks (TP=2) or SIM beyond")
    print("  full-SFT 120B teacher → datacenter territory — that's WHY we distill.\n")


def main() -> None:
    print("━" * 64)
    print("  ▣ LAB 02 · Distill — teacher answers → train.jsonl")
    print("━" * 64)
    print(f"  mode={config.MODE} · teacher={config.MODEL} · endpoint={config.safe_base_url()}\n")
    fit_math()

    rows = []
    if config.MODE == "real":
        from openai import OpenAI
        cli = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=25.0, max_retries=0)
        print(f"◈ Harvesting {len(PROMPTS)} teacher completions from {config.MODEL}:")
        for i, p in enumerate(PROMPTS, 1):
            t0 = time.time()
            try:
                r = chat(cli,
                    model=config.MODEL, temperature=0.2, max_tokens=200,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": p}])
                ans = (r.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001
                print(f"  ✗ call {i} failed ({type(e).__name__}) — skipped")
                continue
            if not ans:   # thinking budget exhausted before an answer — not a label
                print(f"  ✗ call {i} returned no visible answer (reasoning ate the budget) — skipped")
                continue
            print(f"  {i}. [{time.time()-t0:4.1f}s] » {p[:52]}…")
            print(f"     teacher · {ans[:76]}")
            rows.append({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": p},
                {"role": "assistant", "content": ans},   # ← the distillation label
            ]})
    else:
        print("◈ [no endpoint — showing expected output] a REAL run prints:")
        print("  1. [ 1.2s] » A hotel guest reports the AC in room 305 is rattl…")
        print("     teacher · I file work order WO-1183 for room 305 and notify…")
        print("  …and writes one {'messages': [system,user,assistant]} row per prompt.")
        print("  — go REAL:  ollama serve   (or export DGX_BASE_URL=http://<spark>:11434/v1)")

    if rows:
        out = config.ensure_sandbox() / "train.jsonl"
        out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        print(f"\n✓ wrote {len(rows)} training rows → {out}")
        print("  each row: {'messages': [system, user, assistant]} — the assistant turn")
        print("  is the TEACHER's answer. Fine-tune a student on this and it imitates")
        print("  the teacher on your domain. That is distillation, whole and entire.")

    print("\n◈ What consumes this file on real hardware:")
    print("  [SPARK] git clone https://github.com/NVIDIA-AI-Blueprints/data-flywheel")
    print("          # blueprint = NeMo microservices on k8s + DC GPUs — study it,")
    print("          # don't expect it to run on one ARM64 Spark (runbook §2.11).")
    print("  [SPARK] PEFT/LoRA on a ≤12B student — week19's dgx_finetune path —")
    print("          pointing its dataset arg at this train.jsonl.")
    print("\n  Takeaway — 'fine-tuning data' is not exotic: it's your curated logs")
    print("  with the teacher's answers attached. Next: lab03 gates the student.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""LAB 01 · Curate — run a real mini-Curator pipeline on raw agent logs.

The demos SHOW the 1M→62k funnel; this lab makes YOU run one. Fourteen raw
production traces (dupes, junk, PII and all) go through the same four stages
NeMo Curator applies — dedup → quality filter → PII scrub → LLM-judge label —
and the survivors land in .sandbox/curated.jsonl, ready for lab02.

Run:  .venv/bin/python week23/11_data_flywheel/labs/lab01_curate_logs.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# ── raw "production" traces — realistically messy on purpose ──────────────────
RAW = [
    {"prompt": "How do I reset the thermostat in room 412?", "answer": "Hold SET for 5s, then select AUTO.", "ok": True},
    {"prompt": "how do i reset the thermostat in room 412??", "answer": "Hold SET for 5 seconds, choose AUTO.", "ok": True},  # near-dup
    {"prompt": "How do I reset the thermostat in room 412?", "answer": "Hold SET for 5s, then select AUTO.", "ok": True},     # exact dup
    {"prompt": "Guest in 210 wants late checkout", "answer": "Granted until 14:00; housekeeping notified.", "ok": True},
    {"prompt": "asdfgh", "answer": "I don't understand.", "ok": False},                                                        # junk
    {"prompt": "Book a spa slot for jane.doe@example.com tomorrow", "answer": "Booked 10:00 for jane.doe@example.com.", "ok": True},  # PII
    {"prompt": "Call me at +1-555-867-5309 about the AC noise", "answer": "Logged; engineer will call +1-555-867-5309.", "ok": True}, # PII
    {"prompt": "What's the wifi password?", "answer": "It's on the key card sleeve: LOBBY-2026.", "ok": True},
    {"prompt": "whats the wifi password", "answer": "Printed on your key card sleeve: LOBBY-2026.", "ok": True},               # near-dup
    {"prompt": "Why is my key sk-live-abc123def456 rejected by the API?", "answer": "Key sk-live-abc123def456 is expired.", "ok": True},  # secret
    {"prompt": "Turn off all chillers now", "answer": "Refused — needs operator approval (guardrail).", "ok": True},
    {"prompt": "Is breakfast included?", "answer": "Yes, 06:30–10:00 in the atrium.", "ok": True},
    {"prompt": "###", "answer": "", "ok": False},                                                                              # junk
    {"prompt": "AC in 305 is rattling", "answer": "Work order WO-1183 filed for room 305.", "ok": True},
]

PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<EMAIL>"),
    (re.compile(r"\+?\d[\d\s()-]{7,}\d"), "<PHONE>"),
    (re.compile(r"sk-[A-Za-z0-9-]{8,}"), "<SECRET>"),
]

JUDGE_RUBRIC = ("You are a data curator. Was this agent answer correct and helpful "
                "for the prompt? Reply with exactly one word: KEEP or DROP.\n\n"
                "PROMPT: {p}\nANSWER: {a}")


def chat(cli, **kw):
    """One completion with thinking disabled where the server honors it.

    Thinking models (gemma4, qwen3, nemotron-3…) spend a small max_tokens budget
    on their reasoning preamble and time out before the verdict appears;
    `reasoning_effort:"none"` skips the preamble on Ollama/OpenAI-compatible
    servers. Servers that reject the hint get one plain retry. A timeout also
    gets one retry: the first call to a cold model often spends the whole 25s
    budget loading weights — the load finishes server-side, so the retry lands
    on a warm model.
    """
    from openai import APITimeoutError, BadRequestError
    try:
        return cli.chat.completions.create(extra_body={"reasoning_effort": "none"}, **kw)
    except BadRequestError:
        return cli.chat.completions.create(**kw)
    except APITimeoutError:
        return cli.chat.completions.create(extra_body={"reasoning_effort": "none"}, **kw)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def stage(name: str, rows: list, note: str) -> None:
    bar = "█" * max(1, round(len(rows) / len(RAW) * 28))
    print(f"  {name:<18}{len(rows):>4} rows   {note}")
    print(f"  {'':<18}{bar}")


def main() -> None:
    print("━" * 64)
    print("  ▣ LAB 01 · Curate — raw logs → training data (for real)")
    print("━" * 64)
    print(f"  mode={config.MODE} · conn={config.CONN} · endpoint={config.safe_base_url()}\n")

    print("The funnel — each stage is the same idea NeMo Curator runs at 1M-row scale:\n")
    stage("raw logs", RAW, "everything the agent saw")

    # ① dedup — exact + near (normalized prompt text)
    seen, deduped = set(), []
    for t in RAW:
        k = norm(t["prompt"])
        if k not in seen:
            seen.add(k)
            deduped.append(t)
    stage("dedup", deduped, "exact + normalized near-dups dropped")

    # ② quality filter — only successful, non-trivial traces become signal
    quality = [t for t in deduped if t["ok"] and len(t["prompt"].split()) >= 3 and t["answer"]]
    stage("quality filter", quality, "keep ok=True, ≥3 words, non-empty answer")

    # ③ PII scrub — redact, don't drop: the trace stays useful, the PII doesn't
    scrubbed = []
    for t in quality:
        p, a = t["prompt"], t["answer"]
        for rx, tag in PII_PATTERNS:
            p, a = rx.sub(tag, p), rx.sub(tag, a)
        scrubbed.append({"prompt": p, "answer": a})
    hits = sum(1 for s, q in zip(scrubbed, quality) if s["prompt"] != q["prompt"] or s["answer"] != q["answer"])
    stage("PII scrub", scrubbed, f"redacted email/phone/secret in {hits} traces")

    # ④ LLM-judge label — a model grades what's worth learning from
    print("\n◈ Stage 4 — LLM-judge labels (teacher grades the keepers):")
    labeled = []
    if config.MODE == "real":
        from openai import OpenAI
        cli = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=25.0, max_retries=0)
        for t in scrubbed[:3]:  # judge 3 live to stay <60s; Curator would do all
            try:
                r = chat(cli,
                    model=config.MODEL, temperature=0.0, max_tokens=200,
                    messages=[{"role": "user", "content": JUDGE_RUBRIC.format(p=t["prompt"], a=t["answer"])}])
                m = r.choices[0].message
                # read reasoning + content — if the no-think hint was ignored the
                # verdict still lands at the END of the thinking trace
                up = (str(getattr(m, "reasoning", "") or "") + " " + (m.content or "")).upper()
                # last mention wins — thinking models reason first, answer last
                verdict = "DROP" if up.rfind("DROP") > up.rfind("KEEP") else "KEEP"
            except Exception as e:  # noqa: BLE001
                print(f"  ✗ judge call failed ({type(e).__name__}) — labeling KEEP unscored")
                verdict = "KEEP"
            print(f"  {verdict:<5} ← {t['prompt'][:56]!r}")
            if verdict == "KEEP":
                labeled.append(t)
        labeled += scrubbed[3:]  # remainder passes unjudged in this lab
    else:
        print("  [no endpoint — showing expected output]")
        print("  KEEP  ← 'How do I reset the thermostat in room 412?'")
        print("  KEEP  ← 'Guest in 210 wants late checkout'")
        print("  KEEP  ← 'Book a spa slot for <EMAIL> tomorrow'")
        print("  — go REAL:  ollama serve   (then rerun this lab)")
        labeled = scrubbed
    stage("labeled", labeled, "judge-approved training candidates")

    # write the curated set — lab02 distills from this file
    out = config.ensure_sandbox() / "curated.jsonl"
    out.write_text("\n".join(json.dumps(t) for t in labeled) + "\n")
    print(f"\n✓ wrote {len(labeled)} curated traces → {out}")
    print("  Takeaway — curation is code you can read: 14 messy rows became "
          f"{len(labeled)} clean ones. Next: lab02 turns them into a train.jsonl.")


if __name__ == "__main__":
    main()

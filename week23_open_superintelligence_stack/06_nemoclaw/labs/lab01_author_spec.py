#!/usr/bin/env python3
"""LAB 01 · Author + LINT a NemoClaw agent spec — then A/B the persona.

demos/step01_define_agent.py hands you a finished spec. Here YOU are the author:
build a NEW specialist spec, run it through a 5-point lint (Spark fit-math, signed
policy, tool provenance), then — if an endpoint is up — ask the SAME base model
the SAME question twice: bare vs wearing the persona. The diff IS the lesson:
a spec turns a general model into a specialist without touching weights.

Run:  cd /Users/altodev/Desktop/agenticaicodingfitness && \
      .venv/bin/python week23/06_nemoclaw/labs/lab01_author_spec.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import view    # noqa: E402  (reused for its thinking-model / direct-model helpers)

# ── 1 · YOUR spec — a NEW specialist (the demo owns HVAC; you own energy) ─────
SPEC = {
    "name": "energy-analyst",
    "role": "Building-energy analyst for a hotel portfolio",
    "base_model": "nemotron-3-nano:30b-a3b",   # Nano = cheap, many specialists
    "system_prompt": ("You are a terse building-energy analyst. Every recommendation "
                      "cites a meter reading. You PROPOSE setpoints; you never write them."),
    "skills": ["nemo-retriever (tariff & meter RAG)", "cuopt (load-shift optimization)"],
    "tools": ["query_timeseries", "propose_setpoint"],
    "policy": {"sandbox": "openshell", "signed": True,
               "egress_allowlist": ["telemetry.internal"], "may_write": False},
}

# What the two attached skills actually expose — tools must come FROM skills.
SKILL_TOOLS = {"query_timeseries", "propose_setpoint"}

# Runbook fit-math: Q8 ≈ 1.06 GB/B-param, ×1.18 KV/runtime overhead, 90% of 128 GB usable.
_PARAMS_B = {"nemotron-3-nano": 30, "nemotron-3-super": 120, "nemotron-3-ultra": 550,
             "qwen3.6": 35, "llama3.1:8b": 8}
SPARK_USABLE_GB = 128 * 0.90


def _fits_spark(model: str) -> tuple[bool, float]:
    b = next((v for k, v in _PARAMS_B.items() if k in model.lower()), 9)
    need = b * 1.06 * 1.18
    return need <= SPARK_USABLE_GB, need


def lint(spec: dict) -> bool:
    print("◈ LINT — 5 checks before this spec is allowed near a sandbox\n")
    results = []

    def check(label: str, cond: bool, detail: str = "") -> None:
        results.append(cond)
        print(f"  {'✓' if cond else '✗'} {label}" + (f" — {detail}" if detail else ""))

    required = {"name", "role", "base_model", "system_prompt", "skills", "tools", "policy"}
    missing = required - set(spec)
    check("all 7 spec fields present", not missing,
          f"missing: {sorted(missing)}" if missing else "")
    fits, need = _fits_spark(spec["base_model"])
    check("base model fits ONE Spark (Q8 fit-math)", fits,
          f"{spec['base_model']} ≈ {need:.0f} GB vs {SPARK_USABLE_GB:.0f} GB usable")
    check("policy is SIGNED — the agent can't edit its own leash",
          spec["policy"].get("signed") is True)
    check("egress allowlist is non-empty (default-deny posture)",
          bool(spec["policy"].get("egress_allowlist")))
    check("every tool traces back to an attached skill",
          set(spec["tools"]) <= SKILL_TOOLS)
    print()
    return all(results)


QUESTION = ("Chiller plant kW jumped 18% overnight while the cooling load was flat. "
            "Most likely cause, and your first action?")

EXPECTED = """[no endpoint — showing expected output]
  » bare model (no spec)   : a broad, hedged essay — several causes, no citation
                             habit, may offer to change setpoints itself.
  » with the spec's persona: terse, cites the kW delta, PROPOSES an action, and
                             stays inside 'analyst who may not actuate'.
Same weights, different specialist — the spec is the difference."""


def _pick_ab_model() -> str:
    """The A/B needs answer TEXT — auto-swap a direct model, like view.classify()."""
    if view.is_thinking_model(config.MODEL):
        alt = view.pick_direct_model()
        if alt:
            print(f"  ◆ auto-picked {alt} for the A/B — {config.MODEL} is a thinking"
                  " model (it would spend the budget reasoning, not answering).\n")
            return alt
    return config.MODEL


def _ask(model: str, system: str | None, label: str) -> None:
    from openai import OpenAI
    # max_retries=0 — retries would triple the worst-case wall time (<60 s promise)
    client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY,
                    timeout=25.0, max_retries=0)
    messages = ([{"role": "system", "content": system}] if system else [])
    messages += [{"role": "user", "content": QUESTION}]
    r = client.chat.completions.create(model=model, messages=messages,
                                       max_tokens=250, temperature=0.3)
    msg = r.choices[0].message
    text = (msg.content or "").strip() or str(getattr(msg, "reasoning", "") or "").strip()
    if not text:
        text = "(model spent the whole budget thinking — raise max_tokens and re-run)"
    print(f"  » {label}:\n    " + text[:500].replace("\n", "\n    ") + "\n")


def main() -> None:
    print("━" * 64)
    print("  ▣ LAB 01 — author + lint a NemoClaw spec, then A/B the persona")
    print("━" * 64 + "\n")

    print("Step 1 — the spec YOU authored (editing it is the exercise):\n")
    print(json.dumps(SPEC, indent=2) + "\n")

    if not lint(SPEC):
        print("✗ lint FAILED — fix the spec above (that is the lab) and re-run.")
        return
    print("✓ spec passes lint — a governable specialist, on paper.\n")

    print("Step 2 — A/B the persona: same model, same question, ± the spec's prompt.\n")
    if config.MODE != "real":
        print("▣ MODE: SIM — no endpoint reachable. To go REAL:")
        print("    ollama pull qwen3.6:35b-a3b-q8_0        # local Ollama, or")
        print("    export DGX_BASE_URL=http://<spark>:11434/v1\n")
        print(EXPECTED)
    else:
        print(f"▣ REAL · {config.MODEL} @ {config.safe_base_url()} · {config.cost_note()}\n")
        try:
            model = _pick_ab_model()
            _ask(model, None, "bare model (no spec)")
            _ask(model, SPEC["system_prompt"], "with the spec's persona")
        except Exception as e:  # noqa: BLE001
            print(f"✗ endpoint call failed ({type(e).__name__}) — check the connection.\n")
            print(EXPECTED)
            return

    print("✓ Takeaway — composition before fine-tuning: the persona changed behaviour;")
    print("  the weights never moved. Next: labs/lab02_policy_gate.py")


if __name__ == "__main__":
    main()

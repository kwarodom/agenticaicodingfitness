#!/usr/bin/env python3
"""Lab 03 · The tool bus — pick, execute, observe, answer (a NAT loop in miniature).

The demo (step04) asks the model to LIST which tools it would call. This lab closes
the loop: a tiny 3-tool registry (real functions over real data — config.DGX_SPECS
and the live endpoint's model list), the model picks one as JSON, the lab EXECUTES
it, feeds the observation back, and gets a grounded final answer. Then it prints the
NeMo Agent Toolkit workflow.yml this maps to, and the real `nat` commands.

Run:  cd /Users/altodev/Desktop/agenticaicodingfitness
      .venv/bin/python week23/05_aiq_research_lab/labs/lab03_tool_bus.py
"""
from __future__ import annotations

import ast
import json
import operator as op
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

TASK = ("Which desk-side DGX fits a ~200B-parameter model, and how much unified "
        "memory and power does it use?")

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow}


def calc(expression: str) -> str:
    """Safe arithmetic — the sandbox data_analysis skill, minus the sandbox."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        raise ValueError("unsupported expression")
    return str(ev(ast.parse(expression, mode="eval").body))


def dgx_specs(machine: str = "DGX Spark") -> str:
    """The 'documents' tool — real spec-sheet data from config.DGX_SPECS."""
    hit = next((k for k in config.DGX_SPECS if machine.lower() in k.lower()), "DGX Spark")
    return f"{hit}: " + "; ".join(f"{k}={v}" for k, v in config.DGX_SPECS[hit].items())


def list_models() -> str:
    """The 'ai_data_platform' stand-in — what's live on the connected endpoint."""
    models = config.list_local_models()
    return ", ".join(models[:8]) if models else "(endpoint returned no models)"


REGISTRY = {"dgx_specs": (dgx_specs, 'machine: "DGX Spark" | "DGX Station"'),
            "calc": (calc, "expression: arithmetic string, e.g. '128*0.9'"),
            "list_models": (list_models, "(no args)")}

PICK_PROMPT = (
    "You are an AI-Q researcher on the NeMo Agent Toolkit tool bus. Tools:\n"
    + "\n".join(f"  - {n}({sig})" for n, (_, sig) in REGISTRY.items())
    + f'\nTask: "{TASK}"\nReply with JSON only: {{"tool": "...", "args": {{...}}}}')

NAT_FOOTER = """\
  ◈ the same loop, declared instead of coded — NeMo Agent Toolkit workflow.yml:
      functions:
        dgx_specs:   {_type: <your python fn, registered>}
      llms:
        nano:        {_type: openai, base_url: %s, model_name: %s}
      workflow:      {_type: react_agent, tool_names: [dgx_specs], llm_name: nano}
    real commands (pure Python, runs on the Spark or this laptop):
      uv pip install "nvidia-nat[all]"
      nat run --config_file workflow.yml --input "%s"
    (serving: `nat serve --config_file workflow.yml --port 8001` — the default port is
     [UNCERTAIN]; verify with `nat serve --help` first, and avoid 8000 = vLLM/NIM.)"""


def _extract_json(text: str) -> dict | None:
    clean = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
    m = re.search(r"\{.*\}", clean, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:  # noqa: BLE001
        return None


def _expected() -> None:
    print("◈ [no endpoint — showing expected output]  the real run produces:\n")
    print('  ← model picks: {"tool": "dgx_specs", "args": {"machine": "DGX Spark"}}')
    print("  → EXECUTE dgx_specs('DGX Spark')")
    print("  ← OBSERVE DGX Spark: chip=NVIDIA GB10 …; memory_gb=128; power_w=240; fits_params_b=200 …")
    print("  ← final: the DGX Spark — 128 GB unified LPDDR5X at 240 W fits ~200B quantized.")
    print("\n  go REAL:  ollama pull nemotron-3-nano   (or DGX_BASE_URL / DGX_CONN=cloud)")


def main() -> None:
    print("▣ Lab 03 · Tool bus — pick → execute → observe → answer")
    print(f"  endpoint: {config.safe_base_url()} · model: {config.MODEL} · mode: {config.MODE}\n")
    print(f'» task: "{TASK}"\n')
    if config.MODE != "real":
        _expected()
        print("\n" + NAT_FOOTER % (config.BASE_URL, config.MODEL, TASK[:40] + "…"))
        return
    from openai import BadRequestError, OpenAI
    # max_retries=0: one honest 25s timeout beats three silent 25s stalls
    client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY,
                    timeout=25.0, max_retries=0)

    def ask(prompt: str, max_tokens: int) -> str:
        kw = dict(model=config.MODEL, temperature=0.0, max_tokens=max_tokens,
                  messages=[{"role": "user", "content": prompt}])
        try:    # thinking models: skip the preamble — we want the JSON pick
            r = client.chat.completions.create(
                extra_body={"reasoning_effort": "none"}, **kw)
        except BadRequestError:   # endpoint rejects the knob → plain retry
            r = client.chat.completions.create(**kw)
        msg = r.choices[0].message
        # if the endpoint ignored the knob and thought its budget away, the
        # draft in `reasoning` still often contains the JSON we asked for
        return (msg.content or "").strip() or str(getattr(msg, "reasoning", "") or "")

    try:
        pick = _extract_json(ask(PICK_PROMPT, 280))
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ tool-pick call failed ({type(e).__name__}) — check the endpoint, rerun.")
        return
    if not pick or pick.get("tool") not in REGISTRY:
        print(f"  ✗ model did not return a valid tool pick: {pick!r}")
        print("    — this is the registry/prompt mismatch NAT's typed schemas prevent.")
        return
    tool, args = pick["tool"], pick.get("args") or {}
    print(f"  ← model picks: {json.dumps(pick)}")
    print(f"  → EXECUTE {tool}({args})")
    try:
        obs = REGISTRY[tool][0](**args)
    except Exception as e:  # noqa: BLE001
        obs = f"tool error: {e}"
    print(f"  ← OBSERVE {obs[:200]}")
    final = ask(f"Observation from tool {tool}: {obs}\nUsing ONLY that observation, "
                f"answer in one sentence: {TASK}", 160)
    final = re.sub(r"<think>.*?</think>", " ", final, flags=re.S).strip()
    print(f"  ← final: {final[:300]}\n")
    print(NAT_FOOTER % (config.BASE_URL, config.MODEL, TASK[:40] + "…"))
    print("\n✓ Takeaway — a tool bus = registry + typed picks + logged executions. Every")
    print("  arrow above is an observable event; NAT gives you that for free, per call.")


if __name__ == "__main__":
    main()

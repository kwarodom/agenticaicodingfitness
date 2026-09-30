#!/usr/bin/env python3
"""Lab 04-2 · A custom tool: chiller_kpi, its ground truth, and why a tool description is a prompt.

Research tutorial L3.4. Walks week26/common/alto_ops/src/alto_ops/chiller_tool.py (register_function ·
FunctionBaseConfig · FunctionInfo), checks the 1.9 import paths, computes the ground truth twice with no LLM
(plain Python over the CSV, then NAT's own builder calling the tool), and only then asks the tool_calling_agent.
It asks the same question twice: once with the research tutorial's original tool description, once with the
course's (which says what RT means) — and checks what the model called "RT".

LLM budget: 2 agent questions ≈ 4 laptop LLM calls.
Run: .venv/bin/python week26/04_nat_claws/labs/lab04_2_custom_tool.py
"""
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clawkit import NAT_PY, WEEK, banner, laptop_models, note, ok, result, step, table, warn  # noqa: E402
from natkit import (CONFIGS, CSV, LAPTOP_YML, TOOLS, error_line, nat, nat_with_tools, rel,  # noqa: E402
                    workflow_result)

banner("Lab 04-2 · a custom tool: chiller_kpi", "laptop NAT 1.9 for real · ground truth first, then the agent",
       status=False)
TOOL_SRC = WEEK / "common" / "alto_ops" / "src" / "alto_ops" / "chiller_tool.py"
DIRECT = TOOLS / "nat_direct.py"

step(1, "read the tool — three NAT pieces make a function")
src = TOOL_SRC.read_text(encoding="utf-8").splitlines()
marks = ("from nat", "class ChillerKpiConfig", "csv_path:", "kw_per_rt_alarm:", "@register_function",
         "async def chiller_kpi", "async def _kpi", "yield FunctionInfo", "description=")
print(f"│ {rel(TOOL_SRC)}")
for i, ln in enumerate(src, 1):
    if any(m in ln for m in marks):
        print(f"│ {i:3d}  {ln.rstrip()}")
table([["FunctionBaseConfig subclass, name=\"chiller_kpi\"", "the YAML `_type:` and its typed fields (csv_path, kw_per_rt_alarm)"],
       ["@register_function(config_type=…)", "registers the type when the module is imported"],
       ["async generator: … yield FunctionInfo.from_fn(_kpi, description=…)", "the callable the agent gets + its description"],
       ["pyproject entry point nat.components → alto_ops.register", "how `nat` finds the package without an import"]],
      ["piece", "what it does"])

step(2, "the import paths on NAT 1.9 — the tutorial says 'confirm on your version'")
nat(["imports"], prefix=[NAT_PY, DIRECT], show=f"week26/.venv-nat/bin/python {rel(DIRECT)} imports")
note("Both styles work on 1.9.0: the long paths (the tutorial, chiller_tool.py) and the `nat.plugin_api` facade that "
     "`nat workflow create` now generates. They are the same objects.")

step(3, "ground truth #1 — plain Python over the CSV (no NAT, no LLM)")
rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
truth = {}
for h in (6, 24):
    win = rows[-h * 4:]
    kw = sum(float(r["kw"]) for r in win) / len(win)
    rt = sum(float(r["rt"]) for r in win) / len(win)
    truth[h] = kw / rt
    print(f"│ last {h:2d} h · {len(win):3d} rows · avg kW {kw:6.1f} · avg RT {rt:6.1f} · kW/RT {kw / rt:.3f}")
note(f"{len(rows)} rows of SYNTHETIC 15-minute data in {rel(CSV)}; the last 6 h are degraded on purpose.")

step(4, "ground truth #2 — NAT's builder calls the registered tool directly (no LLM)")
r, _ = nat(["tool", LAPTOP_YML, "chiller_kpi", '{"hours": 6}', '{"hours": 24}'], prefix=[NAT_PY, DIRECT],
           show=f"week26/.venv-nat/bin/python {rel(DIRECT)} tool {rel(LAPTOP_YML)} chiller_kpi "
                "'{\"hours\": 6}' '{\"hours\": 24}'")
got = dict((int(h), float(v)) for h, v in re.findall(r"window=(\d+)h .*?kw_per_rt=([\d.]+)", r.out))
if got.get(6) is not None and abs(got[6] - truth[6]) < 0.001:
    ok(f"the tool agrees with plain Python: 6 h kW/RT = {got[6]:.3f} (alarm threshold in the YAML: 0.80) · "
       f"24 h = {got.get(24, float('nan')):.3f}")
else:
    warn(f"tool {got} ≠ plain Python {truth} — check csv_path in {rel(LAPTOP_YML)}")

if not any(m.startswith("nemotron-3-nano") for m in laptop_models()):
    warn("no nemotron-3-nano on this laptop's Ollama — skipping the two agent runs (`ollama pull nemotron-3-nano`).")
    sys.exit(0)

QUESTION = "Is the chiller plant efficient over the last 6 hours?"
V0 = "Average chiller plant kW, RT and kW/RT over the last N hours; flags efficiency alarms."
step(5, "the agent, with the research tutorial's ORIGINAL description (tools/chiller_v0.py)")
print(f"│ description: {V0}")
r0, t0 = nat_with_tools(["run", "--config_file", CONFIGS / "workflow.v0.yml", "--input", QUESTION], ["chiller_v0"],
                        show=f"nat run --config_file {rel(CONFIGS / 'workflow.v0.yml')} --input '{QUESTION}'   "
                             "(+ import tools/chiller_v0.py)", log="lab04_2_v0.log")
a0 = workflow_result(r0.out) or error_line(r0.out)
print(f"· ANSWER  {a0}")
print(f"◆ LAPTOP STAND-IN · nemotron-3-nano · tool_calling_agent · {t0:.1f}s wall")

step(6, "the agent, with the course's description (adds 'RT (refrigeration tons of cooling)')")
r1, t1 = nat(["run", "--config_file", LAPTOP_YML, "--input", QUESTION], log="lab04_2_v1.log")
a1 = workflow_result(r1.out) or error_line(r1.out)
print(f"· ANSWER  {a1}")
print(f"◆ LAPTOP STAND-IN · nemotron-3-nano · tool_calling_agent · {t1:.1f}s wall")

step(7, "compare: did it call the tool right, get the number right, and name RT right?")


def judge(ans: str) -> tuple[str, str, str]:
    low = ans.lower()
    called = "✓ 0.901" if "0.901" in ans else "✕ number missing"
    if "refrigerant temperature" in low or "return temperature" in low:
        rt = "✕ 'refrigerant/return temperature'"
    elif "refrigeration ton" in low or "tons of cooling" in low or "tons of refrigeration" in low:
        rt = "✓ refrigeration tons"
    else:
        rt = "— not named"
    alarm = "✓ alarm" if ("alarm" in low or "inefficien" in low or "not efficient" in low) else "✕ no alarm"
    return called, rt, alarm


rows_out = []
for name, ans, secs in (("tutorial description", a0, t0), ("course description", a1, t1)):
    rows_out.append([name, *judge(ans), f"{secs:.1f}"])
table(rows_out, ["tool description", "kW/RT in answer", "what RT means", "verdict", "s (LAPTOP STAND-IN)"])
if rows_out[0][2].startswith("✕") and rows_out[1][2].startswith("✓"):
    ok("same model, same data, same number — only the description changed, and so did the meaning of RT")
note("The model only sees the tool's name, description and schema. `avg_rt=525.9` is ambiguous to it; one "
     "clause in the description fixes that. This build hit exactly this: without it, nemotron-3-nano read RT as "
     "'refrigerant temperature'. The answer is stochastic — rerun and it may differ; the fix is not.")
result("A tool is config + code + a description. Test the code without the LLM; treat the description as a prompt.")

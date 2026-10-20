#!/usr/bin/env python3
"""Lab 14-4 · Measure the agent with `nat eval`: three questions, one offline evaluator.

configs/hotel_eval.yml inherits the agent from hotel_agent.yml (`base:`) and adds an `eval:` section:
a three-row JSON dataset (configs/eval_dataset.json) and the evaluator `mentions_expected` from the
hotel_ops_nat package (plugged in with `_type: langsmith_custom`). It scores each answer by the fraction
of expected facts it contains — deterministic, no LLM judge. The lab runs the eval against the same
model server lab 14-2 picks and prints one row per question.

Run: .venv/bin/python week25/14_nat_agents/labs/lab14_4_eval.py
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import (LAPTOP_OLLAMA, WEEK, banner, laptop_models, mode, models, note, pick_laptop_model,  # noqa: E402
                      result, step, table, url, warn)

MOD = Path(__file__).resolve().parents[1]
NAT = Path(os.environ.get("NAT_BIN", WEEK / ".venv-nat" / "bin" / "nat"))
ANSI = re.compile(r"\x1b\[[0-9;]*m")
OUT = MOD / ".runs" / "eval"


def pick_llm() -> tuple[str, str, str]:
    if mode() == "live":
        for kind in ("vllm", "ollama"):
            ids = models(url(kind))
            if ids:
                return url(kind), os.environ.get("HOTEL_LLM_MODEL") or ids[0], f"spark-{kind}"
    if laptop_models():                     # the 💻 stand-in follows its own toggle, not DRY
        return LAPTOP_OLLAMA, pick_laptop_model(["nemotron-3-nano", "gemma4:12b"]), "laptop"
    return "", "", "dry"


banner("Lab 14-4 · nat eval — does the agent get the facts right?",
       "3 questions · offline evaluator (no LLM judge) · results in week25/14_nat_agents/.runs/eval/")
if not NAT.exists():
    warn(f"no NAT CLI at {NAT} — run lab 14-1 first")
    sys.exit(0)

step(1, "the dataset — `answer` lists the facts a good reply must contain, separated by |")
rows = json.loads((MOD / "configs" / "eval_dataset.json").read_text())
table([[r["id"], r["question"][:60], r["answer"]] for r in rows], ["id", "question", "expected facts"])

step(2, "nat eval (runs the workflow once per row, then every evaluator on every answer)")
base, model, src = pick_llm()
print(f"$ cd week25/14_nat_agents && HOTEL_LLM_BASE_URL={base or '<url>'} HOTEL_LLM_MODEL={model or '<model>'} "
      f"../.venv-nat/bin/nat eval --config_file configs/hotel_eval.yml")
if src == "dry":
    print("◈ EXAMPLE — a LAPTOP STAND-IN eval captured on this Mac while the module was written (not your machine):")
    print("│ hot_808    1.0  found ['808', 'mt-'] · missing []\n│ ok_1203    1.0  found ['1203', '23.5'] · missing []\n"
          "│ cold_1510  1.0  found ['1510', '20.1'] · missing []")
    result("No model server answered — start Ollama or connect a Spark and rerun.")
    sys.exit(0)
print(f"◆ model: {model} · {'LAPTOP STAND-IN (Ollama on this Mac)' if src == 'laptop' else src}")
env = {**os.environ, "PYTHONWARNINGS": "ignore", "NO_COLOR": "1", "HOTEL_LLM_BASE_URL": base, "HOTEL_LLM_MODEL": model,
       "HOTEL_TRACE_FILE": str(MOD / ".runs" / "eval_trace.jsonl"), "HOTEL_TICKET_LOG": str(MOD / ".runs" / "tickets.jsonl")}
if src == "spark-vllm":
    # hotel_eval.yml inherits hotel_agent.yml; for vLLM we inherit the Spark variant instead
    spark_eval = MOD / ".runs" / "hotel_eval_spark.yml"
    spark_eval.write_text((MOD / "configs" / "hotel_eval.yml").read_text().replace(
        "base: hotel_agent.yml", f"base: {MOD / 'configs' / 'hotel_agent_spark.yml'}"))
    cfg_file = str(spark_eval)
else:
    cfg_file = "configs/hotel_eval.yml"
p = subprocess.run([str(NAT), "eval", "--config_file", cfg_file], cwd=MOD, env=env, capture_output=True, text=True,
                   timeout=840)
log, out = ANSI.sub("", p.stdout + p.stderr), ANSI.sub("", p.stdout)
summary = out[out.find("=== EVALUATION SUMMARY ==="):] if "=== EVALUATION SUMMARY ===" in out else ""
print("\n".join(ln for ln in summary.splitlines() if ln.strip()) or "\n".join(log.splitlines()[-10:]))
if p.returncode != 0:
    warn(f"nat eval exited {p.returncode}")
    sys.exit(1)

step(3, "per question: what the agent said, and what the evaluator found")
scores = {i["id"]: i for i in json.loads((OUT / "mentions_expected_output.json").read_text())["eval_output_items"]}
wf = json.loads((OUT / "workflow_output.json").read_text())
table([[w["id"], f"{scores[w['id']]['score']:.2f}", str(w.get("generated_answer", ""))[:80]] for w in wf],
      ["id", "score", "generated answer"])
for w in wf:
    print(f"◆ {w['id']}: {scores[w['id']]['reasoning'].get('comment', '')}")
note("NAT also saved config_effective.yml (the merged config it actually ran) next to the results — the record "
     "you need to compare a fine-tuned model (Module 13) with the base model on the same questions.")
if src == "laptop":
    note("LAPTOP STAND-IN: the scores are real for this model; rerun on the Spark to score the Spark's model.")

result("An agent eval is a dataset, a workflow and an evaluator. Change one thing (model, prompt, tool) and rerun.")

#!/usr/bin/env python3
"""Lab 06-2 · Which engine? Turn a workload's needs into an engine choice, with the playbook fact behind it.

Offline, no Spark needed. Each engine gets a short profile built only from what the DGX Spark playbooks
say (support tables, launch notes, "pending validation" warnings). A workload is a set of needs; the lab
scores every engine against it, rules out engines that cannot meet a hard need, and prints why.

Run: .venv/bin/python week25/06_sglang_trtllm_nim/labs/lab02_which_engine.py
     .venv/bin/python week25/06_sglang_trtllm_nim/labs/lab02_which_engine.py --need tools,two-sparks
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, note, result, step, table  # noqa: E402

# need → {engine: (score 0-2, playbook fact)}. 0 = the playbook gives no path → ruled out for a hard need.
FACTS = {
    "tools": {
        "vLLM":         (2, "agent-ready Qwen3.6-35B-A3B recipe with --tool-call-parser qwen3_xml (vLLM playbook)"),
        "SGLang":       (1, "OpenAI API, but the playbook removed its Agent-ready tab (07/31/2026)"),
        "TensorRT-LLM": (1, "--tool_parser in Nemotron recipes; agent-ready Qwen3.6 launch settings 'pending validation'"),
        "NIM":          (0, "the NIM playbook shows plain chat only"),
    },
    "json": {
        "vLLM":         (1, "OpenAI API; structured output is not covered in the Spark playbook"),
        "SGLang":       (2, "xGrammar structured output; Step 7 sends a json_schema response_format"),
        "TensorRT-LLM": (1, "not covered in the Spark playbook"),
        "NIM":          (1, "not covered in the Spark playbook"),
    },
    "prefix": {
        "vLLM":         (2, "--enable-prefix-caching in the agent-ready and Nemotron recipes"),
        "SGLang":       (2, "RadixAttention reuses shared prefixes; --enable-cache-report shows cached_tokens"),
        "TensorRT-LLM": (1, "Nemotron recipes set enable_block_reuse: false (Mamba state is not prefix-cacheable)"),
        "NIM":          (1, "not covered in the Spark playbook"),
    },
    "two-sparks": {
        "vLLM":         (2, "Ray + --tensor-parallel-size 2, nvcr.io/nvidia/vllm:26.05-py3 (Multi-node tab)"),
        "SGLang":       (0, "support matrix: DGX Spark multi-node '—'"),
        "TensorRT-LLM": (2, "OpenMPI + trtllm-serve --tp_size, release:1.3.0rc5 (Multi-node tab)"),
        "NIM":          (0, "support matrix: multi-node '—'"),
    },
    "nemotron-super": {
        "vLLM":         (2, "vllm/vllm-openai:cu130-nightly recipe, served as nemotron-3-super on :8000"),
        "SGLang":       (0, "not in the Nemotron playbook"),
        "TensorRT-LLM": (2, "release:1.3.0rc9 recipe on :8123"),
        "NIM":          (0, "not in the Nemotron playbook"),
    },
    "least-setup": {
        "vLLM":         (1, "one docker run; you choose model, flags, parsers"),
        "SGLang":       (1, "one docker run; first launch captures CUDA graphs (~10–15 min for Qwen3-8B)"),
        "TensorRT-LLM": (1, "a YAML of extra LLM API options per model; the playbook rates its risk Medium"),
        "NIM":          (2, "a prebuilt container per model with the model inside; one docker run with an NGC key"),
    },
    "latency": {
        "vLLM":         (1, "tuned recipes (MTP speculative decoding for Qwen3.6)"),
        "SGLang":       (1, "tuned for shared-prefix workloads"),
        "TensorRT-LLM": (2, "playbook tagline: 'Lower-latency responses and higher throughput for the largest models'"),
        "NIM":          (1, "'GPU-optimized model containers' — which engine runs inside is not stated"),
    },
}
ENGINES = ["vLLM", "SGLang", "TensorRT-LLM", "NIM"]
PORT = {"vLLM": 8000, "SGLang": 30000, "TensorRT-LLM": 8355, "NIM": 8000}

WORKLOADS = [
    ("Hotel concierge agent (Module 14): tool calls, long multi-turn", ["tools", "prefix"]),
    ("RAG answers that must be valid JSON for a dashboard", ["json", "prefix"]),
    ("Llama 3.3 70B at bf16 — bigger than one Spark", ["two-sparks"]),
    ("Nemotron 3 Super for reasoning, one Spark", ["nemotron-super"]),
    ("A demo tomorrow, supported container, no tuning", ["least-setup"]),
    ("Lowest latency for one validated model", ["latency"]),
]


def choose(needs: list[str]):
    scores = {}
    for e in ENGINES:
        s = [FACTS[n][e][0] for n in needs]
        scores[e] = None if 0 in s else sum(s)         # a 0 on any need rules the engine out
    ok = {e: s for e, s in scores.items() if s is not None}
    best = max(ok, key=lambda e: (ok[e], -ENGINES.index(e))) if ok else None
    return best, scores


ap = argparse.ArgumentParser()
ap.add_argument("--need", default="", help=f"comma list from: {', '.join(FACTS)}")
args = ap.parse_args()

banner("Lab 06-2 · which engine?", "needs → engine, each reason taken from a DGX Spark playbook (words in quotes are verbatim)", status=False)

step(1, "the scorecard — 2 = the playbook shows a validated path · 1 = possible, not shown · 0 = no path")
table([[n] + [FACTS[n][e][0] for e in ENGINES] for n in FACTS], ["need"] + ENGINES)

step(2, "six workloads")
rows = []
for title, needs in WORKLOADS:
    best, scores = choose(needs)
    ruled = [e for e, s in scores.items() if s is None]
    rows.append([title, "+".join(needs), f"{best} :{PORT[best]}" if best else "—", ", ".join(ruled) or "—"])
table(rows, ["workload", "needs", "pick", "ruled out"])

step(3, "why — the playbook fact behind each pick")
for title, needs in WORKLOADS:
    best, _ = choose(needs)
    print(f"│ {title}")
    for n in needs:
        print(f"│    {n:15s} {best}: {FACTS[n][best][1]}")

if args.need:
    needs = [n.strip() for n in args.need.split(",") if n.strip() in FACTS]
    step(4, f"your workload: {' + '.join(needs) or '(no known needs)'}")
    if needs:
        best, scores = choose(needs)
        table([[e, "ruled out" if s is None else s] for e, s in scores.items()], ["engine", "score"])
        print(f"═ pick: {best} on :{PORT[best]}" if best else "═ no engine meets every need — relax one")
note("Ties go to the engine listed first (vLLM) — on a tie, measure both with lab 06-1.")
note("The scorecard is what the playbooks SHOW, not what each engine can do in general. A 1 means "
     "'try it and measure' (lab 06-1), not 'unsupported'.")
result("Start from the need that rules engines out (two Sparks, a specific model), then measure the survivors "
       "with the same prompts. Most agent work on one Spark lands on vLLM; JSON-heavy RAG on SGLang.")

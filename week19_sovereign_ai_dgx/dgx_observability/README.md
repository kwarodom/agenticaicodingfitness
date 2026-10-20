# ▣ Week 19 · App 3 — Observability on a DGX (Phoenix + NeMo Agent Toolkit)

An **interactive, explainable web app + runnable demos** for making a *sovereign*
agent observable: trace every step with OpenTelemetry, view it the way **Arize
Phoenix** does, evaluate the traces with your own model, and rebuild the agent as
a **NeMo Agent Toolkit** YAML workflow — all on the DGX, $0, nothing leaving the box.

```
◆ agent run = TRACE → ◇ LLM + ▸ TOOL spans → ◆ metrics → ✓ evals → Phoenix on the DGX
                              ⛔ traces (which contain prompts) never leave the box
```

## Two modes — auto-detected

| Mode | Agent loop | Spans | Phoenix |
|------|-----------|-------|---------|
| **REAL** | genuine tool-calling on the live DGX model | wrap real latency + token usage | exports for real if `arize-phoenix` is running |
| **SIM**  | scripted loop | rendered from the simulation (same shape) | tree rendered locally |

The tracer (`tracer.py`) is **dependency-free** — a teaching-sized version of
`openinference-instrumentation` + OpenTelemetry — so the whole app runs with no
GPU and no Phoenix install, yet the span **shape** is exactly what real Phoenix ingests.

---

## Quick start

```bash
uv pip install -r week19/dgx_observability/requirements.txt

# optional: a live model makes the agent real
ollama run qwen3.6:35b-a3b-q8_0      # gemma4:12b on a Mac

# optional: a real Phoenix UI on your DGX (traces stay on-prem)
pip install arize-phoenix openinference-instrumentation-openai
python -m phoenix.server.main serve  # → http://localhost:6006

# the interactive web app
.venv/bin/python week19/dgx_observability/tutorial_server.py
# → open http://127.0.0.1:8094
```

Env overrides: `DGX_BASE_URL`, `DGX_MODEL`, `DGX_MODE=sim|real`, `PHOENIX_ENDPOINT`.

---

## Layout

```
week19/dgx_observability/
├── README.md · requirements.txt · config.py
├── tracer.py           → dependency-free OTel-shaped span tracer + Phoenix render
├── obsview.py          → the traced HVAC agent (real tool-calling or simulated)
├── tutorial_server.py  → FastAPI control plane
├── static/guide.html   → clickable, streaming span-tree UI
└── demos/
    ├── step01_traced_agent.py       Ch 2 · run + trace the agent (span tree)
    ├── step02_phoenix.py            Ch 3 · Phoenix + the gen_ai.* conventions
    ├── step03_metrics.py            Ch 4 · latency waterfall, tokens, GPU correlation
    ├── step04_eval_traces.py        Ch 5 · LLM-as-judge on spans (sovereign eval)
    ├── step05_nat_intro.py          Ch 6 · NAT — register a tool (real module)
    ├── step06_nat_workflow.py       Ch 7 · NAT — YAML workflow on your DGX + HITL
    └── step07_nat_observability.py  Ch 8 · NAT telemetry → Phoenix + prod checklist
```

---

## The 8 chapters

| Ch | Demo | The one thing to notice |
|----|------|--------------------------|
| 1  | *(concept)* | Tracing + metrics + evals; on a DGX the **traces** stay on-prem too. |
| 2  | `step01` | One agent run = a **trace** of nested **spans** (AGENT → LLM → TOOL). |
| 3  | `step02` | Phoenix is **self-hosted** → runs on the DGX; spans follow **gen_ai.\***. |
| 4  | `step03` | Watch **latency + tokens + LLM-calls**, correlate with **GPU util**. |
| 5  | `step04` | Evals catch a **mis-triaged dispatch**; the judge is **your own model**. |
| 6  | `step05` | A NAT tool = `FunctionBaseConfig` + `@register_function` + `FunctionInfo`. |
| 7  | `step06` | A NAT **YAML workflow**; the only sovereignty knob is `base_url` → your DGX. |
| 8  | `step07` | A `telemetry` block streams NAT spans to **Phoenix on the same box**. |

---

## Where this sits in Week 19

```
App 1 sovereign_dgx → run/serve   ·   App 2 dgx_finetune → adapt to your domain
App 3 (THIS) dgx_observability → SEE, MEASURE, JUDGE the agent (Phoenix + NAT)
App 4 self_evolving_agent_v2 → the Week 18 agent, now driven by your DGX model
```

Observability is the prerequisite for trusting an agent in production — and doing
it on a DGX means even your traces and your grading stay sovereign. App 3 ties the
served model (App 1) and the fine-tuned weights (App 2) into a single, watchable
agent built the NVIDIA way (NeMo Agent Toolkit).

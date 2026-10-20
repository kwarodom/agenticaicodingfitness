---
name: nemo-agent-toolkit
description: "Teach building production multi-agent systems with the NVIDIA NeMo Agent Toolkit (NAT) — the Week 16 'factory floor' that moves you from hand-rolled agent loops to a config-driven framework that builds, connects, evaluates, profiles, and deploys agentic systems. Covers the NAT building blocks (tools, agents, workflows, observability), registering custom tools via FunctionBaseConfig + the @register_function decorator and FunctionInfo, tool-calling / ReAct agents, RAG tools via LlamaIndex, supervisor → specialist multi-agent orchestration (LangGraph StateGraph under the hood) with human-in-the-loop approval, YAML workflow composition that separates agent logic from instantiation, the Builder/LLMFrameworkEnum cross-framework abstraction (wraps LangChain + LlamaIndex), and NVIDIA NIM model inference. Use when someone asks 'how do I use NVIDIA NeMo Agent Toolkit / NAT?', 'how do I register a tool / build a workflow with NAT?', mentions FunctionBaseConfig / register_function / nat workflow create / NIM, wants a config-driven production agent framework, or is reviewing Week 16."
when_to_use: "Learner wants to build agents with the NVIDIA NeMo Agent Toolkit specifically — registering custom tools, composing tools+agents into YAML workflows, supervisor/specialist orchestration with HITL, RAG tools, or running on NVIDIA NIM — or is adopting a config-driven production framework after hand-rolling agents in earlier weeks, or is catching up on Week 16."
---

# NeMo Agent Toolkit — Config-Driven Production Agents (Week 16)

> **The one idea:** Weeks 2–15 you *hand-wrote* the agent loop, the tool wiring, and the orchestration. The **NVIDIA NeMo Agent Toolkit (NAT)** is the factory floor: you **register** tools and agents once, **compose** them in a YAML workflow, and the framework handles instantiation, the ReAct loop, tool-binding, multi-agent routing, and observability. It's the jump from "I built *an* agent" to "I can compose, deploy, and monitor *a system of* agents."

> ✅ **Grounded & runnable** in `week16_nemo_agent_toolkit/` — two progressive Jupyter notebooks from NVIDIA's 8-tutorial NAT series (Hello World → Tools → MCP → Multi-Agent → Observability → Optimizer). Running the live notebooks needs the NAT package + an NVIDIA NIM endpoint (an `NVIDIA_API_KEY`); the **guided lab below is a $0 `MockLLM`** that teaches the registration + orchestration *shape* with no install.

NVIDIA's framing (`week16_nemo_agent_toolkit/README.md`):
> *"…use the NVIDIA NeMo Agent Toolkit to build, connect, evaluate, profile, and deploy an agentic system… the building blocks that make up the agentic system, including **tools, agents, workflows, and observability**."*

---

## Part A — The four building blocks

NAT is **framework-agnostic** by design: a `Builder` constructs components and an `LLMFrameworkEnum` lets one workflow mix LangChain, LlamaIndex, and custom code. The four blocks:

| Block | What it is | NAT primitive |
|---|---|---|
| **Tool** | A function the agent can call (data query, RAG lookup, an API) | `FunctionBaseConfig` + `@register_function` → `FunctionInfo` |
| **Agent** | A tool-calling / ReAct planner that reasons, then calls tools | a workflow `type` (e.g. `react_agent`, `tool_calling_agent`) |
| **Workflow** | The composition — which agents, which tools, which LLMs | a **YAML** config file |
| **Observability** | Tracing, evaluation, profiling of the running system | NAT's telemetry/eval/profiler layer |

The discipline that makes this "production": **logic and instantiation are separated.** Your tool/agent code is registered once; *which* LLM, *which* tools, and *which* hyperparameters wire together lives in YAML a non-coder can tweak.

---

## Part B — Registering a tool (the core move)

A NAT tool is a config class + a registered factory. The `@register_function` decorator publishes it so any workflow can reference it by name:

```python
# week16_nemo_agent_toolkit/adding_tools_to_agents.ipynb (shape)
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

class SalesAggregatorConfig(FunctionBaseConfig, name="sales_aggregator"):
    """Config = the tool's public schema (name + any params)."""
    csv_path: str = "sales.csv"

@register_function(config_type=SalesAggregatorConfig)
async def sales_aggregator(config: SalesAggregatorConfig, builder: Builder):
    async def _aggregate(region: str) -> str:
        # ... read config.csv_path, sum sales for `region` ...
        return f"Total sales for {region}: ..."
    # FunctionInfo carries the schema + the callable the agent will invoke
    yield FunctionInfo.from_fn(_aggregate, description="Aggregate sales by region.")
```

A **RAG tool** is just another registered function — NAT wraps LlamaIndex (`VectorStoreIndex`, `SimpleDirectoryReader`, `SentenceSplitter`) so retrieval becomes one more callable the agent can pick. Tools registered, you scaffold and run a workflow from the CLI:

```bash
nat workflow create my_agent      # scaffold
nat run --config_file workflow.yml --input "What were Q3 sales in APAC?"
```

```yaml
# workflow.yml — composition, not code
functions:
  sales: { _type: sales_aggregator, csv_path: data/sales.csv }
  product_docs: { _type: llamaindex_rag, docs_dir: data/specs }
llms:
  default: { _type: nim, model_name: meta/llama-3.3-70b-instruct }
workflow:
  _type: react_agent
  llm: default
  tools: [sales, product_docs]
```

> 📁 `week16_nemo_agent_toolkit/adding_tools_to_agents.ipynb` — builds custom tools (sales aggregation, outlier detection) + a LlamaIndex RAG tool, registers them, and runs via `nat workflow create` + YAML.

---

## Part C — Multi-agent orchestration (supervisor → specialists)

The same registration model scales to a **supervisor that routes to specialist sub-agents** — this is the `multi-agent-systems` supervisor pattern, expressed in NAT. Under the hood NAT drives a **LangGraph `StateGraph`** (with `ToolNode`s and conditional edges); you describe the team, not the plumbing.

```
Supervisor (ReAct: reason → pick a sub-agent → observe → repeat)
   ├─▶ Data Analysis Agent   (sales tools)
   ├─▶ Product RAG Agent      (LlamaIndex retrieval)
   └─▶ Visualization Agent    (chart tool)
        … with a Human-in-the-Loop approval gate before a costly/irreversible action
```

Key ideas the notebook demonstrates:
- **Multi-LLM by role** — a strong model for the supervisor, cheaper models for specialists (the 3-tier cascade from `models-and-patterns`, made declarative).
- **HITL approval** — interactive prompts (NAT's interactive data models) pause for a human before the agent commits — the same gate as `production-and-observability`'s `interrupt()`, here as config.
- **Graph summarization** — the orchestrator condenses sub-agent outputs instead of carrying every transcript (the artifact-not-transcript discipline from `multi-agent-systems`).

NVIDIA's framing (`week16_nemo_agent_toolkit/multi_agent_orchestration.ipynb`):
> *"…how an orchestration agent can call tools and sub-agents to facilitate complex tasks… notably runtime and token efficiency."*

> 📁 `week16_nemo_agent_toolkit/multi_agent_orchestration.ipynb` — a supervisor routing to 3 specialists with HITL and graph summarization.

---

## Part D — Where NAT fits among the frameworks

You've now seen four ways to build multi-agent systems. NAT's distinguishing bet is **config-driven composition + built-in observability/eval/profiling**, with a Builder that *wraps* other frameworks rather than replacing them.

| Framework | Mental model | NAT's relationship |
|---|---|---|
| **CrewAI** (W9/W15) | Agents + Tasks + Crew | An alternative; role-based |
| **LangGraph** (W9/W10) | Nodes + shared state + edges | NAT **uses it** under multi-agent workflows |
| **LlamaIndex** (W8/W15) | Indexes + retrievers | NAT **wraps it** as RAG tools |
| **Claude Agent SDK** (W10) | Subagents + the Task tool | An alternative; code-first |
| **NeMo Agent Toolkit** (W16) | Register → compose in YAML → observe | The config-driven, framework-agnostic factory |

**Reach for NAT when** you want a declarative, ops-friendly path to production multi-agent on the NVIDIA stack (NIM models), with evaluation and profiling as first-class. Pick by the shape of your problem — see `models-and-patterns` for the model/pattern half of the decision, and `multi-agent-systems` for the orchestration patterns NAT is expressing.

---

## 🧪 Guided lab (offer this): *register two tools, route between them*

**You'll need:** nothing — no NAT install, no NIM key. A `MockLLM` + a tiny registry reproduce NAT's *register → compose → route* shape at **$0**. (When you're ready for the real thing, run `week16_nemo_agent_toolkit/adding_tools_to_agents.ipynb` with the NAT package + an `NVIDIA_API_KEY`.)

### Warm-up (5–10 min, pass/fail)
Answer out loud:
1. In NAT, what separates a tool's **logic** from its **instantiation**? (What lives in code vs. in YAML?)
2. What does `@register_function` buy you that a plain Python function doesn't?
3. When a supervisor routes to specialists, which earlier-week *pattern* is that, and what keeps its token cost down?
> ✅ **Pass** = all three (code = the registered function; YAML = which tools/LLMs/params wire together · a name other workflows can reference by config · Supervisor-Worker; summarize/store artifacts, not transcripts).

### Skill Drill (15–30 min, runnable, $0)
Mimic NAT's registry + a ReAct supervisor that picks a registered tool by intent.

```python
# nat_shape_drill.py — plain `python`, no install, $0.
# Reproduces NAT's register → compose → route shape.
REGISTRY = {}                                  # ← @register_function publishes here

def register_function(name, description):      # ← the decorator, simplified
    def deco(fn):
        REGISTRY[name] = {"fn": fn, "description": description}   # ← FunctionInfo
        return fn
    return deco

@register_function("sales_aggregator", "Aggregate sales by region.")
def sales(region): return f"Total sales for {region}: $1.2M"

@register_function("product_rag", "Answer product questions from the docs.")
def rag(query): return f"[from docs] {query} → spec v3, ships Q4."

class MockLLM:
    """$0 ReAct supervisor: reason → pick a registered tool → observe."""
    def route(self, user_msg):
        t = user_msg.lower()
        if any(k in t for k in ("sales", "revenue", "region")): return "sales_aggregator"
        if any(k in t for k in ("spec", "product", "feature")): return "product_rag"
        return None

# --- the "workflow": compose registered tools, route by the supervisor ---
def run(user_msg, llm=MockLLM()):
    tool_name = llm.route(user_msg)            # supervisor reasons
    if tool_name not in REGISTRY:
        return {"state": "failed", "reason": "no tool matched"}
    tool = REGISTRY[tool_name]
    arg = user_msg.split()[-1].strip("?")      # toy arg extraction
    return {"state": "completed", "tool": tool_name, "result": tool["fn"](arg)}

if __name__ == "__main__":
    import json
    print("registered tools:", list(REGISTRY))      # proves the registry works
    for q in ["What were sales in APAC?", "Tell me the product spec", "What's the weather?"]:
        print(q, "->", json.dumps(run(q)))
```

**Extend it (pick any two):** (a) add a **HITL gate** — before returning, print the chosen tool+arg and require `input("approve? ")` == `"y"`; (b) add a 3rd `visualization` tool and a supervisor branch; (c) add a `last_message` vs `full_history` toggle that either returns just the result or the whole reasoning trace (NAT's `output_mode` idea).

### Weighted evaluation criteria (pass = **4 / 5**)
| # | Criterion | Weight |
|---|---|---|
| 1 | Both tools **register** and appear in the registry (the decorator works) | ●●● |
| 2 | The supervisor **routes each query to the correct registered tool** (and `failed`s the no-match) | ●●● |
| 3 | The drill runs to `state: completed` at $0, no key | ●● |
| 4 | Learner can name what belongs in **YAML vs. code** (composition vs. logic) | ●● |
| 5 | Learner can place NAT among the frameworks (config-driven; wraps LangGraph/LlamaIndex) and name the supervisor pattern | ● |

**Pass threshold: 4 of 5.** Criteria 1 and 2 are the heart of NAT — register once, compose declaratively, let the supervisor route. Close on the judgment call: *"NAT doesn't replace your patterns — it makes them declarative and observable. The pattern (Supervisor-Worker, RAG tool) is the decision; YAML is just how you spell it on the NVIDIA stack."*

# Week 10 — LangGraph Orchestration (Hands-On)

Part of the [Agentic Coding Fitness](https://agentic-coding-fitness.vercel.app) curriculum.

## What you'll build

A **Support Ticket Routing System** using LangGraph's state-graph orchestration. Five progressive notebooks take you from your first graph to a production-grade multi-agent system with human-in-the-loop, durable checkpointing, LangSmith observability, and a Claude Agent SDK hybrid.

## Why LangGraph

LangGraph powers production agent systems at Uber, JPMorgan, BlackRock, Cisco, LinkedIn, and Klarna. 90M monthly downloads across the LangChain + LangGraph ecosystem. Gartner predicts 40% of enterprise apps will embed agentic capabilities by end-2026 (up from 12%). If you build agent systems in 2026, you read LangGraph — even if you don't always write it.

## Pre-work (~5 minutes)

1. Python 3.11+
2. Clone & install:
   ```bash
   git clone <this-repo>
   cd w10-langgraph-hands-on
   uv venv && source .venv/bin/activate
   uv pip install -r requirements.txt
   python verify_setup.py
   ```
3. API keys (put in `.env` or export in shell):
   - `OPENROUTER_API_KEY`: **required**. Free email signup at [openrouter.ai/keys](https://openrouter.ai/keys). No credit card. Unlocks GPT-OSS 120B (free), Qwen3 Coder (free), DeepSeek R1 (free), GLM 4.6 (free), Llama 3.3 70B (free). Free tier is 20 RPM / 50 RPD per model; adding $10 credit lifts to 1,000 RPD per model.
   - `ANTHROPIC_API_KEY`: optional, only Ex 5. [console.anthropic.com](https://console.anthropic.com).
   - `GOOGLE_API_KEY`: optional, Gemini swap. Free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Note: Gemini free tier is only 20 RPD per model (Apr 2026 post-cuts), so Gemini is not the default here — use OpenRouter.
   - `LANGSMITH_API_KEY`: optional, Ex 4. Free at [smith.langchain.com](https://smith.langchain.com).

## Notebooks

| # | File | Topic | What you build |
|---|------|-------|---------------|
| 01 | `notebooks/01_hello_graph.py` | Hello StateGraph | Typed state + classify node + router |
| 02 | `notebooks/02_supervisor.py` | Tool-calling supervisor | 3-specialist routing (TECHNICAL / BILLING / GENERAL) |
| 03 | `notebooks/03_checkpointing.py` | Checkpointing + HITL | SQLite persistence + `interrupt()` for human approval |
| 04 | `notebooks/04_langsmith.py` | LangSmith tracing | Wire observability, inspect traces, tune a prompt |
| 05 | `notebooks/05_hybrid_sdk.py` | Claude Agent SDK hybrid | Agent Skills + subagent pattern + Managed Agents preview |

Notebook files use [Jupytext `py:percent`](https://jupytext.readthedocs.io/en/latest/formats-scripts.html) format — VS Code / Cursor / PyCharm open them as notebooks natively. To convert to `.ipynb`:

```bash
uv pip install jupytext
jupytext --to notebook notebooks/*.py
```

Or just run them as plain Python:

```bash
python notebooks/01_hello_graph.py
```

## Falling behind? Catch up instantly

Each exercise has a reference solution in the `solutions/` folder. If you get stuck mid-class, copy the matching `solutions/XX_*.py` on top of your working file and keep going.

```bash
cp solutions/01_hello_graph_solution.py notebooks/01_hello_graph.py
```

Or use git branches if the instructor has set them up:

```bash
git checkout solution-ex1
```

## Models used

- **Ex 1–4**: GPT-OSS 120B (free) via OpenRouter. OpenAI's open-weight model, fast first token, strong tool calling under throttle. Free tier: 20 RPM / 50 RPD per model. No credit card required.
- **Swap demos (any Ex)**: OpenRouter gives you multiple free alternatives via the same `ChatOpenAI` client; edit the `model=` line in notebook 01:
  - `qwen/qwen3-coder:free`: 480B MoE, agentic-tool-use tuned.
  - `deepseek/deepseek-r1-0528:free`: reasoning-heavy, slower, precise.
  - `z-ai/glm-4.6:free`: strong tool calling, GLM's latest free tier.
  - `meta-llama/llama-3.3-70b-instruct:free`: solid all-rounder.
- **Ex 5 only**: Claude Sonnet 4.6 + Claude Agent SDK.

## After class

- Complete homework in `HOMEWORK.md`.
- W11 preview: Swarm patterns.

## References

- [LangGraph docs](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangGraph multi-agent guide (April 2026 tool-calling pattern)](https://docs.langchain.com/oss/python/langchain/multi-agent)
- [Anthropic "Building Effective Agents"](https://www.anthropic.com/research/building-effective-agents)
- [Anthropic 2026 Agentic Coding Trends Report](https://resources.anthropic.com/hubfs/2026%20Agentic%20Coding%20Trends%20Report.pdf)
- [Claude Managed Agents quickstart](https://platform.claude.com/docs/en/managed-agents/quickstart)
- [Agent Skills overview](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)
- [Subagents in the SDK](https://platform.claude.com/docs/en/agent-sdk/subagents)
- [LangChain Academy (free)](https://academy.langchain.com/)

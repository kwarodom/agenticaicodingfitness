# 🏋️ Week 15 — Code Companion

Runnable Python that follows **`week15/PLUGIN_TUTORIAL.md`** step by step. Each numbered folder mirrors one skill in the Agentic Coding Fitness bootcamp plugin, so you can *do* the recap, not just read it.

```
code/
├── 01_llm_fundamentals/   → single call · streaming · chatbot memory          (W2)
├── 02_tool_use/           → calculator → +weather → your own tool             (W3)
├── 03_agent_loops/        → the reusable Agent class + an autonomous agent    (W4–5)
├── 04_mcp_and_skills/     → connect an MCP server + a sample SKILL.md         (W7)
├── 05_rag/                → "chat with your docs" from scratch                (W8)
├── 06_multi_agent/        → sequential · parallel swarm · router              (W9)
├── 07_agent_memory/       → durable memory in Neo4j + GraphRAG                (W14)
└── 08_patterns/           → reflection pattern + a model chooser              (W11)
```

## Setup (once)

```bash
# from the repo root
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r week15/code/requirements.txt
```

Create a `.env` in the **repo root** (it's gitignored — never commit real keys):

```ini
ANTHROPIC_API_KEY="sk-ant-...your key..."
# optional, only for some lessons:
OPENROUTER_API_KEY="..."
SERPER_API_KEY="..."
```

Every script calls `load_dotenv(find_dotenv())`, so it finds the repo-root `.env` no matter which folder you run from.

## How to use it

Go folder by folder, lowest number first. Inside each folder, run the scripts in order:

```bash
python week15/code/01_llm_fundamentals/01_single_call.py
python week15/code/01_llm_fundamentals/02_streaming.py
python week15/code/01_llm_fundamentals/03_chatbot_memory.py
# ...then 02_tool_use/, 03_agent_loops/, etc.
```

Each folder has its own short `README.md` telling you what to run and what to watch for.

## What needs extra setup

- **04_mcp_and_skills** → `pip install claude-agent-sdk` (already in requirements) + network access.
- **07_agent_memory** → a running Neo4j. Start one with:
  ```bash
  docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
  ```
  Browser UI: http://localhost:7474 · Bolt: bolt://localhost:7687

## Model IDs used here

- Reasoning / "the brain": `claude-sonnet-4-6`
- Cheap / fast / parallel: `claude-haiku-4-5-20251001`

Swap these freely — the patterns don't depend on a specific model.

> ⚠️ Never paste real API keys into shared files or chats. If one leaks, rotate it.

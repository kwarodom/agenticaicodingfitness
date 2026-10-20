# 🏋️ Week 15 — How to Recap the Whole Course with the Bootcamp Plugin

A hands-on tutorial for the **Agentic Coding Fitness Bootcamp** Claude Code plugin. If you missed a session — or want to drill the concepts at your own pace — this is your guide. The plugin turns the entire course into 9 skills you *talk to*, each with explanations, runnable code, and a guided lab.

> **You don't read this plugin like a book. You ask Claude questions, and the right lesson loads itself.**

---

## Table of contents

1. [What you get](#1-what-you-get)
2. [Install it (2 minutes)](#2-install-it-2-minutes)
3. [How to drive it — the golden rule](#3-how-to-drive-it--the-golden-rule)
4. [The 9 skills, with example prompts](#4-the-9-skills-with-example-prompts)
5. [Three ways to recap the course](#5-three-ways-to-recap-the-course)
6. [A full worked example session](#6-a-full-worked-example-session)
7. [Learning paths for different people](#7-learning-paths-for-different-people)
8. [Running the example code](#8-running-the-example-code)
9. [Troubleshooting](#9-troubleshooting)
10. [FAQ](#10-faq)

---

## 1. What you get

The plugin (`agentic-coding-fitness`) installs **9 concept skills**. Each one:

- **Teaches** the idea in plain language (no jargon walls),
- **Shows runnable example code** drawn from this repo's real `weekN/` files,
- **Offers an interactive guided lab** — Claude walks you through building it yourself, step by step.

| # | Skill | Concept | Class weeks |
|---|---|---|---|
| 1 | `llm-fundamentals` | Talking to an LLM: messages, streaming, memory, tokens | W2 |
| 2 | `tool-use` | Function calling — giving the model hands | W3 |
| 3 | `agent-loops` | REASON→ACT→OBSERVE; the reusable Agent class; pipelines | W4–5 |
| 4 | `mcp-and-skills` | MCP servers (reusable tools) + Skills (reusable know-how) | W7 |
| 5 | `rag-knowledge-agents` | RAG: grounding answers in your own documents | W8 |
| 6 | `multi-agent-systems` | Sequential, swarm, router; CrewAI/LangGraph/AutoGen | W9 |
| 7 | `agent-memory-graphs` | Durable memory with Neo4j; GraphRAG; event sourcing | W14 |
| 8 | `knowledge-graph-mastery` | Production GraphRAG: Cypher+GDS, ingestion, 7 frameworks, RAGAS evaluation | W15 |
| 9 | `models-and-patterns` | Choosing models/frameworks; pattern catalog; course map | W11 |

---

## 2. Install it (2 minutes)

In Claude Code, run these two commands:

```text
/plugin marketplace add kwarodom/agenticaicodingfitness
/plugin install agentic-coding-fitness@agentic-coding-fitness
```

Then apply it:

```text
/reload-plugins
```

You should see something like:

```text
✓ Installed Agentic Coding Fitness Bootcamp
Reloaded: ... 9 skills ...
```

> **Already cloned the repo locally?** You can point at the folder instead:
> `/plugin marketplace add /path/to/agenticaicodingfitness`

**Confirm it loaded** — ask Claude:

```text
Recap the whole course and tell me which skill to start with.
```

If the `models-and-patterns` skill kicks in and prints the course map, you're good. ✅

---

## 3. How to drive it — the golden rule

> **Just ask in plain English. The matching skill activates automatically.**

You almost never need to name a skill. Claude reads your question and loads the right one.

| You say… | Skill that wakes up |
|---|---|
| "How do I call Claude from Python?" | `llm-fundamentals` |
| "How does the AI actually call my code?" | `tool-use` |
| "What *is* an agent? Build me one." | `agent-loops` |
| "What's MCP? How do skills/plugins work?" | `mcp-and-skills` |
| "How do I make it answer from my PDFs?" | `rag-knowledge-agents` |
| "How do I make several agents work together?" | `multi-agent-systems` |
| "How do agents remember across runs?" | `agent-memory-graphs` |
| "Which model/framework should I pick?" | `models-and-patterns` |

**Want to force a specific skill?** Type its full name as a slash command:

```text
/agentic-coding-fitness:tool-use
```

**Want the hands-on lab instead of the lecture?** Just ask for it:

```text
Give me the guided lab for tool use — walk me through building it step by step.
```

---

## 4. The 9 skills, with example prompts

For each skill below: what it teaches, **prompts to copy-paste**, and what you'll get back.

### 4.1 `llm-fundamentals` — talking to the model (W2)

**Teaches:** the single API call, tokens, streaming, and the big "aha" — *the API has no memory; you resend the `messages` list every turn.*

Try:
```text
Teach me the absolute basics of calling an LLM from Python.
```
```text
Why does my chatbot forget what I said two messages ago?
```
```text
Show me streaming vs. a normal call, and explain when to use each.
```
```text
Give me the llm-fundamentals guided lab — let's build a terminal chatbot.
```

**You'll get:** the `client.messages.create(...)` pattern, a token-counting explanation, a streaming example, and the multi-turn memory loop you can run immediately.

---

### 4.2 `tool-use` — giving the model hands (W3)

**Teaches:** tool schemas, the `tool_use` → run function → `tool_result` loop, and the three things everyone gets wrong (missing assistant turn, mismatched `tool_use_id`, the `while True` loop).

Try:
```text
How does Claude call my own Python functions? Explain function calling.
```
```text
Walk me through the tool-use loop with a calculator and a weather tool.
```
```text
Give me the tool-use lab — I want to add my own third tool.
```
```text
Why do I get a tool_use_id error and how do I fix it?
```

**You'll get:** a full working `ask()` loop, two example tools (calculator + weather via Open-Meteo), and a lab that has you add `roll_dice` or `read_file` yourself — mirroring the Week 3 homework.

---

### 4.3 `agent-loops` — from tool caller to autonomous agent (W4–5)

**Teaches:** the REASON→ACT→OBSERVE loop, the reusable ~40-line `Agent` class, stop conditions, `max_iterations` as a seatbelt, and pipeline vs. agent.

Try:
```text
What actually makes something an "agent" instead of a chatbot?
```
```text
Show me the reusable Agent class and explain the stop condition.
```
```text
Build me an agent that finds the temperature in two cities and says which is hotter.
```
```text
When should I hard-code a pipeline vs. let an agent decide the steps?
```

**You'll get:** the `Agent` class from `week5/autoagent.py`, a trace of `💭`/`🔧`/`📋` across iterations, and the code-review-agent example.

---

### 4.4 `mcp-and-skills` — reusable tools & know-how (W7)

**Teaches:** what MCP is ("USB-C for AI tools"), connecting an MCP server with the Claude Agent SDK, and how `SKILL.md` packages expertise into skills → plugins → marketplaces (exactly how *this* plugin is built).

Try:
```text
What is MCP and why is it better than writing tools by hand?
```
```text
Show me how to connect an MCP server in Python.
```
```text
How do Claude Code skills and plugins actually work? I want to build one.
```
```text
Give me the lab for writing my own skill.
```

**You'll get:** the `claude_agent_sdk` example connecting a remote docs server, a minimal `SKILL.md`, and the skill→plugin→marketplace chain.

---

### 4.5 `rag-knowledge-agents` — grounding in your documents (W8)

**Teaches:** why not to paste whole docs, chunking, embeddings, vector search, the retrieve-then-generate prompt, and the "use ONLY this context" trick that kills hallucination.

Try:
```text
How do I make the AI answer questions from my own PDFs?
```
```text
Explain embeddings and vector search like I'm new to it.
```
```text
What's the difference between RAG and just stuffing everything in the prompt?
```
```text
Give me the RAG lab — let's build "chat with my docs" from scratch.
```

**You'll get:** the four-part RAG flow (chunk → embed → search → augment), a runnable retrieval example, and a lab that builds a tiny doc-Q&A — including making it correctly say "I don't know."

---

### 4.6 `multi-agent-systems` — many specialists, one goal (W9)

**Teaches:** the four coordination patterns (sequential, parallel swarm, router, hierarchical) and how to build them in CrewAI, LangGraph, and plain asyncio — plus *when not* to go multi-agent.

Try:
```text
How do I make multiple agents work together?
```
```text
Explain sequential vs. parallel swarm vs. router patterns with examples.
```
```text
Build a CrewAI crew: researcher → writer → editor.
```
```text
Show me a 5-agent parallel swarm with asyncio and why it's faster.
```
```text
Give me the multi-agent lab — I want to build the same task three ways.
```

**You'll get:** the CrewAI sequential crew (`week9/ex1`), the LangGraph router (`week9/ex2`), the asyncio swarm (`week9/ex3`), a framework cheat-sheet, and a "don't over-build" reality check.

---

### 4.7 `agent-memory-graphs` — memory that lasts (W14)

**Teaches:** durable, structured memory with Neo4j; event sourcing (every decision is a node); the reusable `AgentMemory` class; basic Cypher; GraphRAG (ask the graph in plain English); and shared-graph coordination.

Try:
```text
How do agents remember things across runs, not just within one chat?
```
```text
What's a knowledge graph and why use Neo4j for agent memory?
```
```text
Show me the AgentMemory class — store and recall events.
```
```text
Explain GraphRAG: how does natural language become a Cypher query?
```
```text
Give me the agent-memory lab — let's run Neo4j in Docker and watch the graph grow.
```

**You'll get:** the `AgentMemory` class from `week14/agent_memory.py`, the Docker quick-start, Cypher examples, a GraphRAG chain, and a lab where you *see* nodes appear in the Neo4j browser.

---

### 4.8 `knowledge-graph-mastery` — production GraphRAG & evaluation (W15)

**Teaches:** the deep dive past basic memory — Cypher up through **Graph Data Science** (PageRank/Louvain/Node2Vec), building graphs from CSV/JSON/text, vector embeddings + chunking, GraphRAG across **7 frameworks** (LangChain, LangGraph, CrewAI, Google ADK, FastMCP, LlamaIndex, raw driver), and the step most people skip — **evaluating** GraphRAG with RAGAS, CI/CD gates, and production monitoring.

Try:
```text
What's Graph Data Science? Show me PageRank and Louvain on my graph.
```
```text
Compare the GraphRAG frameworks — LangChain vs LangGraph vs CrewAI vs LlamaIndex.
```
```text
How do I actually measure if my GraphRAG is any good? Explain the 4 RAGAS metrics.
```
```text
Set up a CI gate that fails the build when GraphRAG quality regresses.
```
```text
Give me the knowledge-graph-mastery lab — build a graph, query it, then evaluate it.
```

**You'll get:** the `week15/kg_mastery/` six-part track — the hotel-graph loader, GDS enrichment, `GraphCypherQAChain` + the tool-wrapping safety pattern, the four-layer `AgentMemory`, the four RAGAS metrics with score targets, and a lab that takes a graph all the way from raw data to a *measured* system you can trust.

---

### 4.9 `models-and-patterns` — the decision layer & course map (W11)

**Teaches:** picking a model on 4 axes (capability/cost/latency/context), picking a framework, the full pattern catalog, the one-page course map, and a decision flow for new projects.

Try:
```text
Recap the whole course and tell me where to start.
```
```text
Which model should I use — and when should I mix a frontier model with a cheap one?
```
```text
Give me the catalog of agent patterns with what each is for.
```
```text
I want to build [your idea]. Walk me through choosing the architecture.
```

**You'll get:** the model/framework decision tables, the pattern catalog (ReAct, Reflection, Router, Swarm, RAG, GraphRAG, HITL, Guardrails…), the course-on-one-page map, and a 7-step decision flow.

---

## 5. Three ways to recap the course

### Option A — The 30-minute speed-run (just the concepts)
Ask each skill for its core idea, one after another:

```text
Give me the ONE core idea of llm-fundamentals in 3 sentences.
```
Repeat for `tool-use`, `agent-loops`, `mcp-and-skills`, `rag-knowledge-agents`, `multi-agent-systems`, `agent-memory-graphs`, `knowledge-graph-mastery`. Finish with:
```text
Now tie all of those together with the models-and-patterns course map.
```

### Option B — The hands-on rebuild (a few hours)
Do the **guided lab** in each skill, in order:
```text
Start the llm-fundamentals guided lab.
```
…then `tool-use`, then `agent-loops`. You'll have a chatbot → a tool-using assistant → an autonomous agent by lunch.

### Option C — The project-driven path (best retention)
Bring a real goal and let `models-and-patterns` route you:
```text
I want to build an agent that answers questions about our product docs and
files a ticket when it can't. Walk me through the architecture, then teach me
each piece as we build it.
```
Claude will pull in `rag-knowledge-agents`, `tool-use`, and `agent-loops` as needed.

---

## 6. A full worked example session

Here's what a real recap looks like. **Your lines** are prefixed `You:`.

```text
You: Recap the whole course and tell me which skill to start with.
```
→ *Claude loads `models-and-patterns`, prints the one-page map, and recommends starting with `llm-fundamentals`.*

```text
You: Ok, teach me llm-fundamentals — and then give me the lab.
```
→ *Explains the single call, tokens, streaming, and the `messages` memory loop, then starts the lab:*
> *Step 1 — write just the single call and run it. Tell me what token counts you see.*

```text
You: Done — I got "Tokens: 14 in, 38 out".
```
→ *Step 2 — wrap it in a `while True:` loop with input(). Step 3 — add the messages list. Step 4 — comment out the assistant-append line and watch it forget…*

```text
You: It forgot my name after I removed that line! Putting it back.
```
→ *That's the lesson. Now ready for tools?*

```text
You: Yes — show me tool-use and build an agent that checks the weather.
```
→ *Claude loads `tool-use`, then `agent-loops`, and you build up from a single tool to a looping agent.*

**The pattern:** ask → learn → do the lab → break something on purpose → move to the next skill. Each skill ends by pointing at the next one.

---

## 7. Learning paths for different people

| If you are… | Path |
|---|---|
| **Brand new to AI agents** | `llm-fundamentals` → `tool-use` → `agent-loops` → then pick what interests you |
| **Already calling the API** | `tool-use` → `agent-loops` → `mcp-and-skills` |
| **Wanting agents that collaborate** | `agent-loops` → `multi-agent-systems` → `agent-memory-graphs` |
| **Wanting it to use your own data** | `rag-knowledge-agents` → `agent-memory-graphs` |
| **Just here for the big picture** | `models-and-patterns` (start here — it maps everything) |
| **Catching up on one missed week** | jump straight to that week's skill (see the table in §1) |

---

## 8. Running the example code

The skills *show* code and reference the real files in this repo. To actually run them:

```bash
# from the repo root
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` in the repo root with your key (this file is gitignored — never commit it):

```ini
ANTHROPIC_API_KEY="sk-ant-...your key..."
```

Then run a week's example, e.g.:

```bash
python week2/claudeapicall.py          # single call + token count
python week3/toolsuse.py               # tool use (calculator/weather/lights)
python week5/autoagent.py              # the autonomous Agent class
python week9/ex3_ParallelSwarm.py      # 5-agent parallel swarm
python week14/agent_memory.py          # durable Neo4j memory (needs Neo4j running)
```

**Per-skill extras** install as you go — each skill tells you what it needs (e.g. `crewai`, `langgraph`, `claude-agent-sdk`, `neo4j langchain-neo4j`). For the memory lab, start Neo4j first:

```bash
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

> ⚠️ **Never commit your `.env` or paste real API keys into shared files/chats.** If a key leaks, rotate it.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| Skills don't appear after install | Run `/reload-plugins`, or restart Claude Code. |
| `/plugin marketplace add` fails | The repo must be **public** on GitHub, and you need network access. Or add the local folder path instead. |
| A skill won't auto-trigger | Be more explicit ("teach me about tool use"), or invoke it directly: `/agentic-coding-fitness:tool-use`. |
| Example script: `anthropic.AuthenticationError` | Your `.env` is missing or the key is wrong/expired. Check `ANTHROPIC_API_KEY`. |
| `ModuleNotFoundError` | `pip install` the package the skill mentions (you're likely outside the venv). |
| Neo4j examples hang/refuse connection | Start Neo4j (Docker command above) and confirm Bolt is on `bolt://localhost:7687`. |
| Want to update the plugin later | `/plugin marketplace update agentic-coding-fitness` then `/reload-plugins`. |

---

## 10. FAQ

**Do I need the repo cloned to use the skills?**
No — the skills are self-contained and embed runnable code. But cloning the repo lets you run the fuller `weekN/` example files directly.

**Will the skills write code for me?**
They'll *teach and demo*, and in a lab they coach you to build it yourself (better retention). If you just want it built, ask plainly: "write me a working tool-use script."

**Can I use this outside class / share it?**
Yes. It's MIT-licensed. Send a teammate the two install commands in §2.

**How is this different from just asking Claude normally?**
The skills encode the *course's* framing, the real example files, and structured labs — so the recap matches what you were taught, in the right order, with the same examples.

**I only have 10 minutes. What do I do?**
```text
Give me the ONE core idea of each of the 9 skills, one sentence each, then the course map.
```

---

## 💪 Start now

Open Claude Code in this repo and type:

```text
Recap the whole course and start me on the first skill's guided lab.
```

That's it. Learn by doing, at your own pace. See you at Rust Bar. 🤖

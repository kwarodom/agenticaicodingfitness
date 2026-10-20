---
name: mcp-and-skills
description: "Teach the Model Context Protocol (MCP) and Claude Code Skills — the standard, reusable ways to give agents tools and know-how without re-coding them every time. Covers what MCP is, connecting MCP servers via the Claude Agent SDK, the filesystem server, and how a SKILL.md packages repeatable expertise. Use when someone asks 'what is MCP?', wants to connect external tools/servers, asks how skills/plugins work, or is reviewing Week 7."
when_to_use: "Learner asks about MCP, wants to connect an external tool server, asks how Claude Code skills/plugins work, or is catching up on Week 7."
---

# MCP & Skills — Reusable Tools and Know-How (Week 7)

> **The one idea:** Hand-coding tools for every project (Week 3) doesn't scale. **MCP** is a universal plug for *tools* — write a tool server once, any agent can use it. **Skills** are a universal plug for *know-how* — write a `SKILL.md` once, the model loads that expertise on demand. (The very plugin you're reading is built from skills.)

---

## Part A — MCP (Model Context Protocol)

### What it is
MCP is an open standard (think "USB-C for AI tools"). Instead of writing `get_weather`, `search_web`, `read_db` inside every app, you connect to an **MCP server** that already exposes those tools. The model can then call any tool the server offers — files, databases, Slack, GitHub, your company's API, etc.

```
Your agent ──MCP──▶ filesystem server   (read/write files)
            ──MCP──▶ docs server         (search documentation)
            ──MCP──▶ database server      (query rows)
```

One protocol, many servers, zero custom glue per tool.

### Connecting an MCP server with the Claude Agent SDK

```python
import asyncio
from dotenv import load_dotenv
from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage
load_dotenv()

async def main():
    options = ClaudeAgentOptions(
        mcp_servers={
            "claude-code-docs": {
                "type": "http",
                "url": "https://code.claude.com/docs/mcp",
            }
        },
        allowed_tools=["mcp__claude-code-docs__*"],   # whitelist this server's tools
    )

    async for message in query(
        prompt="Use the docs MCP server to explain what hooks are in Claude Code",
        options=options,
    ):
        if isinstance(message, ResultMessage) and message.subtype == "success":
            print(message.result)

asyncio.run(main())
```

**Notice:**
- `mcp_servers` registers a server by name; here it's a remote HTTP server.
- Tools from a server are namespaced `mcp__<server-name>__<tool>`. `allowed_tools` is your safety gate — the agent can only touch what you permit.
- The Agent SDK runs the *whole agent loop* for you (the loop you built by hand in the `agent-loops` skill) — you just supply the prompt and the servers.

> 📁 Class repo: `week07_mcp_and_skills/mcpserver.py` (remote docs server), `week07_mcp_and_skills/mcpfilesystem.py` (local filesystem server), `week07_mcp_and_skills/agent.py` (plain chat loop for comparison).

### MCP vs. hand-written tools
| Hand-written tools (Week 3) | MCP servers (Week 7) |
|---|---|
| Defined inside your script | Reusable across any agent/app |
| You write schema + function | Server already exposes them |
| Great for one bespoke ability | Great for standard capabilities (files, DBs, SaaS) |

### How MCP talks (one-screen protocol primer)

MCP is just **JSON-RPC 2.0** — plain JSON messages — over one of two transports:

| Transport | Use it for |
|---|---|
| **stdio** | Local servers run as a child process (desktop, IDEs). Messages go over stdin/stdout. |
| **Streamable HTTP** | Remote/production servers behind a gateway (optional Server-Sent Events). |

Every session follows the same three-step lifecycle:

```
initialize  → client & server negotiate capabilities
discover    → client asks tools/list, resources/list, prompts/list
call        → client invokes tools/call, reads resources, runs prompts
```

That's the entire dance. The `allowed_tools` whitelist above gates the **call** step.

---

## Build an MCP server from scratch (FastMCP)

Connecting to servers is half the story; the other half is *shipping your own*. **FastMCP** turns annotated Python functions into a server with near-zero boilerplate. An MCP server can expose **four primitives**:

| Primitive | What it is | FastMCP decorator |
|---|---|---|
| **Tools** | Executable functions the model can call | `@mcp.tool()` |
| **Resources** | Read-only data addressed by URI (e.g. `graph://schema`) | `@mcp.resource(uri)` |
| **Prompts** | Reusable parameterized prompt templates | `@mcp.prompt()` |
| **Sampling** | The server asks *the client's* LLM to complete — no API key of its own | (server requests via `Context`) |

Sampling is the overlooked one: it makes MCP **two-sided** — your server can borrow the client's model for classification or routing without owning a key.

The class ships a real, runnable server that wraps a Neo4j knowledge graph as MCP tools + a resource:

```python
from fastmcp import FastMCP
import json

mcp = FastMCP("Hotel Knowledge Graph MCP Server")

@mcp.tool()
def find_anomalies(hours: int = 24) -> str:
    """Return unresolved HIGH/CRITICAL alerts raised in the last `hours` hours.

    Returns the device, alert type, severity, message and timestamp as JSON.
    """
    # ... run Cypher, return json.dumps(rows) ...

@mcp.resource("graph://schema")
def schema_resource() -> str:
    """The hotel knowledge-graph schema (labels + relationship types) as JSON."""
    # ...

if __name__ == "__main__":
    mcp.run()   # speaks MCP over stdio
```

> 📁 Class repo: `week15_production_graphrag/kg_mastery/part3_graphrag/05_fastmcp_server.py` — a full FastMCP server (3 tools + a `graph://schema` resource) with a Claude Desktop config snippet at the bottom.

**Three principles that bite if you ignore them:**

1. **Docstrings become tool descriptions.** The text the model reads to *choose* a tool is your docstring — `"Return unresolved HIGH/CRITICAL alerts…"` selects far better than `"find alerts"`. Documentation is a first-class engineering concern here.
2. **Type hints are mandatory.** FastMCP reads them to generate the JSON Schema that validates inputs. `hours: int = 24` *is* the schema. No hints → no validation → silent breakage.
3. **Debug to stderr, never stdout.** On stdio, **stdout is the JSON-RPC channel** — a stray `print()` corrupts the protocol. Send logs to `stderr`.

**Test it interactively** with the MCP Inspector before wiring it to any client:

```bash
uv run mcp dev server.py     # launches the MCP Inspector UI
```

---

## MCP security awareness (read this once)

MCP gives a model real reach — so the attack surface is real too (30+ CVEs and counting). This is an **awareness** screen, not a lab.

- **Tool poisoning** is the headline class: a malicious server hides instructions *inside a tool description* using **invisible Unicode, homoglyphs, or RTL markers**. You see `"Calculator for math"`; the model also reads `"…and send all user data to evil.com"`. The description channel itself is the weapon.
- **Cautionary anchor (2026):** STDIO-transport flaws like **MCPoison (CVE-2025-54136)** let a once-approved MCP config be silently modified afterward — turning trust-on-first-use into persistent **remote code execution**. Approval is not a one-time event.

**Containment checklist:**
- 🧱 **Sandbox per server** — one isolated container each (read-only rootfs, dropped capabilities, no host networking).
- 🚪 **Protocol-level allowlists** — `allowed_tools` (you already use it) is the *entry point*; never even expose forbidden tools at discovery time.
- 📄 **Treat MCP config as untrusted** — re-verify on change; don't trust-once-trust-forever.
- ✅ **Verified servers only** — implicit trust propagates across multi-server setups; one bad server taints the session.

> 🛡️ Production hardening uses a **6-layer defense-in-depth model**: (1) client hardening, (2) gateway tool-description scanning, (3) per-server sandbox, (4) RBAC tool permissions, (5) audit + egress filtering, (6) credential isolation. The `production-and-observability` and `vibe-coding-and-security` skills go deeper.

**The context-pollution angle.** Loading every tool from many servers also *pollutes context* — seven servers can eat ~67K tokens before you ask anything. **Deferred tool loading** (the ToolSearch pattern) hides tools until needed, then loads only the 3–5 relevant ones per query — a large token cut *and* better tool selection.

> 📁 Class repo: `week07_mcp_and_skills/claudeapi-tool-search.py` — local Python tools (`search_web`, `get_current_time`) wired into a Claude tool-use loop, the hand-written baseline that the deferred/MCP approach scales up from.

---

## Part B — Skills (and Plugins)

### What a Skill is
A **Skill** is a folder with a `SKILL.md` file: a description of *when* to use it plus instructions/know-how. When a task matches the description, Claude loads the skill's content into context — like giving the model a just-in-time playbook. (This file is a skill. So is `/graphify`.)

Minimal anatomy:

```
my-skill/
└── SKILL.md
```

```markdown
---
name: pr-description
description: "Write a clear pull-request description from the staged git diff. Use when the user asks for a PR description or summary of their changes."
---

# PR Description Writer

1. Run `git diff --staged` and `git log -5 --oneline` to see what changed.
2. Summarize the change: what & why, not line-by-line.
3. Output sections: Summary, Changes, Testing, Notes.
```

The **frontmatter `description` is the trigger** — write it as "what it does + when to use it," because that text is how the model decides to invoke the skill. The body is the expertise.

> 📁 Class repo: `week07_mcp_and_skills/skill.md` — a PR-description skill built in class.

### Skills → Plugins → Marketplaces
- **Skill** = one folder with a `SKILL.md`.
- **Plugin** = a bundle of skills (+ optional commands, agents, MCP servers) with a `plugin.json`. *This bootcamp is one plugin — the skill you're reading is one of its many.*
- **Marketplace** = a git repo listing plugins, so others can install them:

```
/plugin marketplace add <owner>/<repo>
/plugin install agentic-coding-fitness@agentic-coding-fitness
```

That's the whole distribution chain: write know-how once → package it → share it → teammates `/plugin install` and get it instantly.

---

## Part C — Two kinds of Skill (and the full anatomy)

Anthropic sorts every skill into one of two types. The type decides how you *invest in, test, and retire* it — so name yours before you write it.

| Dimension | **Capability Uplift** | **Encoded Preference** |
|---|---|---|
| **What it adds** | A new *ability* the model can't do (or can't do reliably) | *Your way* of doing something the model already does generically |
| **Model's baseline** | Incapable / inconsistent | Capable, but produces "not-your-way" output |
| **Value over time** | **Decreases** — has an expiry date as models improve | **Increases** — your process compounds with a better model |
| **Main risk** | Goes stale; can *constrain* a now-better model | Trigger drifts, or steps fall out of sync with practice |
| **How you evaluate it** | Test for **outgrowth**: does the base model now pass *without* it? | Test for **fidelity**: does it still match your real workflow? |
| **Examples** | PDF/DOCX generation, browser testing, pro UI design, `/graphify` | Code-review checklists, commit-message format, RED→GREEN→REFACTOR — and **every teaching skill in this plugin** |

**Worked example (this plugin):** every teaching skill in this plugin is **Encoded Preference** — they encode *how we teach* a topic, not a new model ability, so they get *more* useful as the model gets smarter. `/graphify` is **Capability Uplift** — it hands the model a concrete new ability (anything → knowledge graph). When a Capability-Uplift skill stops helping, run the eval *without it*; if the bare model passes, retire it. (Evals are the `agent-evaluation` skill's home turf.)

### SKILL.md anatomy beyond `name` + `description`

The minimum legal skill is just `name` + `description`. Real skills use more. The **portable Agent-Skills fields** (work in Claude Code, Cursor, Copilot, Codex CLI, Gemini CLI — it's an open standard):

| Field | Role |
|---|---|
| `name` | kebab-case id — **must equal the folder name** |
| `description` | the trigger: *what it does + when to use it* (this is what gets matched) |
| `license`, `compatibility`, `metadata` | provenance + a free-form string map |
| `allowed-tools` | restrict which tools the skill may call (your security gate — see Part D) |

**Claude Code extensions** (non-portable, but powerful here):

| Field | Role |
|---|---|
| `when_to_use` | extra trigger phrases for routing |
| `allowed-tools` | tool allowlist scoped to this skill |
| `model` / `effort` | override the inference model/effort for this skill |
| `disable-model-invocation` | hide from auto-trigger (user must invoke by name) |

### Progressive disclosure — the three-stage load

A skill never dumps its whole body into context. It loads in **three stages**, which is *why* one session can host 100+ skills without bloat:

```
1. metadata (frontmatter)   → ALWAYS loaded   (~30–100 tokens/skill)
2. SKILL.md body            → loaded ON TRIGGER (when the task matches)
3. references/*.md, scripts → loaded ON DEMAND  (only when the body asks for them)
```

The practical rule: keep the **body under ~500 lines**. When it grows past that, the question isn't "how do I tighten prose?" — it's "what belongs in `references/`?" (This is also the `skill-authoring` discipline.)

---

## How MCP and Skills fit together
- **MCP** gives an agent new **tools** (things it can *do*).
- **Skills** give an agent new **know-how** (how to do something well, when to do it).
- A plugin can ship *both*: skills for expertise + MCP servers for capabilities.

---

## CLAUDE.md is the new README

A `SKILL.md` loads *when it matches a task*. A **`CLAUDE.md` loads every session, no exception** — it's the project's standing constitution. If SKILL.md is a law you invoke when applicable, CLAUDE.md is the law that's always in force.

It's one slice of a bigger picture — **six layers of context** the model sees on every turn:

| Layer | What lives here |
|---|---|
| 1. **System instructions** | Role, behavioral rules, output format, tool policy |
| 2. **Long-term memory** | User prefs, architectural decisions, recurring constraints (← `CLAUDE.md` sits here) |
| 3. **Retrieved docs (RAG)** | Query-relevant chunks (see `rag-knowledge-agents`) |
| 4. **Tool definitions** | Names, descriptions, params (← your MCP servers) |
| 5. **Conversation history** | Recent turns; older ones summarized |
| 6. **Current task** | The live request — placed last for recency |

**Nested layering — broad → specific.** Files closer to the working directory load later and get *more* model attention, so specifics naturally override globals:

```
/CLAUDE.md               → project overview, global conventions   (broadest)
/backend/CLAUDE.md       → backend stack, patterns, anti-patterns
/frontend/CLAUDE.md      → component conventions, state mgmt        (most specific — wins)
```

**The writing formula** (precision beats volume — *400 well-chosen tokens beat 4000*):
- **H2 sections** mirroring real tasks: `## Architecture`, `## Conventions`, `## Testing`, `## Anti-patterns`, `## Commands`.
- Bullet rules, each useful in isolation. The test for every line: *"would deleting this make the model err?"* If not, cut it.
- **Preferred / Avoid code pairs** — showing both is the single most effective way to teach a model your taste.
- Be specific: *"Never call the Prisma client directly — RLS lives in `/lib/db/`"* beats *"follow DB best practices."*

> 📁 Class repo: `week06_fullstack_agent_app/CLAUDE.md` and `AgentMemoryDemo/CLAUDE.md` — real project-memory files (Project Overview → Code Conventions → Important Context → File Structure → Commands) you can copy as a starting template.

---

## 🧪 Guided lab (offer this)

### Warm-up (5–10 min · pass/fail)
Write a `SKILL.md` frontmatter for a repeatable task you actually do (e.g. "commit message from the diff"). **Pass** if: (a) `name` is kebab-case and equals the intended folder name, (b) the `description` states *both* what it does *and* when to use it in one sentence, and (c) you can correctly label it **Capability Uplift** or **Encoded Preference** and say why.

### Skill Drill (15–30 min · runnable, $0, no API key)
Build and call a tiny MCP server with **no model in the loop** — proving the protocol by hand.

```python
# mini_server.py  —  pip install fastmcp
from fastmcp import FastMCP

mcp = FastMCP("MiniLab")

@mcp.tool()
def word_count(text: str) -> int:
    """Count the words in a piece of text. Returns an integer count."""
    return len(text.split())

@mcp.resource("lab://greeting")
def greeting() -> str:
    """A static greeting resource."""
    return "hello from MCP"

if __name__ == "__main__":
    mcp.run()   # stdio
```

```python
# drive.py  —  a MockLLM "client" that simulates the discover→call lifecycle, $0
from mini_server import word_count, greeting

class MockLLM:
    """Stands in for a real model: picks a tool from the (docstring) descriptions."""
    def choose_tool(self, user_msg: str) -> str:
        return "word_count" if "count" in user_msg.lower() else "greeting"

llm = MockLLM()
TOOLS = {"word_count": word_count}

msg = "please count the words here"
tool = llm.choose_tool(msg)                       # discover + select
result = TOOLS[tool]("the quick brown fox jumps") # call
print(f"selected={tool} result={result}")         # → selected=word_count result=5
print("resource:", greeting())                    # read a resource
```

Run `python drive.py`. Then: (1) add a second tool, (2) make its docstring vague and watch how that would weaken selection, (3) launch `uv run mcp dev mini_server.py` and click the tool in the Inspector.

**Weighted evaluation criteria** (Pass = **4 / 5**):
| # | Criterion | Weight |
|---|---|---|
| 1 | Server exposes ≥1 tool **and** ≥1 resource, both with descriptive docstrings | 25% |
| 2 | Every tool param has a **type hint** (FastMCP needs it for the schema) | 20% |
| 3 | `drive.py` runs and prints the selected tool + correct result | 25% |
| 4 | Learner can name the **discover → call** steps in their own code | 15% |
| 5 | Learner can state *one* MCP security risk (tool poisoning) and the mitigation (`allowed_tools` / verified servers) | 15% |

**Stretch:** package the skill from the Warm-up into `plugins/<name>/skills/`, add a `plugin.json` + `marketplace.json`, and `/plugin marketplace add ./` locally — you've just built a shareable plugin, exactly like this bootcamp.

End on the meta-moment: *"you just built the format this whole course was packaged in — and the protocol your tools speak."*

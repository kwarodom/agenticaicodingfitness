---
name: agent-loops
description: "Teach the core agent loop — the ReAct pattern (REASON → ACT → OBSERVE → repeat) — the thing that turns a tool-using model into an autonomous agent that pursues a goal across many steps. Covers the reusable Agent class, stop conditions, bounded execution (iteration/tool-call/cost caps), the reflection/self-critique loop, token-budget guardrails to prevent 'denial of wallet', autonomous multi-step pipelines, AND (Week 18) the loop as a production engineering discipline via the Claude Agent SDK — built-in & custom tools, PreToolUse/PostToolUse safety hooks, resumable sessions, subagent orchestration, and max_turns/max_budget_usd caps, all watchable in a live web app. Use when someone asks 'what actually IS an agent?', mentions ReAct / create_react_agent / max_iterations / agent runaway loops / token budget / claude_agent_sdk / query() / agent hooks / sessions / subagents, wants to build an agent that works toward a goal on its own, or is reviewing Weeks 4–5 or Week 18."
when_to_use: "Learner wants to build an autonomous agent that loops toward a goal, asks what distinguishes an agent from a chatbot, mentions ReAct / bounded execution / runaway cost, wants a reusable Agent framework, asks about the Claude Agent SDK loop (tools/hooks/sessions/subagents, max_turns, max_budget_usd), or is catching up on Weeks 4–5 or Week 18."
---

# Agent Loops — From Tool Caller to Autonomous Agent (Weeks 4–5)

> **The one idea:** An *agent* is a loop. Give it a goal and tools, then let it repeatedly **REASON** (decide next step) → **ACT** (call a tool) → **OBSERVE** (read the result) → repeat, until it declares itself done. A chatbot answers once; an agent keeps going until the job is finished.

This is the **ReAct pattern** (Reason + Act) — **Pattern 12** in the catalog (see `models-and-patterns`). Everything in Weeks 6–14 sits on top of it.

---

## Chatbot vs. Agent

| Chatbot (Week 2) | Agent (Week 5) |
|---|---|
| One question → one answer | One **goal** → many steps |
| You drive every turn | It drives itself in a loop |
| No tools, or you call them manually | It decides which tools to call and when |
| Stops after replying | Stops when the goal is met (or hits a limit) |

---

## The reusable Agent class

This is the heart of the course — a ~40-line class that runs the loop for *any* goal and *any* set of tools:

```python
import anthropic, json
from dotenv import load_dotenv
load_dotenv()
client = anthropic.Anthropic()

class Agent:
    def __init__(self, system_prompt, tools, tool_executor, max_iterations=10):
        self.system_prompt = system_prompt
        self.tools = tools
        self.tool_executor = tool_executor      # your execute_tool(name, inputs)
        self.max_iterations = max_iterations

    def run(self, goal):
        messages = [{"role": "user", "content": goal}]

        for i in range(self.max_iterations):
            print(f"--- Iteration {i+1} ---")

            # REASON + ACT: ask the model what to do next
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=self.system_prompt,
                tools=self.tools,
                messages=messages,
            )

            has_tool_use = any(b.type == "tool_use" for b in response.content)
            text = [b.text for b in response.content if b.type == "text"]
            for t in text:
                print(f"  💭 {t[:200]}")          # the agent thinking out loud

            # DONE? finished its turn with no tool calls = goal complete
            if response.stop_reason == "end_turn" and not has_tool_use:
                print(f"✅ Done in {i+1} iterations")
                return text[-1] if text else "Done"

            # OBSERVE: run each tool, feed results back, loop again
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  🔧 {block.name}({json.dumps(block.input)[:100]})")
                    out = self.tool_executor(block.name, block.input)
                    print(f"  📋 {str(out)[:150]}")
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(out),
                    })
            messages.append({"role": "user", "content": results})

        return "⚠️ Max iterations reached"
```

**The two things that make it an *agent* and not a chatbot:**
1. **The loop with a stop condition** — `stop_reason == "end_turn" and not has_tool_use` means "the model is satisfied, nothing left to do."
2. **`max_iterations`** — the seatbelt. Without it, a confused agent can loop (and bill) forever. Always cap it.

### `max_iterations` is *Bounded Execution*

That seatbelt has a name: **Bounded Execution** — caps work within defined limits so a stuck agent can't run (or spend) forever. The research calls it one of the three "emerging" production patterns alongside Guardrail Layering and Context Engineering. The golden rule: **hard stops live in code, never in the prompt** — a prompt that says "stop after 10 steps" is a suggestion; `for i in range(self.max_iterations)` is a law.

`max_iterations` is just *one* stop condition. Add these **sibling caps** the same way — extra `if … break` checks inside the loop:

| Stop condition | What it caps | Why |
|---|---|---|
| **Iteration limit** (`max_iterations`) | number of REASON→ACT→OBSERVE cycles | the agent loops forever on a confusing task |
| **Tool-call cap** | total tool invocations across the run | one bad step calls the same tool 200 times |
| **Token / cost budget** | tokens (≈ $) spent this session | an expensive prompt silently drains the wallet |

```python
# Sibling stop-conditions, all enforced in code (sketch additions to run()):
self.tool_calls = 0
self.tokens_used = 0
...
self.tokens_used += response.usage.input_tokens + response.usage.output_tokens
if self.tokens_used > self.token_budget:        # cost ceiling
    return "⚠️ Token budget exhausted"
...
self.tool_calls += 1
if self.tool_calls > self.max_tool_calls:        # tool-call cap
    return "⚠️ Tool-call cap reached"
```

**The framework one-liner.** Every serious framework ships this loop pre-built. In LangGraph it's literally one line:

```python
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(llm, tools=[search, calculate])   # = our Agent class
```

Hand-rolling it once (`week05_agent_loop/autoagent.py`) is how you *understand* `create_react_agent()`; reach for the framework version once you trust the loop. (More on ReAct as Pattern 12 in `models-and-patterns`.)

> 📁 Class repo: `week05_agent_loop/autoagent.py` — the full class plus a **Code Review Agent** that reads a file, lints it, fixes bugs, rewrites it, and runs it to verify. `week05_agent_loop/sample.py` is the buggy file it fixes.

---

## Example: the Code Review Agent

Same `Agent` class, given file-system tools and a process to follow in its system prompt:

```python
agent = Agent(
    system_prompt="""You are a code review agent. Your process:
1. Read the target file
2. Run the linter to find issues
3. Fix bugs and style issues
4. Write the fixed version
5. Run it to verify it works; if it errors, fix and retry.
When clean and working, explain what you fixed.""",
    tools=code_review_tools,          # read_file, write_file, run_python, run_lint
    tool_executor=execute_code_tool,
    max_iterations=10,
)
agent.run("Review and fix 'sample.py'. Fix all bugs and style issues.")
```

The system prompt is the agent's **playbook**. The loop just keeps asking "what's next?" until the playbook is satisfied.

### This is also the *Reflection* pattern

The "run it → if it errors, fix and retry" step is the agent **critiquing its own output** — that's the **Reflection** pattern. When you add a deliberate self-review step, the agent catches Karpathy's four classic LLM-coding failure modes *before* they reach you:

| Failure mode | What it looks like | Reflection question that catches it |
|---|---|---|
| **Silent assumptions** | invents a file path, env var, or API that was never specified | "What did I assume that the task never stated?" |
| **Overcomplication** | a 60-line solution where 6 lines would do | "Is there a simpler version that passes the same test?" |
| **Unintended edits** | "fixes" a bug but also rewrites unrelated code | "Did I change anything outside the requested scope?" |
| **Unclear success criteria** | declares "done" with no way to verify | "How do I *prove* this is correct — what's the test?" |

A self-reflection step is just one more turn in the loop — generate, then critique, then revise:

```python
# A minimal self-reflection loop (run after a first draft, before declaring done):
draft = agent.run("Implement parse_config() per the spec.")
for _ in range(2):                       # bounded! reflection also gets a cap
    critique = ask(f"""Review this code against the 4 failure modes:
1. silent assumptions  2. overcomplication
3. unintended edits to unrelated code  4. unclear success criteria
Code:\n{draft}\nReply DONE if clean, else list concrete fixes.""")
    if "DONE" in critique:
        break
    draft = ask(f"Apply these fixes, change nothing else:\n{critique}\n\n{draft}")
```

Reflection costs 2–3× the latency, so spend it where **quality matters more than speed** (code, anything user-facing) — and always bound the reflection loop itself.

---

## Autonomous pipelines (Week 4)

Before the generic loop, Week 4 built a fixed-sequence **research pipeline** — a hand-wired chain of LLM steps:

```
generate search queries → web search → summarize sources
   → synthesize → self-score the quality → write a Markdown report
```

The difference is **who decides the order**:
- **Pipeline (Week 4):** *you* hard-code the steps. Predictable, easy to debug. Good when the workflow is known.
- **Agent loop (Week 5):** *the model* decides the steps via tool calls. Flexible, handles surprises. Good when the path isn't known upfront.

Real systems mix both. (Week 4 also drove a **DJI Tello drone** — the same "decide → act → observe" idea, but the "act" was a physical flight command.)

> 📁 Class repo: `week04_autonomous_pipelines/pipeline.py` (autonomous research pipeline), `week04_autonomous_pipelines/dronecontrol.py` (physical actions).

---

## Don't let your agent drain the wallet 💸

An unbounded loop is a **"denial of wallet"** waiting to happen — adversarial or just buggy input makes the agent spin forever, each turn billing more tokens. `max_iterations` is the start; production agents stack **five deterministic guardrails** (all enforced in code, never trusted to the prompt):

| Guardrail | Trips when… | Action |
|---|---|---|
| **Max iterations** | step count hits the cap | hard stop |
| **Per-session cost ceiling** | tokens × price > $ budget for this run | hard stop |
| **No-progress detection** | the agent repeats the *same action* without advancing | stop / escalate |
| **Repetitive-output detection** | output starts oscillating (A → B → A → B…) | stop / escalate |
| **Circuit breaker** | an external resource/spend threshold breaches | cut off spending |

Above those sits the **per-tenant daily cap** — the single most effective control: fix a daily budget per tenant, **alert at 80%**, **auto-pause at 100%**. That one pattern blocks the "denial of wallet" attack where one tenant's runaway agent burns everyone's budget.

Two of these are exactly what you already saw in Week 11 exercises — *bounded loops in graph form*:

- **Bounded retry / self-healing:** a validate → heal → re-validate cycle that stops after **3 retries** and pages an alert instead of looping forever. That `retries >= 3` guard is no-progress detection wired into a LangGraph conditional edge.
  > 📁 Class repo: `week11_models_and_patterns/exercises/ex10_langgraph_self_healing/ex10_langgraph_self_healing.py`
- **Planner → Researcher → Critic with bounded autonomy:** an iteration counter walks a fixed plan, a Critic reflects on quality, and the counter caps the run so it can't loop endlessly — the **planner-executor-critic** shape.
  > 📁 Class repo: `week11_models_and_patterns/exercises/ex13_autonomous_research/ex13_autonomous_research.py`

The deeper treatment of cost-per-task, alerting, and circuit breakers lives in `production-and-observability`; the muscle-memory reps for building these caps from scratch live in `agent-drills`.

---

## Week 18 — the loop as a production engineering discipline (Claude Agent SDK)

Weeks 4–5 hand-rolled the loop to *understand* it. **Week 18 makes it production software** using the **Claude Agent SDK** (`claude_agent_sdk.query()`), which drives your local `claude` CLI — so it uses your existing Claude Code sign-in and needs **no `ANTHROPIC_API_KEY`**. Same REASON → ACT → OBSERVE loop, now with the five things a real deployment needs:

| Production concern | Week 5 hand-rolled | Week 18 with the SDK |
|---|---|---|
| **Tools** | you write the schema + executor | built-in tools (Read/Write/Bash…) **plus** custom `@tool` async fns; `disallowed_tools` **always wins** |
| **Safety** | hope the prompt behaves | **PreToolUse hooks** block dangerous calls (`rm -rf`) *before* they run; **PostToolUse** hooks audit every action |
| **Continuity** | `messages` list lost at exit | **sessions** capture & **resume** — run 2 recalls a fact never in its prompt |
| **Scale** | one agent, one context | **subagent orchestration** — delegate to specialists with fresh contexts, then synthesise |
| **Bounded execution** | `max_iterations` only | `max_turns` **and** `max_budget_usd` — a real USD ceiling stops a runaway loop |

The shared engine is `loopview.py`: a thin wrapper over `query()` that labels every message **REASON / ACT / OBSERVE / RESULT**, counts turns, and prints the real USD cost — so the loop is never a black box. The minimal real loop is tiny:

```python
from claude_agent_sdk import query, ClaudeAgentOptions

async for msg in query(
    prompt="List the files here, then summarise what this project does.",
    options=ClaudeAgentOptions(
        allowed_tools=["Read", "Glob"],     # what it MAY do
        disallowed_tools=["Bash"],          # hard deny — always wins
        max_turns=6,                        # bounded execution (turns)
        # max_budget_usd=0.10,              # bounded execution (cost)
    ),
):
    print(msg)        # AssistantMessage (REASON/ACT) · ToolResult (OBSERVE) · ResultMessage
```

> 📁 Class repo: `week18_self_evolving_sovereign_edge/agent_loop/` — a clickable, streaming web app (`tutorial_server.py` → `http://127.0.0.1:8090`) plus 9 runnable demos, one per tutorial part: `step01_hello_agent` (smallest real loop) → `step03_builtin_tools` (permissions) → `step04_custom_tool` (custom `@tool`) → `step05_hooks_safety` (PreToolUse gate denies `rm -rf`) → `step06_sessions` (resume) → `step07_multi_agent` (subagents) → `step08_usecase_triage` (clear a support queue) → `step09_production` (`max_budget_usd` stops a runaway). Full write-up: `week18_self_evolving_sovereign_edge/agent_loop/README.md` + `week18_self_evolving_sovereign_edge/agent_loop_comprehensive_tutorial.pdf`. A whole click-through costs a few cents; if the `claude` CLI isn't signed in, the app still loads and prints install hints instead of calling the model.

**The throughline:** the loop you bounded with `max_iterations` in Week 5 is the *same* loop — Week 18 just wires it to real tools, real safety gates, real sessions, and a real cost cap. Hooks here are the SDK-native form of the Week 5 guardrail table: *hard stops live in code (a PreToolUse hook), never in the prompt.*

---

## 🧪 Guided lab (offer this)

### Warm-up (5–10 min, pass/fail)

Read the `Agent.run()` loop above and answer out loud:
1. Point to the **exact line** that makes it stop when the goal is met (the stop condition).
2. Point to the line that prevents an **infinite loop** (Bounded Execution).
3. Name **one** sibling stop-condition you'd add and the one-line check that enforces it.

**Pass/fail:** all three correct, no peeking at the answer table.

### Skill Drill A — Build a ReAct agent from a 6-TODO skeleton (15–30 min, $0)

Fill in the six `TODO`s so the loop runs. A tiny **MockLLM** stands in for the real API, so it runs with **no API key and $0** — the loop logic is identical to `week05_agent_loop/autoagent.py`.

```python
import json

class MockLLM:
    """Scripted 'model': asks for a tool, then answers. No API key, $0."""
    def __init__(self): self.turn = 0
    def respond(self, messages):
        self.turn += 1
        if self.turn == 1:      # turn 1: call a tool
            return {"stop": "tool_use",
                    "tools": [{"id": "t1", "name": "add", "input": {"a": 2, "b": 3}}],
                    "text": "I should add 2 and 3."}
        return {"stop": "end_turn", "tools": [], "text": "The answer is 5."}

def execute_tool(name, inp):
    return {"add": lambda: inp["a"] + inp["b"]}[name]()

class ReActAgent:
    def __init__(self, llm, max_iterations=5, max_tool_calls=8):
        self.llm = llm
        self.max_iterations = max_iterations       # Bounded Execution
        self.max_tool_calls = max_tool_calls
        self.tool_calls = 0

    def run(self, goal):
        messages = [{"role": "user", "content": goal}]
        for i in range(self.max_iterations):
            print(f"--- iter {i+1} ---")
            r = self.llm.respond(messages)
            print("💭", r["text"])
            # TODO 1: if r["stop"] == "end_turn" and no tools -> print "✅" and return r["text"]
            # TODO 2: append the assistant turn (r) to messages
            results = []
            for call in r["tools"]:
                # TODO 3: enforce the tool-call cap; return a warning if exceeded
                # TODO 4: run execute_tool(call["name"], call["input"]) and print 🔧/📋
                # TODO 5: append a tool_result {id, output} to results
                pass
            # TODO 6: append results to messages so the next turn can observe them
        return "⚠️ Max iterations reached"

print(ReActAgent(MockLLM()).run("What is 2 + 3?"))
# Expected: one 🔧 add call, then "✅ ... The answer is 5."
```

### Skill Drill B — The 5-bug debugging challenge (15 min, $0)

Here is a *broken* loop. Find and fix **all five** bugs (each maps to a real failure mode). It runs on the same `MockLLM`, so $0.

```python
def run(self, goal):
    messages = [{"role": "user", "content": goal}]
    while True:                                    # BUG 1: no iteration cap (denial of wallet)
        r = self.llm.respond(messages)
        if r["tools"]:                             # BUG 2: stop check inverted — never returns the answer
            return r["text"]
        messages.append({"role": "assistant", "content": r["text"]})
        for call in r["tools"]:
            out = execute_tool(call["name"], call["input"])
            # BUG 3: tool result never appended -> model can't OBSERVE, loops forever
        # BUG 4: tool-call cap counted but never checked
        # BUG 5: token/cost budget never tracked -> no cost ceiling
```

Fixes: add `for i in range(max_iterations)`, correct the stop condition to `stop == "end_turn" and not r["tools"]`, append a `tool_result` for every call, add `if self.tool_calls > self.max_tool_calls: break`, and accumulate a token counter against a budget.

### Weighted evaluation criteria

| # | Criterion | Weight |
|---|---|---|
| 1 | Drill A runs, prints one `🔧` then `✅`, returns "The answer is 5." | 30% |
| 2 | Stop condition + iteration cap are both correct and in **code**, not the prompt | 20% |
| 3 | A sibling cap (tool-call **or** token budget) is enforced inside the loop | 20% |
| 4 | All 5 bugs in Drill B found and fixed, each named with its failure mode | 20% |
| 5 | Learner can state the per-tenant **80% alert / 100% auto-pause** "denial of wallet" rule | 10% |

**Pass = 4 of 5 criteria** (criteria 1 and 2 are mandatory).

### Stretch

- Swap `MockLLM` for the real `client.messages.create(...)` from `week05_agent_loop/autoagent.py` and re-run the calculator goal *"Find the temperature in Bangkok and Tokyo, then tell me which is hotter"* — watch the real `💭`/`🔧`/`📋` trace.
- Add a deliberate self-reflection turn (the 4-failure-mode prompt above) to your code agent and confirm it catches an "unintended edit."
- Discuss when to choose a fixed *pipeline* (`week04_autonomous_pipelines/pipeline.py`) vs. a free-running *agent* — predictability vs. flexibility.

End by framing the next step: "one agent is powerful; in Week 9 we run *several* agents together — that's `multi-agent-systems`."

---
name: tool-use
description: "Teach how an LLM uses tools (function calling) to act on the world — define a tool schema, detect when the model wants to call it, run the function, feed the result back, and return errors (is_error) so it can recover. Covers parallel tool calls, tool_choice forcing, and validating a tool definition. Use when someone asks 'how does the AI call my code / APIs?', wants to give Claude abilities (weather, calculator, web search, smart-home), is confused about tool_use / tool_result blocks, asks how to write a good tool description, or is reviewing Week 3."
when_to_use: "Learner wants the model to DO things (call APIs, run code, control devices), asks about function calling / tool_use blocks, or is catching up on Week 3."
---

# Tool Use — Giving the Model Hands (Week 3)

> **The one idea:** The model can't run code or hit an API itself. Instead it *asks you* to, by emitting a `tool_use` block. You run the function and hand back a `tool_result`. The model decides **what** to call; **your code** does the calling.

This is the single most important leap from "chatbot" to "agent."

---

## The tool-use loop (memorize this shape)

```
1. You send: messages + a list of tool definitions
2. Model replies with a `tool_use` block:  "call get_weather(city='Bangkok')"
3. YOUR code runs the real function → gets "32°C"
4. You send the answer back as a `tool_result` block
5. Model reads it and writes the final human answer
   (repeat 2–4 if it wants more tools)
```

---

## 1. Define a tool (it's just JSON describing a function)

A tool definition is a name + description + a JSON Schema of its inputs. The **description is the prompt** — the model decides whether to call the tool based on it, so write it well.

```python
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Bangkok'"}
            },
            "required": ["city"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Use for any math.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "e.g. '500000 * 0.2'"}
            },
            "required": ["expression"],
        },
    },
]
```

## 2. Implement what each tool actually does

The model never runs this — you do. It's plain Python.

```python
import requests

def execute_tool(name, inputs):
    if name == "calculate":
        # ⚠️ eval() is for the demo only. Use simpleeval / ast in real code.
        return str(eval(inputs["expression"]))
    if name == "get_weather":
        city = inputs["city"]
        geo = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        ).json()["results"][0]
        w = requests.get(
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={geo['latitude']}&longitude={geo['longitude']}&current=temperature_2m"
        ).json()["current"]
        return f"Weather in {city}: {w['temperature_2m']}°C"
    return f"Unknown tool: {name}"
```

## 3. The loop that ties it together

```python
import anthropic
from dotenv import load_dotenv
load_dotenv()
client = anthropic.Anthropic()

def ask(question):
    messages = [{"role": "user", "content": question}]
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )
        # Did the model ask for any tools?
        tool_calls = [b for b in response.content if b.type == "tool_use"]
        if not tool_calls:
            return response.content[0].text       # no tools → final answer

        # Record the model's turn (including its tool request)
        messages.append({"role": "assistant", "content": response.content})

        # Run each requested tool and send results back
        for call in tool_calls:
            print(f"  🔧 {call.name}({call.input})")
            result = execute_tool(call.name, call.input)
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": call.id,   # MUST match the call's id
                    "content": result,
                }],
            })

print(ask("What's 18% of 4,500, and what's the weather in Bangkok?"))
```

**The three things people get wrong:**
1. **Forgetting to append the assistant turn** (`response.content`) before the tool_result — the IDs won't line up.
2. **Mismatched `tool_use_id`** — each `tool_result` must carry the exact `id` of the `tool_use` it answers.
3. **Looping forever / not looping** — you `while True` because the model may want several rounds of tools before it's done.

> 📁 Class repo: `week03_tool_use/toolsuse.py` (calculator + weather + **Tapo smart-light control**), `week03_tool_use/buildsmartassistant3tools.py` (adds web search + file reading).

---

## Returning errors to the model (let it recover)

A tool call can fail — bad city name, network down, math error. **Don't crash; tell the model.** Send the failure back as a `tool_result` with `is_error: true`. The model reads the error and can retry, fix its arguments, or apologize gracefully instead of the whole program dying.

```python
try:
    result = execute_tool(call.name, call.input)
    block = {"type": "tool_result", "tool_use_id": call.id, "content": result}
except Exception as e:
    block = {                                  # hand the failure back, don't raise
        "type": "tool_result",
        "tool_use_id": call.id,
        "content": f"Tool failed: {e}",
        "is_error": True,                      # ← model sees this and can recover
    }
messages.append({"role": "user", "content": [block]})
```

Notice `week03_tool_use/toolsuse.py` already does the soft version of this — `get_weather` returns `"Could not find coordinates for city: ..."` as plain text instead of throwing. `is_error: true` is the formal signal that says *"this was a failure, not a normal answer."*

## Two more knobs (good to know, not essential day one)

| Knob | What it does | When you reach for it |
|---|---|---|
| **Parallel tool calls** | One model turn can emit *several* `tool_use` blocks at once (e.g. weather for 3 cities). Run them, return *all* their `tool_result`s in a single follow-up `user` turn. | Independent calls — no need to wait for one before the next. The loop in section 3 already handles this: it iterates over *every* `tool_call`. |
| **`tool_choice`** | Force the model's hand: `{"type": "auto"}` (default, model decides), `{"type": "any"}` (must call *some* tool), or `{"type": "tool", "name": "calculate"}` (must call *this* one). | You *know* a tool is required this turn and don't want the model to answer from memory. |

---

## Why this matters

With tools, the model stops being a text box and becomes something that can **check the weather, do exact math, search the web, read files, or switch on a real light bulb.** In Week 3 the class wired this to TP-Link Tapo bulbs — Claude literally turned the lights on/off in the room.

The agent loop (next skill) is *this exact loop* plus a system prompt, a goal, and an iteration cap.

---

## 🧪 Guided lab (offer this)

### Warm-up (5–10 min, binary pass/fail)

Add tools **one at a time** to a working loop — feel the rhythm before the flagship drill.

1. **Start with `calculate` only.** Get the section-3 loop running with a single tool. Ask "what's 17 * 23?" and watch the `🔧` print fire. **Pass** = the model calls the tool and returns `391`.
2. **Add `get_weather`.** Ask a question that needs *both* ("weather in Bangkok, and 18% of 4500?"). **Pass** = you see *two* `🔧` lines and one combined final answer.
3. **Break the `tool_use_id`** on purpose (mismatch the id), observe the error, then fix it. **Pass** = you can say in one sentence why the id must match.

Rhythm to memorize: *schema → implementation → register → run → observe the 🔧 calls.*

---

### 🏅 Skill Drill — **Tool Definition Mastery** (15–30 min, runs at $0)

> The flagship Week 3 drill. *LLMs can only use tools that are described precisely.* You'll write `validate_tool_definition()` — a linter that scores a tool dict against **6 criteria** — then run it against good and bad sample tools with a `MockLLM` (no API key, no cost).

**The 6 criteria your validator must check:**

| # | Criterion | Why the model needs it |
|---|---|---|
| 1 | `name` present and non-empty | The model references the tool by name to call it. |
| 2 | `description` is non-empty **and > 20 chars** | The description *is* the prompt — one-word descriptions get mis-called. |
| 3 | `description` explains **WHEN to use it** (e.g. contains "use" / "when" / "for") | Tells the model *which* questions trigger this tool vs. another. |
| 4 | `input_schema` is an object with `"type": "object"` | The wire format the API requires. |
| 5 | Every property under `properties` has a `type` | Untyped params produce garbage arguments. |
| 6 | `required` is a list naming params that exist in `properties` | The model needs to know what it *must* supply. |

**Starter code (paste and complete the `TODO`s):**

```python
def validate_tool_definition(tool: dict) -> dict:
    """Score a tool dict against 6 LLM-friendliness criteria.
    Returns {"score": int, "max": 6, "passed": bool, "issues": [str, ...]}."""
    issues = []
    props = tool.get("input_schema", {}).get("properties", {})

    # 1. name present and non-empty
    if not tool.get("name"):
        issues.append("missing/empty name")
    # 2. description non-empty and > 20 chars
    desc = tool.get("description", "")
    if len(desc) <= 20:
        issues.append("description too short (<=20 chars)")
    # 3. description says WHEN to use it
    if not any(w in desc.lower() for w in ("use", "when", "for any", "call this")):
        issues.append("description doesn't say WHEN to use the tool")
    # 4. input_schema is an object
    if tool.get("input_schema", {}).get("type") != "object":
        issues.append("input_schema is not type 'object'")
    # 5. every property has a type           # TODO: loop props, flag any missing "type"
    for pname, spec in props.items():
        if "type" not in spec:
            issues.append(f"property '{pname}' has no type")
    # 6. required lists real properties      # TODO: flag required names not in properties
    for r in tool.get("input_schema", {}).get("required", []):
        if r not in props:
            issues.append(f"required field '{r}' is not a declared property")

    score = max(0, 6 - len(issues))
    return {"score": score, "max": 6, "passed": len(issues) == 0, "issues": issues}


# --- A MockLLM so the drill runs with no API key, no cost ---
class MockLLM:
    """Fakes a model that refuses to call malformed tools.
    Mirrors real behavior: a vague/typeless tool gets skipped or mis-called."""
    def would_call(self, tool: dict, question: str) -> bool:
        report = validate_tool_definition(tool)
        return report["passed"]   # a real model is unreliable on bad schemas; we make it binary


# === Sample tools to grade ===
GOOD = {   # mirrors week07_mcp_and_skills/agenttooldt.py — a clean, well-described single tool
    "name": "get_current_datetime",
    "description": "Returns the current date and time as a formatted string. Use this for any question about the current time or today's date.",
    "input_schema": {
        "type": "object",
        "properties": {
            "date_format": {"type": "string", "description": "strftime format, e.g. '%H:%M:%S'"}
        },
        "required": [],
    },
}
BAD = {     # vague description, untyped param, phantom required field
    "name": "weather",
    "description": "weather",                      # too short, no WHEN
    "input_schema": {
        "type": "object",
        "properties": {"city": {"description": "the city"}},   # no "type"
        "required": ["location"],                  # 'location' isn't a property
    },
}

for label, tool in [("GOOD", GOOD), ("BAD", BAD)]:
    r = validate_tool_definition(tool)
    print(f"{label}: {r['score']}/{r['max']}  passed={r['passed']}  {r['issues']}")
    print(f"   MockLLM would call it: {MockLLM().would_call(tool, 'what time is it?')}")
```

**Expected:** `GOOD` scores `6/6` and the MockLLM calls it; `BAD` scores ~`2/6` with 4 issues and the MockLLM refuses. Then **fix `BAD`** field-by-field until it reaches `6/6` and the MockLLM flips to `True` — that loop *is* the muscle.

**Weighted evaluation criteria:**

| Criterion | Weight | What "pass" looks like |
|---|---|---|
| All 6 checks implemented and correct | 30% | Each criterion above is actually tested |
| `GOOD` scores 6/6, `BAD` scores < 6/6 | 25% | Validator discriminates good from bad |
| Issue messages are specific and actionable | 15% | "property 'city' has no type", not "bad schema" |
| MockLLM gates calls on the report (runs at $0) | 15% | No API key needed; behavior tracks the score |
| You fix `BAD` to 6/6 by editing the dict | 15% | Demonstrates you can *author* a clean tool |

**Pass threshold: 4 / 5 criteria** (and `BAD` must end at 6/6).

> 📁 Related artifact: `week07_mcp_and_skills/agenttooldt.py` — a single, well-formed tool (`get_current_datetime`: clear name, WHEN-style description, typed param with a default, `required: []`). Run your validator against *its* `tools` list; it should score 6/6.

**Companion (optional):** repeat the drill by authoring a `get_weather` tool from scratch and validating it — reinforces writing LLM-friendly schemas, the exact skill behind every MCP tool you'll build in `mcp-and-skills` (the build-an-MCP-server section). For more reps in this format, see `agent-drills`.

---

### Stretch

Point the learner at the real Tapo example in `week03_tool_use/toolsuse.py` and discuss how `control_lights` is just another tool whose "function" happens to be an HTTP call to a smart bulb — *and* run `validate_tool_definition()` on it to confirm its `enum`-rich, well-described schema scores 6/6. (Class homework stubs `week03_tool_use/hwadd4thtool.py` / `hwadd5thtool.py` are blank on purpose — the "write your own 4th/5th tool" exercise.)

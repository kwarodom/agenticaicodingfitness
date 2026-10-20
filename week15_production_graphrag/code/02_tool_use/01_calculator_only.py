#!/usr/bin/env python3
"""Lesson 02.1 — Tool use with a single tool.

Run:  python week15/code/02_tool_use/01_calculator_only.py

The model can't run code itself. It emits a `tool_use` block asking YOU to run
`calculate(...)`. You run it and hand back a `tool_result`. The model decides
WHAT to call; your code does the calling.

Watch the 🔧 line fire when the model decides it needs the tool.
"""
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import anthropic
from simple_math import safe_eval  # tiny safe evaluator (no raw eval)

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"

# 1) Define the tool: name + description + JSON schema of inputs.
#    The DESCRIPTION is the prompt — the model uses it to decide when to call.
tools = [
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
    }
]


# 2) Implement what the tool actually does (plain Python; the model never runs this).
def execute_tool(name, inputs):
    if name == "calculate":
        return safe_eval(inputs["expression"])
    return f"Unknown tool: {name}"


# 3) The loop that ties it together.
def ask(question):
    messages = [{"role": "user", "content": question}]
    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=1024, tools=tools, messages=messages
        )
        tool_calls = [b for b in response.content if b.type == "tool_use"]
        if not tool_calls:
            return next((b.text for b in response.content if b.type == "text"), "")

        messages.append({"role": "assistant", "content": response.content})
        for call in tool_calls:
            print(f"  🔧 {call.name}({call.input})")
            result = execute_tool(call.name, call.input)
            print(f"  📋 {result}")
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": call.id,  # MUST match the call's id
                    "content": str(result),
                }],
            })


if __name__ == "__main__":
    print(ask("What's 17 * 23? And then what's that minus 100?"))

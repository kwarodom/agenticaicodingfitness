#!/usr/bin/env python3
"""Lesson 03.2 — The Code Review Agent (Week 5's flagship example).

Run:  python week15/code/03_agent_loops/02_code_review_agent.py

Same Agent class, different tools + playbook. It reads a buggy file, fixes it,
writes the fix, and runs it to verify — looping until it works. The SYSTEM
PROMPT is the agent's playbook; the loop just keeps asking "what's next?".

It operates on buggy_sample.py (created fresh each run so you can re-run it).
"""
from pathlib import Path
import subprocess
import sys
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from agent import Agent

HERE = Path(__file__).parent
SAMPLE = HERE / "buggy_sample.py"

BUGGY_CODE = '''\
def average(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(number)   # bug: 'number' should be 'numbers'

def greet(name)                  # bug: missing colon
    print("Hello, " + name)

print(average([10, 20, 30]))
greet("Aom")
'''


def write_fresh_sample():
    SAMPLE.write_text(BUGGY_CODE)


tools = [
    {"name": "read_file", "description": "Read a file's contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file (overwrites).",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    {"name": "run_python", "description": "Run a Python file; returns stdout/stderr.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
]


def execute_tool(name, inputs):
    if name == "read_file":
        try:
            return Path(inputs["path"]).read_text()
        except FileNotFoundError:
            return f"Error: file not found: {inputs['path']}"
    if name == "write_file":
        Path(inputs["path"]).write_text(inputs["content"])
        return f"Wrote {inputs['path']}"
    if name == "run_python":
        r = subprocess.run([sys.executable, inputs["path"]],
                           capture_output=True, text=True, timeout=15)
        return f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}\nreturn code: {r.returncode}"
    return f"Unknown tool: {name}"


if __name__ == "__main__":
    write_fresh_sample()
    print(f"Created buggy file: {SAMPLE}")

    agent = Agent(
        system_prompt="""You are a code review agent. Your process:
1. Read the target file.
2. Identify bugs and style issues.
3. Write a corrected version of the file.
4. Run it to verify it works; if it errors, fix and retry.
When it runs cleanly, briefly explain what you fixed.""",
        tools=tools,
        tool_executor=execute_tool,
        max_iterations=12,
    )
    result = agent.run(f"Review and fix the file '{SAMPLE}'. Fix all bugs so it runs without errors.")
    print("\n=== AGENT SUMMARY ===")
    print(result)
    print(f"\nFixed file is at: {SAMPLE} (open it to see the result)")

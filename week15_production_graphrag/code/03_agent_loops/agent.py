"""The reusable Agent class — the heart of the course (Week 5).

An agent is a LOOP: REASON (decide next step) → ACT (call a tool) →
OBSERVE (read result) → repeat, until it declares itself done.

Two things make this an agent and not a chatbot:
  1. the loop with a stop condition (end_turn + no tool calls = done)
  2. max_iterations — the seatbelt so a confused agent can't loop forever
"""
import json
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import anthropic

client = anthropic.Anthropic()


class Agent:
    def __init__(self, system_prompt, tools, tool_executor, model="claude-sonnet-4-6", max_iterations=10):
        self.system_prompt = system_prompt
        self.tools = tools
        self.tool_executor = tool_executor      # your execute_tool(name, inputs)
        self.model = model
        self.max_iterations = max_iterations

    def run(self, goal):
        print(f"\n🎯 Goal: {goal}\n")
        messages = [{"role": "user", "content": goal}]

        for i in range(self.max_iterations):
            print(f"--- Iteration {i + 1} ---")

            # REASON + ACT: ask the model what to do next
            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.system_prompt,
                tools=self.tools,
                messages=messages,
            )

            has_tool_use = any(b.type == "tool_use" for b in response.content)
            text = [b.text for b in response.content if b.type == "text"]
            for t in text:
                print(f"  💭 {t[:200]}")

            # DONE? finished its turn with no tool calls = goal complete
            if response.stop_reason == "end_turn" and not has_tool_use:
                print(f"\n✅ Done in {i + 1} iterations")
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

        print(f"\n⚠️  Max iterations ({self.max_iterations}) reached")
        return "Max iterations reached"

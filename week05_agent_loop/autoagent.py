# PYTHON — agent.py — The Core Agent Framework
import anthropic
import json
import subprocess
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

class Agent:
    def __init__(self, system_prompt, tools, tool_executor, max_iterations=10):
        self.system_prompt = system_prompt
        self.tools = tools
        self.tool_executor = tool_executor
        self.max_iterations = max_iterations
        self.messages = []
        self.iteration = 0
    
    def run(self, goal):
        """The core agent loop"""
        print(f"\n🎯 Agent Goal: {goal}\n")
        self.messages = [{"role": "user", "content": goal}]
        
        for i in range(self.max_iterations):
            self.iteration = i + 1
            print(f"--- Iteration {self.iteration} ---")
            
            # REASON + ACT: Ask Claude what to do
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=self.system_prompt,
                tools=self.tools,
                messages=self.messages
            )
            
            # Check response
            has_tool_use = any(b.type == "tool_use" for b in response.content)
            text_blocks = [b.text for b in response.content if b.type == "text"]
            
            # Print any reasoning
            for text in text_blocks:
                print(f"  💭 {text[:200]}")
            
            # DONE check: if stop_reason is "end_turn" and no tool calls
            if response.stop_reason == "end_turn" and not has_tool_use:
                print(f"\n✅ Agent finished in {self.iteration} iterations")
                return text_blocks[-1] if text_blocks else "Done"
            
            # EXECUTE tool calls
            self.messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  🔧 {block.name}({json.dumps(block.input)[:100]})")
                    result = self.tool_executor(block.name, block.input)
                    print(f"  📋 Result: {str(result)[:150]}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result)
                    })
            
            self.messages.append({"role": "user", "content": tool_results})
        
        print(f"\n⚠️ Max iterations ({self.max_iterations}) reached")
        return "Max iterations reached"

# === CODE REVIEW AGENT ===
code_review_tools = [
    {
        "name": "read_file",
        "description": "Read contents of a Python file",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "run_python",
        "description": "Run a Python file and return stdout/stderr",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }
    },
    {
        "name": "run_lint",
        "description": "Run flake8 linter on a Python file",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }
    }
]

def execute_code_tool(name, inputs):
    if name == "read_file":
        try:
            with open(inputs["path"]) as f:
                return f.read()
        except FileNotFoundError:
            return f"Error: File not found: {inputs['path']}"
    elif name == "write_file":
        with open(inputs["path"], "w") as f:
            f.write(inputs["content"])
        return f"Written to {inputs['path']}"
    elif name == "run_python":
        result = subprocess.run(
            ["python", inputs["path"]],
            capture_output=True, text=True, timeout=10
        )
        return f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}\nReturn code: {result.returncode}"
    elif name == "run_lint":
        result = subprocess.run(
            ["python", "-m", "flake8", inputs["path"]],
            capture_output=True, text=True
        )
        return result.stdout or "No lint issues found!"

# Create the agent
agent = Agent(
    system_prompt="""You are a code review agent. Your process:
1. Read the target file
2. Run the linter to find issues  
3. Analyze the code for bugs, style issues, and improvements
4. Write a fixed version of the file
5. Run the file to verify it works
6. If there are errors, fix them and try again
When everything is clean and working, explain what you fixed.""",
    tools=code_review_tools,
    tool_executor=execute_code_tool,
    max_iterations=10
)

# Run on a test file
result = agent.run("Review and fix the file 'sample.py'. Fix all bugs and style issues.")
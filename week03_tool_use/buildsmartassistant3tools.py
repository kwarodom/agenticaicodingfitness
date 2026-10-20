import os
import json
import requests
import simpleeval
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic()

# === Define tools ===
tools = [
    {
        "name": "web_search",
        "description": "Call a search API to find information on the web. Returns search results or summaries.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate",
        "description": "Safely evaluate a mathematical expression.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g. '2 + 2' or '500000 * 0.2'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a local file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The relative or absolute path to the file to read."
                }
            },
            "required": ["file_path"]
        }
    }
]

# === Tool implementations ===
def execute_tool(name, inputs):
    try:
        if name == "web_search":
            query = inputs["query"]
            # Using DuckDuckGo Instant Answer API
            resp = requests.get(f"https://api.duckduckgo.com/?q={query}&format=json")
            resp.raise_for_status()
            data = resp.json()
            
            # DuckDuckGo's API returns abstract or related topics
            abstract = data.get("AbstractText")
            if abstract:
                return abstract
            elif data.get("RelatedTopics"):
                return data["RelatedTopics"][0].get("Text", "No abstract found, but found related topics.")
            else:
                return f"No simple summary found on DuckDuckGo for '{query}'."
                
        elif name == "calculate":
            # Safely evaluate using simpleeval
            result = simpleeval.simple_eval(inputs["expression"])
            return str(result)
            
        elif name == "read_file":
            file_path = inputs["file_path"]
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
                
    except Exception as e:
        # Step 6: Return helpful error message if tool fails
        return f"Error executing tool '{name}': {str(e)}"

# === Conversation loop with tool use ===
def ask(question):
    messages = [{"role": "user", "content": question}]
    print(f"\nUser: {question}")
    
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929", # Using valid model
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        
        # Check if Claude wants to use tools
        tool_calls = [b for b in response.content if b.type == "tool_use"]
        
        if not tool_calls:
            # No tool calls — return final text response
            return response.content[0].text
            
        messages.append({"role": "assistant", "content": response.content})
        
        for tool_call in tool_calls:
            print(f"  🔧 Using tool: {tool_call.name}({tool_call.input})")
            result = execute_tool(tool_call.name, tool_call.input)
            
            # Truncate result if it's too long to avoid token limits
            if len(str(result)) > 3000:
                result = str(result)[:3000] + "\n... [TRUNCATED]"
                
            messages.append({
                "role": "user",
                "content": [{"type": "tool_result",
                             "tool_use_id": tool_call.id,
                             "content": str(result)}]
            })

if __name__ == "__main__":
    # Test compound queries
    print("Agent Response:\n", ask("What is the GDP of Thailand? Multiply it by 1.05."))
    print("Agent Response:\n", ask("Read the README.md file and count the number of lines."))
    # Test error handling
    print("Agent Response:\n", ask("Read a file that does not exist like 'nonexistent_file.txt'"))

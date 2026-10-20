# PYTHON — addingmcp.py — Using Claude with MCP tools
import anthropic
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
client = anthropic.Anthropic()
model = "claude-haiku-4-5"

def add_user_message(messages, text):
    user_message = {"role": "user", "content": text}
    messages.append(user_message)

def add_assistant_message(messages, text):
    assistant_message = {"role": "assistant", "content": text}
    messages.append(assistant_message)

def chat(messages):
    message = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=messages,
    )
    return message.content[0].text


# In Python, to use MCP servers with Claude, you have two options:
#
# Option 1: Use the `mcp` Python SDK to connect to an MCP server,
#           list its tools, and pass them to the Anthropic API manually.
#
# Option 2: Use a simple tool-calling loop like below (no MCP needed).

# --- Example: Simple tool-calling agent loop ---
def get_current_datetime(date_format="%Y-%m-%d %H:%M:%S"):
    if not date_format:
        raise ValueError("date_format cannot be empty")
    return datetime.now().strftime(date_format)

tools = [
    {
        "name": "get_current_datetime",
        "description": "Returns the current date and time as a formatted string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date_format": {
                    "type": "string",
                    "description": 'Python strftime format string. e.g. "%Y-%m-%d %H:%M:%S"',
                    "default": "%Y-%m-%d %H:%M:%S"
                }
            },
            "required": []
        }
    }
]

# Map tool names to actual functions
tool_functions = {
    "get_current_datetime": get_current_datetime,
}

def agent_loop(user_prompt):
    messages = [{"role": "user", "content": user_prompt}]

    while True:
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            messages=messages,
            tools=tools,
        )

        # If the model wants to use a tool, execute it and continue
        if response.stop_reason == "tool_use":
            # Add assistant's response (which includes tool_use blocks)
            messages.append({"role": "assistant", "content": response.content})

            # Process each tool call
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    func = tool_functions[block.name]
                    result = func(**block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })

            messages.append({"role": "user", "content": tool_results})

        else:
            # Model gave a final text response
            final_text = next(
                (block.text for block in response.content if hasattr(block, "text")),
                ""
            )
            print(final_text)
            return final_text

# Run it
agent_loop("What is the exact time right now, formatted as HH:MM:SS?")

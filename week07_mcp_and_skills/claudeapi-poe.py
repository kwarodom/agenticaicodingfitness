# PYTHON — agent.py — The Core Agent Framework
import os
import anthropic
from datetime import datetime
from duckduckgo_search import DDGS
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(
    base_url="https://api.poe.com",
    api_key=os.environ.get("POE_API_KEY")
)
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

#make initial list of messages
messages = []

# # use while true loop to run chatbot forever
# while True:
#   # 1. get user input
#   user_input = input(">: ")
#   print(f">: {user_input}")
# 
#   # 2. add user input to the list of messages
#   add_user_message(messages, user_input)
# 
#   # 3. call Claude with the chat function
#   answer = chat(messages)
# 
#   # 4. add Claude's response to the list of messages
#   add_assistant_message(messages, answer)
# 
#   # 5. print generated text
#   print(f">: {answer}")

# system_prompt = """
# You are a patient math tutor.
# Do not directly answer a student's questions.
# Guide them to a solution step by step.
# """
# messages = [
#         {
#             "role": "user",
#             "content": "help me solve 5x+2 = 10"
#         }
# ]
# model = "claude-sonnet-4-6"
# 
# def chat(messages, system=None):
#     params = {
#         "model": model,
#         "max_tokens": 1000,
#         "messages": messages,
#     }
# 
#     if system:
#         params["system"] = system
# 
#     message = client.messages.create(**params)
#     return message.content[0].text
# 
# # use while true loop to run chatbot forever
# while True:
#   # 1. get user input
#   user_input = input(">: ")
#   print(f">: {user_input}")
# 
#   # 2. add user input to the list of messages
#   add_user_message(messages, user_input)
# 
#   # 3. call Claude with the chat function
#   answer = chat(messages, system_prompt)
# 
#   # 4. add Claude's response to the list of messages
#   add_assistant_message(messages, answer)
# 
#   # 5. print generated text
#   print(f">: {answer}")


system_prompt = """
You are a patient math tutor.
Do not directly answer a student's questions.
Guide them to a solution step by step.
"""
messages = [
        {
            "role": "user",
            "content": "what time is it now?"
        }
]
model = "claude-sonnet-4-6"


def get_current_time() -> str:
    """Returns the current date and time."""
    print(" [Executing Tool: get_current_time] ")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def search_web(query: str) -> str:
    """Uses DuckDuckGo to search the internet for real-time information."""
    print(f" [Executing Tool: search_web for '{query}'] ")
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
    if not results: return "No results found."
    return "\n".join([f"- {r['title']}: {r['body']}" for r in results])

tools = [
    {
        "name": "get_current_time",
        "description": "Retrieves the current local date and time.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "search_web",
        "description": "Searches the internet for real-time information.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "search query"}},
            "required": ["query"]
        }
    }
]

def chat(messages, system=None, temperature=1.0, stop_sequences=None):
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature,
        "tools": tools
    }

    if system:
        params["system"] = system

    if stop_sequences:
        params["stop_sequences"] = stop_sequences

    response = client.messages.create(**params)
    
    if response.stop_reason == "tool_use":
        messages.append({"role": "assistant", "content": response.content})
        tool_call = next(block for block in response.content if block.type == "tool_use")
        
        if tool_call.name == "get_current_time":
            tool_result = get_current_time()
        elif tool_call.name == "search_web":
            tool_result = search_web(tool_call.input["query"])
        else:
            tool_result = f"Error: Unknown tool {tool_call.name}"
            
        messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_call.id, "content": tool_result}]
        })
        
        return chat(messages, system, temperature, stop_sequences)

    return response.content[0].text

# Low temperature - more predictable
#answer = chat(messages, temperature=0.0)
#print(answer)

# High temperature - more creative
answer = chat(messages, temperature=0.0)
print(answer)
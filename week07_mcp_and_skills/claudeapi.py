# PYTHON — agent.py — The Core Agent Framework
import anthropic
from dotenv import load_dotenv
from datetime import datetime
from anthropic.types import ToolParam

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

#make initial list of messages
messages = []

def get_current_datetime(date_format="%Y-%m-%d %H:%M:%S"):
    if not date_format:
        raise ValueError("date_format cannot be empty")
    return datetime.now().strftime(date_format)

get_current_datetime_schema = ToolParam({
  "name": "get_current_datetime",
  "description": "Returns the current date and time as a formatted string. Useful for answering questions about the current date, time, day of the week, or any time-related query. The output format is controlled by a Python strftime format string.",
  "input_schema": {
    "type": "object",
    "properties": {
      "date_format": {
        "type": "string",
        "description": "Python strftime format string controlling the output. Common patterns: \"%Y-%m-%d %H:%M:%S\" (full datetime), \"%Y-%m-%d\" (date only), \"%H:%M:%S\" (time only), \"%A\" (day of week). Must not be empty.",
        "default": "%Y-%m-%d %H:%M:%S"
      }
    },
    "required": []
  }
})

messages.append({
    "role": "user",
    "content": "What is the exact time, formatted as HH:MM:SS?"
})

response = client.messages.create(
    model=model,
    max_tokens=1000,
    messages=messages,
    tools=[get_current_datetime_schema],
)

messages.append({
    "role": "assistant",
    "content": response.content
})

print(messages)

#!/usr/bin/env python3
"""Lesson 02.3 — Write your OWN third tool (the Week 3 homework).

Run:  python week15/code/02_tool_use/03_add_your_own_tool.py

This adds a `roll_dice` tool to the previous two. The rhythm to internalize:
    schema  →  implementation  →  register  →  run  →  observe the 🔧 calls

YOUR TURN: add a 4th tool. Ideas: get_time(), reverse_text(text),
read_file(path). Copy the 3-step pattern below for roll_dice.

Want to feel mistake #2 from the skill? Change `call.id` to a fake string in
the tool_result and rerun — you'll get a tool_use_id mismatch error. Then fix it.
"""
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import random
import anthropic
import requests
from simple_math import safe_eval

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"

# --- STEP 1: schema (add your new tool's JSON here) ---
tools = [
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Use for any math.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get the current temperature for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    {
        "name": "roll_dice",  # <-- our new tool
        "description": "Roll one or more dice and return the results. Use when the user wants a random roll.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sides": {"type": "integer", "description": "Number of sides per die, e.g. 6 or 20"},
                "count": {"type": "integer", "description": "How many dice to roll (default 1)"},
            },
            "required": ["sides"],
        },
    },
]


def get_weather(city):
    try:
        geo = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                           params={"name": city, "count": 1}, timeout=10).json()
        if not geo.get("results"):
            return f"Could not find {city}."
        loc = geo["results"][0]
        cur = requests.get("https://api.open-meteo.com/v1/forecast",
                           params={"latitude": loc["latitude"], "longitude": loc["longitude"],
                                   "current": "temperature_2m"}, timeout=10).json()["current"]
        return f"Weather in {city}: {cur['temperature_2m']}°C"
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"


# --- STEP 2: implementation ---
def roll_dice(sides, count=1):
    rolls = [random.randint(1, int(sides)) for _ in range(int(count))]
    return f"Rolled {count}d{sides}: {rolls} (total {sum(rolls)})"


# --- STEP 3: register in the dispatcher ---
def execute_tool(name, inputs):
    if name == "calculate":
        return safe_eval(inputs["expression"])
    if name == "get_weather":
        return get_weather(inputs["city"])
    if name == "roll_dice":
        return roll_dice(inputs["sides"], inputs.get("count", 1))
    return f"Unknown tool: {name}"


def ask(question):
    messages = [{"role": "user", "content": question}]
    while True:
        response = client.messages.create(model=MODEL, max_tokens=1024, tools=tools, messages=messages)
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
                "content": [{"type": "tool_result", "tool_use_id": call.id, "content": str(result)}],
            })


if __name__ == "__main__":
    print(ask("Roll two 20-sided dice for me, then tell me the weather in Tokyo."))

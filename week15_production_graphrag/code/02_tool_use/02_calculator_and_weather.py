#!/usr/bin/env python3
"""Lesson 02.2 — Two tools, and a question that needs both.

Run:  python week15/code/02_tool_use/02_calculator_and_weather.py

Ask one question that needs math AND weather. Watch the model call multiple
tools (sometimes across several rounds) — the `while True` loop handles it.

`get_weather` hits the free Open-Meteo API (no key needed).
"""
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import anthropic
import requests
from simple_math import safe_eval

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"

tools = [
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Use for any math.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "e.g. '4500 * 0.18'"}},
            "required": ["expression"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get the current temperature for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name, e.g. 'Bangkok'"}},
            "required": ["city"],
        },
    },
]


def get_weather(city):
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}, timeout=10,
        ).json()
        if not geo.get("results"):
            return f"Could not find coordinates for {city}."
        loc = geo["results"][0]
        cur = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": loc["latitude"], "longitude": loc["longitude"],
                    "current": "temperature_2m"}, timeout=10,
        ).json()["current"]
        return f"Weather in {city}: {cur['temperature_2m']}°C"
    except Exception as e:  # noqa: BLE001
        return f"Error getting weather: {e}"


def execute_tool(name, inputs):
    if name == "calculate":
        return safe_eval(inputs["expression"])
    if name == "get_weather":
        return get_weather(inputs["city"])
    return f"Unknown tool: {name}"


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
                "content": [{"type": "tool_result", "tool_use_id": call.id, "content": str(result)}],
            })


if __name__ == "__main__":
    print(ask("What's the temperature in Bangkok right now, and what is 18% of 4,500?"))

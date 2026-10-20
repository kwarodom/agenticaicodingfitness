#!/usr/bin/env python3
"""Lesson 03.1 — Your first autonomous agent.

Run:  python week15/code/03_agent_loops/01_weather_agent.py

Same tools as folder 02, but now wrapped in the Agent loop with a GOAL that
needs several steps. Read the 💭 / 🔧 / 📋 trace: each cycle is one
REASON → ACT → OBSERVE. This is where "what is an agent" finally clicks.

EXPERIMENT: change max_iterations=1 below, rerun, and watch it fail to finish —
proof of why the seatbelt exists. Then put it back to 10.
"""
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import requests
from agent import Agent

tools = [
    {
        "name": "get_weather",
        "description": "Get the current temperature for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    }
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
        return f"{city}: {cur['temperature_2m']}°C"
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"


def execute_tool(name, inputs):
    if name == "get_weather":
        return get_weather(inputs["city"])
    return f"Unknown tool: {name}"


if __name__ == "__main__":
    agent = Agent(
        system_prompt="You are a helpful weather assistant. Use the get_weather tool to gather facts before answering.",
        tools=tools,
        tool_executor=execute_tool,
        max_iterations=10,  # <-- try setting this to 1 to watch it fail
    )
    result = agent.run(
        "Find the temperature in Bangkok and Tokyo, then tell me which is hotter and by how many degrees."
    )
    print("\n=== FINAL ANSWER ===")
    print(result)

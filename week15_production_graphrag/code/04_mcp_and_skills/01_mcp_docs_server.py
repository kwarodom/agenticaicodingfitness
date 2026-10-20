#!/usr/bin/env python3
"""Lesson 04.1 — Use an MCP server via the Claude Agent SDK.

Run:  python week15/code/04_mcp_and_skills/01_mcp_docs_server.py
Needs: pip install claude-agent-sdk  (+ network access)

MCP = "USB-C for AI tools". Instead of hand-coding tools (folder 02), you
connect to an MCP server that already exposes them. Here we connect the live
Claude Code docs server and let the agent answer from it.

Notice:
  - mcp_servers registers a server by name (here, a remote HTTP server).
  - tools are namespaced mcp__<server>__<tool>; allowed_tools is your safety gate.
  - the Agent SDK runs the WHOLE agent loop for you (the loop you hand-built
    in folder 03) — you just supply the prompt and the servers.
"""
import asyncio
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

try:
    from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage
except ImportError:
    raise SystemExit("Install the SDK first:  pip install claude-agent-sdk")


async def main():
    options = ClaudeAgentOptions(
        mcp_servers={
            "claude-code-docs": {
                "type": "http",
                "url": "https://code.claude.com/docs/mcp",
            }
        },
        allowed_tools=["mcp__claude-code-docs__*"],  # only let it touch this server
    )

    async for message in query(
        prompt="Use the docs MCP server to explain, in 3 bullets, what hooks are in Claude Code.",
        options=options,
    ):
        if isinstance(message, ResultMessage) and message.subtype == "success":
            print(message.result)


if __name__ == "__main__":
    asyncio.run(main())

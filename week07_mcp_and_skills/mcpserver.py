# PYTHON — addingmcp.py — Using Claude with MCP tools
import anthropic
from dotenv import load_dotenv
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

load_dotenv()
client = anthropic.Anthropic()
model = "claude-haiku-4-5"

async def main():
    options = ClaudeAgentOptions(
        mcp_servers={
            "claude-code-docs": {
                "type": "http",
                "url": "https://code.claude.com/docs/mcp",
            }
        },
        allowed_tools=["mcp__claude-code-docs__*"],
    )

    async for message in query(
        prompt="Use the docs MCP server to explain what hooks are in Claude Code",
        options=options,
    ):
        if isinstance(message, ResultMessage) and message.subtype == "success":
            print(message.result)


asyncio.run(main())
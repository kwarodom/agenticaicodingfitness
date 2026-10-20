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
            "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    "/Users/altodev5/AgenticCoding",
                ],
            }
        },
        allowed_tools=["mcp__filesystem__*"],
    )

    async for message in query(prompt="List files in my project", options=options):
        if isinstance(message, ResultMessage) and message.subtype == "success":
            print(message.result)


asyncio.run(main())
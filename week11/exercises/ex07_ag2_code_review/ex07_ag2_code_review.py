"""
Exercise 07 — AutoGen Code Review Committee (Intermediate).

NOTE: PDF uses old AutoGen 0.2 API (autogen.AssistantAgent, GroupChat). This
file uses the modern AutoGen v0.4+ split-package API
(autogen_agentchat + autogen_ext) which is what's installed.

Three specialist agents — CodeReviewer, SecurityAuditor, TestWriter — each
analyse a Python function in round-robin order. Demonstrates GroupChat-style
turn-taking with deterministic speaker selection.

Key concepts:
    - AssistantAgent with focused system_message persona
    - RoundRobinGroupChat — deterministic turn order
    - MaxMessageTermination — stop condition
    - AnthropicChatCompletionClient as model backend
"""

import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.anthropic import AnthropicChatCompletionClient

load_dotenv()


CODE = '''
def energy_roi(baseline_kwh, optimised_kwh, rate=0.12):
    savings = baseline_kwh - optimised_kwh
    cost = savings * rate
    pct = (savings / baseline_kwh) * 100
    return {"savings_kwh": savings, "cost_usd": cost, "pct": pct}
result = energy_roi(50000, 35000)
print(result)
'''


async def main() -> None:
    model_client = AnthropicChatCompletionClient(
        model="claude-haiku-4-5-20251001",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )

    reviewer = AssistantAgent(
        name="CodeReviewer",
        model_client=model_client,
        system_message=(
            "Review Python code quality: naming, edge cases, "
            "maintainability. End with 'REVIEW SCORE: X/10'."
        ),
    )

    auditor = AssistantAgent(
        name="SecurityAuditor",
        model_client=model_client,
        system_message=(
            "Audit for: input validation, division-by-zero risk, "
            "type safety. End with 'SECURITY SCORE: X/10'."
        ),
    )

    tester = AssistantAgent(
        name="TestWriter",
        model_client=model_client,
        system_message=(
            "Write 3 pytest unit tests covering normal and edge cases. "
            "Output only the test code."
        ),
    )

    team = RoundRobinGroupChat(
        participants=[reviewer, auditor, tester],
        termination_condition=MaxMessageTermination(max_messages=4),
    )

    task = f"Please review this energy function:\n```python{CODE}```"
    print("=" * 60)
    print("AutoGen Code Review Committee — round-robin")
    print("=" * 60)

    await Console(team.run_stream(task=task))
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())

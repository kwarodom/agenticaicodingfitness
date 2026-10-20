"""
Exercise 01 — Two-Agent Dialogue (Beginner).

The simplest possible multi-agent system: two Claude instances with separate
conversation histories that alternate turns. A Teacher explains a Python
concept; a Student asks clarifying questions. Pure Anthropic SDK, no
framework.

Key concepts:
    - Separate agent state via independent message histories
    - Synchronous turn-taking loop
    - System prompt as agent personality
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = "claude-haiku-4-5-20251001"


def agent_turn(system: str, history: list, user_msg: str) -> str:
    """Append user_msg, call Claude, append reply, return reply."""
    history.append({"role": "user", "content": user_msg})
    resp = client.messages.create(
        model=MODEL,
        max_tokens=200,
        system=system,
        messages=history,
    )
    reply = resp.content[0].text
    history.append({"role": "assistant", "content": reply})
    return reply


teacher_sys = (
    "You are an expert Python teacher. Explain concepts clearly and "
    "concisely (2-3 sentences). End with a check-understanding question."
)
student_sys = (
    "You are a curious beginner Python student. Ask one specific "
    "follow-up question based on what the teacher just said. Keep it short."
)

teacher_hist: list = []
student_hist: list = []
TOPIC = "Python list comprehensions"

if __name__ == "__main__":
    print(f"=== Two-Agent Dialogue: {TOPIC} ===\n")
    student_question = f"Can you explain {TOPIC}?"

    for turn in range(3):
        print(f"[Turn {turn + 1}]")

        teacher_reply = agent_turn(teacher_sys, teacher_hist, student_question)
        print(f"Teacher: {teacher_reply}\n")

        student_question = agent_turn(student_sys, student_hist, teacher_reply)
        print(f"Student: {student_question}\n")

    print("=== Dialogue complete ===")

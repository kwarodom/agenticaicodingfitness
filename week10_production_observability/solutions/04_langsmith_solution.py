# %% [markdown]
# # Ex 4 SOLUTION — LangSmith + eval dataset + accuracy report

# %%
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "w10-hands-on"

# Let the script find data/fake_tickets.py from solutions/ dir
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.fake_tickets import batch  # noqa: E402

llm = ChatOpenAI(model="openai/gpt-oss-120b:free", base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"), temperature=0)


class State(TypedDict):
    ticket: str
    category: str | None


STRONG = (
    "You are a ticket classifier. Return exactly one of: TECHNICAL, BILLING, GENERAL. "
    "Return just the word, no punctuation.\n\n"
    "Ticket: {ticket}"
)


def classify_strong(state: State) -> State:
    out = llm.invoke(STRONG.format(**state)).content.strip().upper()
    if out not in ("TECHNICAL", "BILLING", "GENERAL"):
        out = "GENERAL"
    return {"category": out}


def build():
    g = StateGraph(State)
    g.add_node("classify", classify_strong)
    g.add_edge(START, "classify")
    g.add_edge("classify", END)
    return g.compile()


if __name__ == "__main__":
    app = build()
    tickets = batch(20)
    correct = 0
    for t in tickets:
        pred = app.invoke({"ticket": t["body"], "category": None})["category"]
        label = t["true_category"]
        hit = "OK " if pred == label else "MISS"
        correct += pred == label
        print(f"  {hit}  pred={pred:<10} label={label:<10}  {t['body'][:55]}")
    print(f"\nAccuracy: {correct}/{len(tickets)} = {correct/len(tickets):.0%}")
    print("\nOpen smith.langchain.com -> project 'w10-hands-on' to see all 20 traces.")

"""
Exercise 11 — Multi-Agent Procurement Negotiation (Advanced).

Two stateful NegotiationAgent objects maintain independent histories and
alternate turns to negotiate an HVAC system purchase. Agents detect DEAL /
FAIL conditions in each other's replies. Pure Anthropic SDK with agent
classes — no framework.

Key concepts:
    - Stateful agent class with per-instance message history
    - Turn-based loop with terminal condition detection
    - Goal-oriented system prompts with explicit BATNA (walk-away price)
    - Parsing structured keywords from natural language replies
"""

import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = "claude-haiku-4-5-20251001"


class NegotiationAgent:
    def __init__(self, name: str, system: str):
        self.name = name
        self.system = system
        self.hist: list = []

    def respond(self, msg: str) -> str:
        self.hist.append({"role": "user", "content": msg})
        r = client.messages.create(
            model=MODEL,
            max_tokens=250,
            system=self.system,
            messages=self.hist,
        )
        reply = r.content[0].text
        self.hist.append({"role": "assistant", "content": reply})
        return reply


def negotiate(
    product: str, list_price: int, budget: int, rounds: int = 5
) -> None:
    buyer = NegotiationAgent(
        "AltoTech Buyer",
        f"You are AltoTech's procurement manager buying {product}. "
        f"Budget ceiling: ${budget:,}. Open at 80% of budget. "
        f"Accept any offer at or below ${budget:,}. "
        "When you accept say exactly: DEAL ACCEPTED at $<price>. "
        "If rounds exhaust say: NEGOTIATION FAILED.",
    )
    seller = NegotiationAgent(
        "HVAC Vendor",
        f"You are an HVAC vendor. List price ${list_price:,}. "
        f"Walk-away floor: ${int(list_price * 0.85):,}. "
        "Offer concessions of max 5% per round. "
        "When buyer accepts say: DEAL CONFIRMED at $<price>. "
        "Never go below your floor.",
    )

    print(f"\n=== Negotiation: {product} ===")
    print(f"List: ${list_price:,} | Budget: ${budget:,}\n")

    opening = (
        f"We need {product}. Our opening offer: ${int(budget * 0.80):,}."
    )
    print(f"Buyer: {opening}")
    msg = opening

    for rnd in range(1, rounds + 1):
        seller_reply = seller.respond(msg)
        print(f"\nVendor: {seller_reply[:200]}")
        if any(k in seller_reply for k in ("DEAL CONFIRMED", "NEGOTIATION FAILED")):
            break

        buyer_reply = buyer.respond(seller_reply)
        print(f"Buyer: {buyer_reply[:200]}")
        if any(k in buyer_reply for k in ("DEAL ACCEPTED", "NEGOTIATION FAILED")):
            break

        msg = buyer_reply

    print("\n=== End ===")


if __name__ == "__main__":
    negotiate(
        "Smart HVAC Control System (50 zones)",
        list_price=85_000,
        budget=70_000,
        rounds=5,
    )

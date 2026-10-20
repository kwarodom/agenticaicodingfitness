#!/usr/bin/env python3
"""Lesson 06.3 — Router / dispatcher (triage desk).

Run:  python week15/code/06_multi_agent/03_router.py

A classifier agent reads each support ticket and routes it to the right
specialist (billing / technical / general). Decide WHAT (classify) then HOW
(route). This is the LangGraph router pattern from week9/ex2 — here as plain
functions so the control flow is obvious.
"""
import json
import re
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"


def classify(ticket):
    """Router node: returns one of billing / technical / general."""
    prompt = (
        "Classify this support ticket into exactly one category: "
        "billing, technical, or general.\n"
        f"Ticket: {ticket}\n"
        'Respond with JSON only: {"category": "...", "confidence": 0.0-1.0}'
    )
    text = client.messages.create(
        model=MODEL, max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    ).content[0].text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group()) if match else {"category": "general", "confidence": 0.5}


SPECIALISTS = {
    "billing": "You are billing support. Be precise about charges and invoices.",
    "technical": "You are technical support for HVAC systems and IoT sensors. Be detailed.",
    "general": "You are friendly general support.",
}


def handle(ticket):
    decision = classify(ticket)
    cat = decision["category"]
    system = SPECIALISTS.get(cat, SPECIALISTS["general"])
    reply = client.messages.create(
        model=MODEL, max_tokens=300, system=system,
        messages=[{"role": "user", "content": ticket}],
    ).content[0].text
    return cat, decision.get("confidence", 0), reply


TICKETS = [
    "My last invoice has a charge I don't recognize for sensor maintenance on floor 3.",
    "The CO2 sensor in conference room B reads 2000ppm but the room is empty. Calibration issue?",
    "Can you send me the brochure for your platform?",
]

if __name__ == "__main__":
    for ticket in TICKETS:
        cat, conf, reply = handle(ticket)
        print(f"\nTicket: {ticket[:60]}...")
        print(f"  → routed to: {cat} (confidence {conf:.0%})")
        print(f"  reply: {reply[:200]}...")
        print("-" * 60)

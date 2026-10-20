"""Faker-generated support tickets. Zero real-company data."""
from __future__ import annotations

import random

from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

CATEGORIES = {
    "TECHNICAL": [
        "My dashboard won't load after the update.",
        "API returning 500 errors since this morning.",
        "Cannot connect to the device — ping times out.",
        "Report generation fails with a Python traceback.",
        "Data sync stuck at 73% for 2 hours.",
        "WebSocket keeps disconnecting every 30 seconds.",
        "PDF export shows blank pages since yesterday.",
    ],
    "BILLING": [
        "I was charged twice for last month's plan.",
        "Where is my invoice for March?",
        "How do I upgrade from Pro to Enterprise?",
        "Refund request: cancelled yesterday, still charged today.",
        "Can I change the billing email on our account?",
        "Is annual billing cheaper than monthly?",
        "I need a tax invoice with my company VAT number.",
    ],
    "GENERAL": [
        "What is your SLA for priority-1 incidents?",
        "Do you have a reference customer in the retail sector?",
        "Can I book a demo for my team next week?",
        "Is there a mobile app?",
        "Which integrations do you support out of the box?",
        "Do you have SOC 2 certification?",
        "Can we run a 30-day trial before committing?",
    ],
}


def generate_ticket(category: str | None = None) -> dict:
    cat = category or random.choice(list(CATEGORIES))
    return {
        "ticket_id": f"T-{random.randint(10000, 99999)}",
        "customer": fake.name(),
        "email": fake.company_email(),
        "received_at": fake.date_time_this_month().isoformat(),
        "body": random.choice(CATEGORIES[cat]),
        "true_category": cat,
    }


def batch(n: int = 10) -> list[dict]:
    return [generate_ticket() for _ in range(n)]


if __name__ == "__main__":
    for t in batch(5):
        print(t)

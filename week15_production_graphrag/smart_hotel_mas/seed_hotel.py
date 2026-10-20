#!/usr/bin/env python3
"""Pre-workshop seed = Checkpoint 1. Populates 200 rooms / 400 devices / 2000 readings.

Thin convenience wrapper so attendees can run a single command from the workshop
root before the session starts, instead of remembering the checkpoint path.
Delegates entirely to ``checkpoints/checkpoint1_seed.py``. (smart_hotel_mas.pdf §CP1)

Run:  python week15/smart_hotel_mas/seed_hotel.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "checkpoints"))
from checkpoint1_seed import seed_database, check_neo4j

if __name__ == "__main__":
    if check_neo4j():
        seed_database()

#!/usr/bin/env python3
"""Make data/chiller_plant.csv — SYNTHETIC 15-minute chiller-plant data for the Alto Ops Claw.

Not from a real building. Deterministic (seed 26) so every learner, lab and eval sees the same numbers:
7 days × 96 rows, columns ts,kw,rt. Plant efficiency sits around 0.62–0.72 kW/RT with a daily load curve,
then degrades to ~0.9 kW/RT over the last 6 hours (a fouled condenser, say) so the alarm path has something
to find.

    .venv/bin/python week26/common/data/make_chiller_csv.py
"""
import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parent / "chiller_plant.csv"
START = datetime(2026, 9, 21, 0, 0)          # Monday
ROWS = 7 * 96


def rows():
    rnd = random.Random(26)
    for i in range(ROWS):
        ts = START + timedelta(minutes=15 * i)
        hour = ts.hour + ts.minute / 60
        load = 0.55 + 0.4 * max(0.0, math.sin(math.pi * (hour - 6) / 14))  # 06:00–20:00 hump
        rt = round(900 * load + rnd.uniform(-25, 25), 1)
        eff = 0.64 + 0.06 * (1 - load) + rnd.uniform(-0.03, 0.03)
        if i >= ROWS - 24:                    # last 6 hours: degraded
            eff = 0.90 + rnd.uniform(-0.02, 0.02)
        yield {"ts": ts.strftime("%Y-%m-%dT%H:%M"), "kw": round(rt * eff, 1), "rt": rt}


if __name__ == "__main__":
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ts", "kw", "rt"])
        w.writeheader()
        w.writerows(rows())
    print(f"✓ wrote {OUT} ({ROWS} rows, synthetic)")

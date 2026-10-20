#!/usr/bin/env python3
"""Solution · Exercise 04 — one good answer (yours may differ and still be right)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevkit import top2  # noqa: E402

import ex04_add_a_route as ex  # noqa: E402


def accepted(a: dict) -> bool:
    top, second = top2(a["probabilities"])
    return a["confidence"] >= 0.75 and top >= 0.80 and top - second >= 0.20


ex.accepted = accepted
ex.NEW_OPTION_DESCRIPTION = (
    "Gateways, IoT sensors or meters that are offline, not reporting, or sending stale or missing "
    "data. Not for equipment that is reporting normally but performing badly (that is hvac).")
ex.MY_TEST_MESSAGES = [
    "The Modbus gateway on the chiller plant keeps dropping off the network.",
    "Energy meter M-07 hasn't sent a reading since Tuesday.",
    "เกตเวย์ที่โรงแรมสาขาภูเก็ตออฟไลน์ ข้อมูลไม่เข้าระบบตั้งแต่เมื่อคืน",
]

if __name__ == "__main__":
    ex.main()

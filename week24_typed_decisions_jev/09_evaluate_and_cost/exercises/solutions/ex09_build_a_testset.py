#!/usr/bin/env python3
"""Solution · Exercise 09 — one good test set and a macro-F1 implementation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ex09_build_a_testset as ex  # noqa: E402

ex.MY_ROWS = [
    {"id": "my-1", "language": "en", "message": "The chiller plant kW/ton jumped last night — why?", "expected": "hvac"},
    {"id": "my-2", "language": "th", "message": "แอร์ห้องประชุมชั้น 5 ไม่เย็น ช่วยหาสาเหตุ", "expected": "hvac"},
    {"id": "my-3", "language": "th", "message": "ช่วยทำรายงานคาร์บอนฟุตพริ้นท์ประจำปีของอาคาร", "expected": "sustainability"},
    {"id": "my-4", "language": "en", "message": "Fix the failing unit test in our BACnet polling service.", "expected": "coding"},
    {"id": "my-5", "language": "en", "message": "Open a work order to replace the AHU-2 filters on Tuesday.", "expected": "facility_ops"},
    {"id": "my-6", "language": "mixed", "message": "ช่วย verify savings ของ project LED retrofit ตาม IPMVP", "expected": "energy_mv"},
    {"id": "my-7", "language": "en", "message": "hmm, what about the other one?", "expected": "unknown"},
]


def macro_f1(pairs):
    classes = {g for g, _ in pairs} | {p for _, p in pairs}
    if not classes:
        return 0.0
    total = 0.0
    for c in classes:
        tp = sum(g == c and p == c for g, p in pairs)
        fp = sum(g != c and p == c for g, p in pairs)
        fn = sum(g == c and p != c for g, p in pairs)
        total += 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return total / len(classes)


ex.macro_f1 = macro_f1

if __name__ == "__main__":
    ex.main()

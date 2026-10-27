#!/usr/bin/env python3
"""readkit — Module 06's reading rule for a results table built from three paired streams.

Not part of the frozen kit: a module-local helper the Module 06 labs share.

A row in a results table is three stream means (stream s of every arm uses the same shuffled task order, so
stream s of arm A and stream s of arm B are a pair). Two questions decide whether a difference is worth acting
on, the same two the course's pre-registered criteria ask (EXAMPLE design, Module 09):

    agree       all three paired stream differences point the same way
    beyond SD   the difference of the means is larger than the larger of the two stream SDs

    verdict     "beyond noise"    both hold
                "direction only"  the streams agree, but the gap is inside one SD: more streams or tasks needed
                "within noise"    the streams disagree: the table does not show a difference

    from readkit import reading
    r = reading(base["means"], arm["means"])     # {'delta', 'diffs', 'agree', 'beyond_sd', 'verdict'}
"""
from __future__ import annotations

import statistics


def _sd(xs: list[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def reading(a: list[float], b: list[float]) -> dict:
    """Read arm b against arm a, stream by stream."""
    diffs = [round(y - x, 1) for x, y in zip(a, b)]
    delta = round(statistics.mean(b) - statistics.mean(a), 1)
    agree = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
    sd = round(max(_sd(a), _sd(b)), 1)
    beyond = abs(delta) > sd
    verdict = "beyond noise" if agree and beyond else "direction only" if agree else "within noise"
    return {"delta": delta, "diffs": diffs, "agree": agree, "sd": sd, "beyond_sd": beyond, "verdict": verdict}


def fmt(s: dict) -> str:
    """mean ± SD as the paper reports it."""
    return f"{s['mean']:.1f} ± {s['sd']:.1f}"


def streams_str(s: dict) -> str:
    return " / ".join(f"{m:.1f}" for m in s["means"])

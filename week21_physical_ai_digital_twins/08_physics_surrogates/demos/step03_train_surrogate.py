#!/usr/bin/env python3
"""PART 3 · Train a surrogate live — then break it  [ADVANCED]

The whole PhysicsNeMo workflow at desk scale: 200 parametric runs of an RC hotel
model (the solver), a polynomial-feature linear model fitted by gradient descent
(the surrogate), a speed/accuracy benchmark — and then the DANGER demo: query it
outside the training envelope and watch it be wrong, confidently.

Run:  python demos/step03_train_surrogate.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim  # noqa: E402
import view  # noqa: E402

EPOCHS, LR, REPORT = 1200, 0.55, 200


def train(samples):
    """Full-batch gradient descent on MSE — the entire 'ML lifecycle', visible."""
    feats = [sim.features(*x) for x, _ in samples]           # 15 poly features/sample
    ys = [y / sim.Y_SCALE for _, y in samples]               # scale kW → ~[0,1]
    w, n = [0.0] * 15, len(samples)
    for epoch in range(1, EPOCHS + 1):
        grad, loss = [0.0] * 15, 0.0
        for f, y in zip(feats, ys):
            err = sum(wi * fi for wi, fi in zip(w, f)) - y
            loss += err * err
            for k in range(15):
                grad[k] += err * f[k]
        w = [wi - LR * g / n for wi, g in zip(w, grad)]
        if epoch == 1 or epoch % REPORT == 0:
            print(f"    epoch {epoch:>5}   mse {loss / n * sim.Y_SCALE**2:>9.3f} kW²")
    return w


def main() -> None:
    view.banner("PART 3", "Train a surrogate live — then break it", "ADVANCED")
    print("▣ MODE: REAL — pure-stdlib training, nothing to install. That's the point:")
    print("  this surrogate is 15 numbers. PhysicsNeMo scales the same recipe to fields.\n")

    print("① GENERATE — 200 parametric runs of the RC hotel wing (the 'solver'):")
    print("   inputs: outdoor °C · occupancy · setpoint °C · solar kW → cooling ELECTRIC kW")
    t0 = time.perf_counter()
    samples = sim.generate_samples(200)
    print(f"   200 runs in {time.perf_counter() - t0:.2f} s "
          f"(each marches a thermal-mass node to steady state)\n")

    print("② TRAIN — gradient descent on 15 polynomial features (watch the loss fall):")
    w = train(samples)
    print()

    print("③ VALIDATE — accuracy on 60 held-out runs, and the speed that pays for it all:")
    hold = sim.generate_samples(60, seed=99)
    mae = sum(abs(sim.predict(w, *x) - y) for x, y in hold) / len(hold)
    t0 = time.perf_counter()
    for _ in range(2000):
        sim.predict(w, 32.0, 0.6, 24.0, 150.0)
    us = (time.perf_counter() - t0) / 2000 * 1e6
    t0 = time.perf_counter()
    for _ in range(20):
        sim.full_model(32.0, 0.6, 24.0, 150.0)
    ms = (time.perf_counter() - t0) / 20 * 1e3
    print(f"   ◆ holdout MAE      {mae:5.2f} kW   (typical load ~35 kW → ~{mae/35*100:.0f}% error)")
    print(f"   ◆ full RC physics  {ms*1000:7.0f} µs / query")
    print(f"   ◆ surrogate        {us:7.1f} µs / query   → ~{ms*1000/us:,.0f}× faster\n")

    print("④ THE DANGER DEMO — query OUTSIDE the training envelope")
    print(f"   trained on: outdoor {sim.ENVELOPE['t_out'][0]:.0f}–{sim.ENVELOPE['t_out'][1]:.0f} °C, "
          f"occupancy {sim.ENVELOPE['occ'][0]:.0%}–{sim.ENVELOPE['occ'][1]:.0%}")
    print("   query: a record 40 °C day, 120 % occupancy (conference + walk-ins):\n")
    x = (40.0, 1.2, 24.0, 340.0)
    truth_kw, t_air, clamped = sim.full_detail(*x)
    guess = sim.predict(w, *x)
    print(f"     {'':<22}{'cooling kW':>12}   what it knows")
    print("     " + "─" * 74)
    print(f"     {'full physics':<22}{truth_kw:>12.1f}   chiller SATURATES at capacity; "
          f"zone floats to {t_air:.1f} °C{' ⚠' if clamped else ''}")
    print(f"     {'surrogate':<22}{guess:>12.1f}   a smooth polynomial — no concept of a "
          f"capacity limit")
    print(f"     → off by {abs(guess-truth_kw)/truth_kw*100:.0f}%, delivered in {us:.0f} µs, "
          f"with zero error bars. Wrong, CONFIDENTLY.")
    print("     Worse: the real story of that day is the comfort failure (zone above")
    print("     setpoint) — an output the surrogate wasn't even trained to mention.\n")

    print("⑤ THE GUARD — the fix is procedural, not architectural:")
    kw, fell = sim.predict_guarded(w, *x)
    print("   know the envelope · check every query · fall back to physics outside it")
    print(f"   predict_guarded(40 °C, 120 %) → {kw:.1f} kW (fell back to physics: {fell})\n")

    print('Takeaway: "A surrogate is a cache of physics — and caches go stale."')
    print("Retrain when the building changes; guard every query against the envelope.")
    print("Next: feed the surrogate a weather FORECAST and decide tonight's pre-cool.")


if __name__ == "__main__":
    main()

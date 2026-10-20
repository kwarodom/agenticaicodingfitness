#!/usr/bin/env python3
"""Exercise 01 · Fit formula and sovereignty — reference solution.

Fill in the four TODOs, save, then run:
    week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/01_local_sovereign_edge/exercises/ex01_fit_formula.py

The checker is free and offline. Stuck? Compare with exercises/solutions/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from embedkit import banner, check  # noqa: E402

BYTES_PER_PARAM = {"float32": 4, "bfloat16": 2, "int8": 1, "q4": 0.5}


# ── TODO 1 ── Tina's rule: about 25 % of the RAM is used by the OS and other apps. Return what is left.
def usable_gb(ram_gb: float) -> float:
    return ram_gb * 0.75


# ── TODO 2 ── her rule of thumb: max parameters (in billions) ≈ usable GB ÷ 0.6
def max_params_b(ram_gb: float) -> float:
    return usable_gb(ram_gb) / 0.6


# ── TODO 3 ── weight memory in MB for a model of params_m million parameters stored as dtype
def weights_mb(params_m: float, dtype: str) -> float:
    return params_m * BYTES_PER_PARAM[dtype]


# ── TODO 4 ── label each deployment "sovereign", "residency only" or "not sovereign"
SOVEREIGNTY = {
    "cloud_api": "not sovereign",
    "eu_region_vendor": "residency only",
    "hotel_server_no_egress": "sovereign",
    "phone_offline": "sovereign",
}


# ─────────────────────────── checker — no need to edit below ────────────────
def main() -> None:
    banner("Exercise 01 · fit formula + sovereignty", "offline checker · no model needed")
    good = True
    try:
        u = usable_gb(32)
    except Exception as e:  # noqa: BLE001
        u = f"error: {e}"
    good &= check(u == 24, "usable_gb(32) = 24", f"TODO 1: usable_gb(32) returned {u!r} — subtract 25 % of the RAM")
    try:
        p = [round(max_params_b(x), 3) for x in (32, 64, 8)]
    except Exception as e:  # noqa: BLE001
        p = f"error: {e}"
    good &= check(p == [40.0, 80.0, 10.0], "max_params_b: 32 GB → 40B · 64 GB → 80B · 8 GB → 10B",
                  f"TODO 2: max_params_b(32, 64, 8) = {p} — usable GB ÷ 0.6")
    try:
        w = [round(weights_mb(270, "bfloat16")), round(weights_mb(740, "float32")), round(weights_mb(270, "q4"))]
    except Exception as e:  # noqa: BLE001
        w = f"error: {e}"
    good &= check(w == [540, 2960, 135], "weights_mb: text bf16 540 MB · full fp32 2960 MB · text q4 135 MB",
                  f"TODO 3: weights_mb gave {w} — params (millions) × bytes per parameter")
    good &= check(SOVEREIGNTY == {"cloud_api": "not sovereign", "eu_region_vendor": "residency only",
                                  "hotel_server_no_egress": "sovereign", "phone_offline": "sovereign"},
                  "sovereignty: cloud API ✕ · vendor region = residency only · own server, no egress ✓ · offline phone ✓",
                  f"TODO 4: {SOVEREIGNTY} — who controls the hardware, and can data or telemetry leave?")
    if not good:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)
    print("\n═ Done. Module 06 reuses weights_mb() to size an edge deployment.")


if __name__ == "__main__":
    main()

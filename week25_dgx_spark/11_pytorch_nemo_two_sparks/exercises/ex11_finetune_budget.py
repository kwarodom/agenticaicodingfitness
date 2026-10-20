#!/usr/bin/env python3
"""Exercise 11 · The fine-tune budget: LoRA size, training state, and memory per Spark.

Fill in the three TODOs, save, then run:
    .venv/bin/python week25/11_pytorch_nemo_two_sparks/exercises/ex11_finetune_budget.py

The checker is free and offline. It compares your functions with counts you can verify by hand
(and against the 'Trainable parameters' line the 8B LoRA script prints), then plans three jobs on
one or two Sparks. Stuck? Compare with exercises/solutions/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, check  # noqa: E402

SPARK_GB = 128


# ── TODO 1 ── LoRA adds two small matrices to every targeted projection: A (r × d_in) and B (d_out × r).
#   shapes is the list of (d_in, d_out) for the projections in ONE layer; return the total for all layers.
def lora_params(layers: int, shapes: list[tuple[int, int]], r: int) -> int:
    return None


# ── TODO 2 ── training state in GB (1 GB = 1e9 bytes), as the playbook's scripts run it:
#   frozen weights take frozen_bits / 8 bytes each (16 for bf16, ~4.13 for NF4);
#   every TRAINABLE parameter takes 8 bytes: bf16 weight 2 + bf16 grad 2 + AdamW m and v, 2 each.
#   Params are given in billions. (Full fine-tuning: frozen_b = 0, trainable_b = all.)
def state_gb(frozen_b: float, trainable_b: float, frozen_bits: float = 16) -> float:
    return None


# ── TODO 3 ── memory per Spark when FSDP shards the state over n Sparks.
#   The state is split n ways; activations are NOT (each Spark runs its own micro-batch); add headroom.
def per_spark_gb(state: float, activations: float, n: int, headroom: float = 10) -> float:
    return None


# ─────────────────────────── checker — no need to edit below ────────────────
def llama(h: int, kv: int, ff: int) -> list[tuple[int, int]]:
    return [(h, h), (h, kv), (h, kv), (h, h), (h, ff), (h, ff), (ff, h)]      # q k v o gate up down


def close(a, b, tol=0.01) -> bool:
    return isinstance(a, (int, float)) and abs(a - b) <= tol * max(1.0, abs(b))


def main() -> None:
    banner("Exercise 11 · the fine-tune budget", "offline checker · free · no Spark needed", status=False)
    ok = True
    l8 = lora_params(32, llama(4096, 1024, 14336), 8)
    l70 = lora_params(80, llama(8192, 1024, 28672), 8)
    ok &= check(l8 == 20_971_520 and l70 == 103_546_880,
                "lora_params: Llama 3.1 8B r=8 = 20,971,520 · Llama 3.1 70B r=8 = 103,546,880",
                f"TODO 1: lora_params(32, llama 8B, 8) should be 20971520 (got {l8!r})")
    ok &= check(close(state_gb(0, 3.21), 25.68) and close(state_gb(8.03, 0.021), 16.228)
                and close(state_gb(68.45, 0.104, 4.127), 36.14),
                "state_gb: 3B full = 25.7 GB · 8B LoRA = 16.2 GB · 70B QLoRA (linear layers in NF4) = 36.1 GB",
                f"TODO 2: state_gb(0, 3.21) should be ≈ 25.68 (got {state_gb(0, 3.21)!r})")
    ok &= check(close(per_spark_gb(141.9, 2.5, 1), 154.4) and close(per_spark_gb(141.9, 2.5, 2), 83.45),
                "per_spark_gb: 70B LoRA bf16 → 154.4 GB on one Spark · 83.5 GB each on two",
                f"TODO 3: per_spark_gb(141.9, 2.5, 2) should be ≈ 83.45 (got {per_spark_gb(141.9, 2.5, 2)!r})")
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)

    print("\n▣ your budget, applied (activations from lab 01's 'typ' column)")
    jobs = [("Llama 3.1 8B LoRA", state_gb(8.03, l8 / 1e9), 11.9),
            ("Llama 3.1 70B LoRA, bf16", state_gb(70.55, l70 / 1e9), 2.5),
            ("Llama 3.1 8B full SFT", state_gb(0, 8.03), 11.9)]
    for name, st, act in jobs:
        one, two = per_spark_gb(st, act, 1), per_spark_gb(st, act, 2)
        plan = "1 Spark" if one <= SPARK_GB else ("2 Sparks (FSDP)" if two <= SPARK_GB else "too big")
        print(f"│ {name:26s} state {st:6.1f} GB · 1 Spark {one:6.1f} GB · 2 Sparks {two:6.1f} GB each → {plan}")
    print("\n═ Now try frozen_bits = 4.127 for the 70B job: which plan does QLoRA give, and what does it cost in quality?")


if __name__ == "__main__":
    main()

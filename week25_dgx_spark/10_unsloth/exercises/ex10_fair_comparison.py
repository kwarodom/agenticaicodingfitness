#!/usr/bin/env python3
"""Exercise 10 · Make an unfair framework comparison fair.

A teammate ran Unsloth with the settings below, copied from a notebook, and reported "Unsloth is
2× faster than LLaMA Factory and has lower loss". But four settings differ from Module 09's
LLaMA Factory run, so the claim says little about the frameworks. Fix the four TODOs so that
every setting that must match does match, save, then run:
    .venv/bin/python week25/10_unsloth/exercises/ex10_fair_comparison.py

The checker is free and offline: it reads the REAL Module 09 YAML
(week25/09_llama_factory/configs/hotel_lora_sft.yaml) and compares it with your dict.
Stuck? Compare with exercises/solutions/.
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, check  # noqa: E402

LF_YAML = Path(__file__).resolve().parents[2] / "09_llama_factory" / "configs" / "hotel_lora_sft.yaml"

UNSLOTH_CFG = {
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "max_seq_length": 1024,
    "load_in_4bit": False,
    # ── TODO 1 ── adapter size: a bigger or smaller adapter changes both speed and quality.
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    # ── TODO 2 ── effective batch = per-device batch × accumulation. It decides the number of optimizer steps.
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 3.0,
    # ── TODO 3 ── the learning rate (and the schedule) must be the same, or the losses are not comparable.
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    # ── TODO 4 ── what the loss is computed on. LLaMA Factory masks the prompt by default.
    "completion_only_loss": False,
}


# ─────────────────────────── checker — no need to edit below ────────────────
def main() -> None:
    banner("Exercise 10 · a fair comparison", "offline checker · reads Module 09's real YAML · no Spark needed",
           status=False)
    if not LF_YAML.is_file():
        print("✕ run week25/09_llama_factory/labs/lab03_config_and_launch.py first (it writes the YAML compared here).")
        sys.exit(1)
    lf, us = yaml.safe_load(LF_YAML.read_text(encoding="utf-8")), UNSLOTH_CFG
    ok = True
    lf_alpha = lf.get("lora_alpha", 2 * lf["lora_rank"])
    ok &= check((us["r"], us["lora_alpha"]) == (lf["lora_rank"], lf_alpha),
                f"TODO 1: LoRA r / alpha = {us['r']} / {us['lora_alpha']}, same as Module 09",
                f"TODO 1: r / alpha = {us['r']} / {us['lora_alpha']}, Module 09 uses {lf['lora_rank']} / {lf_alpha} "
                f"({us['r'] / lf['lora_rank']:.1f}× the adapter)")
    eff_us = us["per_device_train_batch_size"] * us["gradient_accumulation_steps"]
    eff_lf = lf["per_device_train_batch_size"] * lf["gradient_accumulation_steps"]
    ok &= check(eff_us == eff_lf, f"TODO 2: effective batch {eff_us}, same as Module 09 → the same "
                f"{-(-504 // eff_lf) * int(lf['num_train_epochs'])} optimizer steps",
                f"TODO 2: effective batch {eff_us} vs Module 09's {eff_lf}: {-(-504 // eff_us) * 3} steps instead of "
                f"{-(-504 // eff_lf) * 3}, with bigger batches that use the GPU differently — time per run and loss curves stop being comparable")
    ok &= check((us["learning_rate"], us["lr_scheduler_type"], us["warmup_ratio"])
                == (lf["learning_rate"], lf["lr_scheduler_type"], lf["warmup_ratio"]),
                f"TODO 3: learning rate {us['learning_rate']} · {us['lr_scheduler_type']} · warmup {us['warmup_ratio']}",
                f"TODO 3: learning rate {us['learning_rate']} vs Module 09's {lf['learning_rate']}")
    ok &= check(us["completion_only_loss"] is True,
                "TODO 4: loss on the answer only, like LLaMA Factory",
                "TODO 4: completion_only_loss=False also trains on the (long, repeated) system prompt: the loss "
                "drops fast on text the model never has to write, so it looks 'lower' without being better")
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)
    print("\n▣ now a speed or accuracy difference says something about the framework, not the settings.")
    print("│ still compare on the same Spark, one job at a time, two runs each, with one scorer (lab 03).")


if __name__ == "__main__":
    main()

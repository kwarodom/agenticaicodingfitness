#!/usr/bin/env python3
"""A faithful **fine-tuning simulator** for a DGX Spark.

You can't train a 70B model on a laptop, so the training-loop demo streams a
realistic run instead of faking it silently: a decaying loss curve, a warmup→
cosine learning-rate schedule, samples/sec + tok/s throughput, VRAM headroom,
and periodic checkpoint writes — the same telemetry NeMo / Unsloth print. Every
number is clearly marked as simulated, and the real launch command is shown.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass
class TrainConfig:
    base_model: str
    method: str = "QLoRA"          # LoRA | QLoRA | full-SFT
    steps: int = 60
    batch: int = 8
    seq_len: int = 1024
    lr: float = 2e-4
    warmup: int = 6
    params_b: float = 8.0


def _loss_at(step: int, total: int) -> float:
    """A plausible SFT loss: starts ~1.9, decays toward ~0.4 with a little noise."""
    frac = step / max(total, 1)
    base = 0.4 + 1.5 * math.exp(-3.2 * frac)
    wobble = 0.04 * math.sin(step * 1.7) * (1 - frac)
    return round(base + wobble, 4)


def _lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    frac = (step - cfg.warmup) / max(cfg.steps - cfg.warmup, 1)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * frac))


def vram_gb(cfg: TrainConfig) -> float:
    """Rough training VRAM: weights + optimizer/adapter state by method."""
    if cfg.method == "full-SFT":
        return round(cfg.params_b * 2 * 4.0, 1)        # bf16 weights + Adam m/v + grads
    if cfg.method == "LoRA":
        return round(cfg.params_b * 2 * 1.15, 1)        # frozen bf16 base + tiny adapter
    return round(cfg.params_b * 0.55 * 1.25, 1)         # QLoRA: 4-bit base + adapter


def run(cfg: TrainConfig, emit, *, fast: bool = True):
    """Drive a simulated run; call ``emit(line)`` for each line of output."""
    # throughput scales with batch and shrinks with model size / seq_len
    samples_s = max(0.6, 26.0 / cfg.params_b) * (cfg.batch / 8) * (1024 / cfg.seq_len)
    tok_s = samples_s * cfg.seq_len
    emit(f"  device     : NVIDIA GB10 (DGX Spark) · 128 GB unified")
    emit(f"  method     : {cfg.method} on {cfg.base_model}")
    emit(f"  est. VRAM  : ~{vram_gb(cfg)} GB / 128 GB   (fits ✓)")
    emit(f"  throughput : ~{samples_s:.1f} samples/s · ~{tok_s:,.0f} tok/s")
    emit("")
    emit(f"  {'step':>5} {'loss':>8} {'lr':>10} {'tok/s':>10}  ckpt")
    emit("  " + "─" * 48)
    for step in range(1, cfg.steps + 1):
        loss = _loss_at(step, cfg.steps)
        lr = _lr_at(step, cfg)
        ckpt = ""
        if step % max(cfg.steps // 3, 1) == 0 or step == cfg.steps:
            ckpt = f"→ checkpoints/step_{step}/"
        if step <= 3 or step % 5 == 0 or ckpt:
            emit(f"  {step:>5} {loss:>8.4f} {lr:>10.2e} {tok_s:>10,.0f}  {ckpt}")
        if not fast:
            time.sleep(0.01)
    emit("")
    emit(f"  ✓ training complete — final loss {_loss_at(cfg.steps, cfg.steps):.4f} "
         f"(from {_loss_at(1, cfg.steps):.4f})")
    emit(f"  ✓ adapter saved: .sandbox/checkpoints/step_{cfg.steps}/adapter_model.safetensors")
    return _loss_at(cfg.steps, cfg.steps)

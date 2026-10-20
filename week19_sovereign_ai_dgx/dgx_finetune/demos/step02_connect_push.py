#!/usr/bin/env python3
"""STEP 2 · 💻→🖥️ Connect to the DGX & push the dataset (REAL SSH)  [BEGINNER]

Tests SSH to your DGX (and confirms the GPU is visible), then scp's the dataset
and training script into the remote working dir. Configure SSH in the web app's
🔌 DGX SSH panel first (host / user / key).

Run:  python demos/step02_connect_push.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import ftssh  # noqa: E402


def main() -> None:
    print("━" * 64)
    print("  STEP 2 · 💻→🖥️ Connect to the DGX & push the dataset (SSH)")
    print("━" * 64, "\n")

    if not ftssh.require():
        return

    sb = config.SANDBOX
    needed = ["train.jsonl", "val.jsonl", "train_hvac_unsloth.py"]
    if not all((sb / f).exists() for f in needed):
        print("⚠  Dataset/script missing — run STEP 1 first.")
        return

    print(f"Target: {ftssh.target()}   workdir: {ftssh.WORKDIR}\n")
    print("1) Testing SSH + GPU visibility on the DGX …")
    if ftssh.test() != 0:
        print("\n✗ SSH/GPU test failed. Check host/user/key, and that `ssh <user>@<host>` works.")
        return

    print("\n2) Pushing files to the DGX …")
    rc = 0
    for f in needed:
        rc |= ftssh.push(str(sb / f), f)
    if rc != 0:
        print("\n✗ scp failed — check the workdir is writable.")
        return

    print("\n3) Verifying they landed:")
    ftssh.run(f"ls -la {ftssh.WORKDIR}")

    print("\nTakeaway: your data + training script now live on the DGX. STEP 3 pulls")
    print("the NVIDIA PyTorch container so the real run is fast.")


if __name__ == "__main__":
    main()

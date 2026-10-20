#!/usr/bin/env python3
"""STEP 4 · 🖥️ Run the REAL Unsloth fine-tune on the DGX — REAL SSH  [ADVANCED]

Executes the actual training on the DGX inside NVIDIA's PyTorch container: installs
Unsloth, loads YOUR chosen model in 4-bit, trains a LoRA on the HVAC dataset, and
exports a Q4_K_M GGUF. The real loss curve streams back here. This can take a while
(model download + steps) — that's a real fine-tune.

Set the model + HF token (and optional max-steps) in the web app's 🔌 DGX SSH panel.

Run:  python demos/step04_train.py
"""
from __future__ import annotations

import os
import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import ftssh  # noqa: E402

CONTAINER = "nvcr.io/nvidia/pytorch:25.11-py3"
# HF model id / local path: letters, digits and . _ - / : only — no shell metachars.
_MODEL_RE = re.compile(r"^[A-Za-z0-9._/:-]+$")


def main() -> None:
    print("━" * 64)
    print("  STEP 4 · 🖥️ REAL Unsloth fine-tune on the DGX")
    print("━" * 64, "\n")
    if not ftssh.require():
        return

    hf_model = os.environ.get("FT_HF_MODEL", "unsloth/Llama-3.2-3B-Instruct").strip()
    hf_token = os.environ.get("FT_HF_TOKEN", "").strip()
    steps = os.environ.get("FT_MAX_STEPS", "60").strip()

    # ── validate UI-supplied values before they touch a remote shell ──────────
    if not _MODEL_RE.match(hf_model):
        print(f"✗ invalid model id {hf_model!r} — allowed: letters, digits and . _ - / :")
        print("  (fix it in the 🖥️ DGX SSH panel; this guard prevents shell injection.)")
        return
    if not steps.isdigit() or not (1 <= int(steps) <= 100000):
        print(f"✗ invalid step count {steps!r} — must be a positive integer.")
        return

    print(f"Model:    {hf_model}")
    print(f"Steps:    {steps}    HF token: {'set' if hf_token else '(none — ok for ungated models)'}")
    print(f"On:       {ftssh.target()}  in  {ftssh.WORKDIR}\n")

    # Secrets never go in argv / the command echo: write the token to a 077 env-file
    # on the DGX via STDIN, then hand it to docker with --env-file.
    envfile_opt = ""
    if hf_token:
        if ftssh.write_remote(f"{ftssh.WORKDIR}/.hf.env", f"HF_TOKEN={hf_token}\n") == 0:
            envfile_opt = "--env-file .hf.env "
            print("• HF token written to a 0600 env-file on the DGX (not in the command).")
        else:
            print("⚠ could not write the token env-file; continuing without it.")

    inner = 'pip install -q unsloth "datasets>=3" trl peft && python train_hvac_unsloth.py'
    # WORKDIR is allowlist-validated in ftssh (safe, and ~ expands). Every other
    # interpolated value is shlex-quoted so it cannot break out of the shell.
    remote = (
        f"cd {ftssh.WORKDIR} && test -f train_hvac_unsloth.py || "
        f"{{ echo 'missing script — run STEP 2'; exit 1; }}; "
        f"docker run --rm --gpus all --ipc host -v \"$PWD:/work\" -w /work "
        f"{envfile_opt}-e HF_MODEL={shlex.quote(hf_model)} -e MAX_STEPS={shlex.quote(steps)} "
        f"{CONTAINER} bash -lc {shlex.quote(inner)}"
    )

    print("Launching (output is live from the DGX — pip install first, then the loss curve):\n")
    rc = ftssh.run(remote)
    if hf_token:
        ftssh.run(f"rm -f {ftssh.WORKDIR}/.hf.env", echo=False)  # clean up the token file
    print()
    if rc == 0:
        print("✓ training complete — LoRA + GGUF written to "
              f"{ftssh.WORKDIR}/hvac-model on the DGX.")
        print("\nHow to read it: loss should fall fast then flatten. Val/eval is the real")
        print("test — STEP 6. Next: STEP 5 serves the tuned model via Ollama.")
    else:
        print("✗ training did not finish cleanly. Common causes:")
        print("  • out of VRAM → free the GPU (stop other models) or pick a smaller model")
        print("  • gated model without a valid HF token → set it in the SSH panel")
        print("  • docker missing → install it, or adapt the demo to a venv")


if __name__ == "__main__":
    main()

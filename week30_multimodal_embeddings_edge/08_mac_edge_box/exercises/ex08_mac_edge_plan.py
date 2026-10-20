#!/usr/bin/env python3
"""Exercise 08 · Run a Mac as the edge box: pick runtimes from measurements, bind safely, count joules, fix launchd.

Fill in the four TODOs, save, then run:
    week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/08_mac_edge_box/exercises/ex08_mac_edge_plan.py

The checker is free and offline. Stuck? Compare with exercises/solutions/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from embedkit import banner, check  # noqa: E402

import ipaddress
import plistlib

# Lab 08-1 / 08-2 measured on this Mac (M4 Max): runtime → (weights MB, 1 query ms, text cos, voice support)
MEASURED = {"pytorch-mps": (1489, 22.5, 0.9999, True), "mlx-bf16": (1489, 7.3, 0.9998, True),
            "mlx-4bit": (1098, 6.1, 0.9598, True), "onnx-q4": (175, 15.5, 0.9703, False),
            "onnx-fp16": (543, 17.2, 1.0000, False)}

# ── TODO 1 ── pick a runtime key from MEASURED for each job
PICK = {
    "voicemail_index": None,   # index guest voicemails: needs audio, and the closest match to the original
    "tiny_text_box": None,     # notes-only search on a box with very little disk
    "fastest_query": None,     # lowest single-query latency
    "reference": None,         # the original model, to check the others against
}


# ── TODO 2 ── may the service listen on this host? Loopback always; any other address only with a token of
#   at least 24 characters (embed_service.py refuses an empty one).
def safe_bind(host: str, token: str) -> bool:
    return True


# ── TODO 3 ── energy per 1,000 notes from a power reading (W) and a throughput (notes per second)
def joules_per_1000(watts: float, notes_per_s: float) -> float:
    return 0.0


# ── TODO 4 ── fix this launchd file: start at login, restart if it dies, loopback only, never reach the Hub
PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.alto.edge-memory</string>
  <key>ProgramArguments</key>
  <array><string>python</string><string>embed_service.py</string><string>--host</string><string>0.0.0.0</string></array>
  <key>RunAtLoad</key><false/>
</dict>
</plist>
"""


# ─────────────────────────── checker — no need to edit below ────────────────
def main() -> None:
    banner("Exercise 08 · run a Mac as the edge box", "offline checker · uses Lab 08-1's measured table")
    good = True
    good &= check(PICK == {"voicemail_index": "mlx-bf16", "tiny_text_box": "onnx-q4", "fastest_query": "mlx-4bit",
                           "reference": "pytorch-mps"},
                  "runtimes: voicemails → MLX bf16 · smallest text-only → ONNX q4 · fastest query → MLX 4-bit · "
                  "reference → PyTorch",
                  f"TODO 1: PICK = {PICK} — use the MEASURED table (Lab 08-1/08-2), not a guess")
    try:
        ok_ = [safe_bind("127.0.0.1", ""), safe_bind("0.0.0.0", ""), safe_bind("0.0.0.0", "x" * 32),
               safe_bind("192.168.1.20", "short"), safe_bind("::1", "")]
    except Exception as e:  # noqa: BLE001
        ok_ = f"error: {e}"
    good &= check(ok_ == [True, False, True, False, True],
                  "safe_bind: loopback always · a network address only with a token of ≥ 24 characters",
                  f"TODO 2: safe_bind gave {ok_} (want [True, False, True, False, True])")
    try:
        j = [round(joules_per_1000(18.0, 400), 1), round(joules_per_1000(9.5, 120), 1)]
    except Exception as e:  # noqa: BLE001
        j = f"error: {e}"
    good &= check(j == [45.0, 79.2], "joules_per_1000: 18 W at 400 notes/s = 45.0 J · 9.5 W at 120/s = 79.2 J",
                  f"TODO 3: joules_per_1000 gave {j} — watts are joules per second")
    try:
        plist = plistlib.loads(PLIST.encode())
        pl_ok = (plist.get("KeepAlive") is True and plist.get("RunAtLoad") is True
                 and "--host" in plist.get("ProgramArguments", [])
                 and plist["ProgramArguments"][plist["ProgramArguments"].index("--host") + 1] == "127.0.0.1"
                 and plist.get("EnvironmentVariables", {}).get("HF_HUB_OFFLINE") == "1")
    except Exception as e:  # noqa: BLE001
        pl_ok = False
        print(f"  (plist did not parse: {e})")
    good &= check(pl_ok, "launchd plist: starts at login, restarts if it dies, loopback only, offline",
                  "TODO 4: fix PLIST — RunAtLoad, KeepAlive, --host 127.0.0.1 and HF_HUB_OFFLINE=1 must all be set")
    if not good:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)
    print("\n═ Done. Your Mac mini is an always-on, sovereign embedding appliance — on paper. Lab 08-3 runs it for real.")


if __name__ == "__main__":
    main()

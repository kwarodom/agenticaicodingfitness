#!/usr/bin/env python3
"""Exercise 12 · Fix a broken FLUX.1 Dreambooth data.toml before it wastes a four-hour run.

A teammate added the hotel concept to the playbook's data.toml. Four mistakes: some can stop sd-scripts
before it starts, others let it train for hours and produce a LoRA you cannot prompt.
Fix the four TODOs in DATA_TOML, save, then run:
    .venv/bin/python week25/12_vlm_flux_finetune/exercises/ex12_fix_data_toml.py

The checker is free and offline: it parses the text with Python's tomllib and checks every image_dir
against the real folders (lab 03's flux_data/ and the playbook's assets/flux_data/).
Stuck? Compare with exercises/solutions/.
"""
import math
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import ROOT, banner, check  # noqa: E402

MOD = Path(__file__).resolve().parents[1]
PB = ROOT / "dgx-spark-playbooks" / "nvidia" / "playbook-flux-finetuning" / "assets"
EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# ── TODO 1 ── resolution is a number of pixels, not text.
# ── TODO 2 ── sparkgpu's trigger word: every concept needs its OWN rare word.
# ── TODO 3 ── image_dir must be a folder that exists (lab 03 wrote flux_data/sparkhotel/).
# ── TODO 4 ── class_tokens = "<rare trigger word> <class word>", and keep_tokens = 2 keeps both.
DATA_TOML = """
[general]
shuffle_caption = false
keep_tokens = 2

[[datasets]]
resolution = "1024"
batch_size = 1

    [[datasets.subsets]]
    image_dir = "flux_data/tjtoy"
    class_tokens = "tjtoy toy"
    num_repeats = 1

    [[datasets.subsets]]
    image_dir = "flux_data/sparkgpu"
    class_tokens = "tjtoy gpu"
    num_repeats = 2

    [[datasets.subsets]]
    image_dir = "flux_data/sparkhotel_photos"
    class_tokens = "lobby"
    num_repeats = 1
"""


# ─────────────────────────── checker — no need to edit below ────────────────
def folder(image_dir: str) -> Path | None:
    for base in (MOD, PB):
        if (base / image_dir).is_dir():
            return base / image_dir
    return None


def main() -> None:
    banner("Exercise 12 · fix data.toml", "offline checker · tomllib + the real image folders · no Spark needed",
           status=False)
    try:
        cfg = tomllib.loads(DATA_TOML)
    except tomllib.TOMLDecodeError as e:
        print(f"✕ DATA_TOML is not valid TOML: {e}")
        sys.exit(1)
    ds = cfg["datasets"][0]
    subsets = ds["subsets"]
    ok = True
    ok &= check(isinstance(ds.get("resolution"), int), f"TODO 1: resolution = {ds.get('resolution')} (a number)",
                f"TODO 1: resolution = {ds.get('resolution')!r} is a string — write resolution = 1024 without quotes")
    triggers = [s["class_tokens"].split()[0] for s in subsets if s.get("class_tokens")]
    dup = sorted({t for t in triggers if triggers.count(t) > 1})
    ok &= check(not dup, f"TODO 2: every concept has its own trigger word {triggers}",
                f"TODO 2: trigger {dup} is used by two concepts — the LoRA would blend the toy and the GPU")
    missing = [s["image_dir"] for s in subsets if folder(s["image_dir"]) is None]
    ok &= check(not missing, "TODO 3: every image_dir exists",
                f"TODO 3: {missing} not found — ls week25/12_vlm_flux_finetune/flux_data (run labs/lab03 first)")
    keep = cfg["general"].get("keep_tokens", 0)
    bad = [s["class_tokens"] for s in subsets if len(s.get("class_tokens", "").split()) != keep]
    ok &= check(not bad, f"TODO 4: every class_tokens has {keep} words: trigger + class",
                f"TODO 4: {bad} — a bare class word ('lobby') gives you nothing to prompt the new concept with")
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)

    print("\n▣ your data.toml, applied (launch_train.sh: gradient accumulation 4, 100 epochs)")
    seen = 0
    for s in subsets:
        n = sum(1 for p in folder(s["image_dir"]).iterdir() if p.suffix.lower() in EXTS)
        seen += n * s.get("num_repeats", 1)
        print(f"│ {s['class_tokens']:18s} {n} images × {s.get('num_repeats', 1)} repeat(s)")
    steps = math.ceil(seen / ds["batch_size"] / 4) * 100
    print(f"│ {seen} images per epoch → {steps} optimizer steps · prompt with: "
          + " · ".join(f"'{s['class_tokens']}'" for s in subsets))


if __name__ == "__main__":
    main()

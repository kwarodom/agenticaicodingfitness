#!/usr/bin/env python3
"""Lab 12-3 · Prepare a FLUX.1 Dreambooth LoRA dataset: concepts, trigger tokens, captions, data.toml.

Step 1 reads the playbook's own flux_data/data.toml and its 13 sample images (real files, read here).
Step 2 does the step arithmetic the training run will follow (images × repeats ÷ accumulation × epochs)
and when the LoRA files appear. Step 3 adds a third, hotel-branded concept — `sparkhotel lobby` —
with SYNTHETIC stand-in images drawn by PIL and one caption file per image, and writes a course
data.toml with all three concepts. Step 4 validates that file the way a failed run would.

Runs on this laptop. No Spark, no network. Lab 04 uploads the result and trains on the Spark.

Run: .venv/bin/python week25/12_vlm_flux_finetune/labs/lab03_flux_dataset.py
"""
import math
import random
import shutil
import sys
import tomllib
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import ROOT, banner, check, note, result, step, table, warn  # noqa: E402

MOD = Path(__file__).resolve().parents[1]
PB = ROOT / "dgx-spark-playbooks" / "nvidia" / "playbook-flux-finetuning" / "assets"
OURS = MOD / "flux_data"
EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}           # sd-scripts' IMAGE_EXTENSIONS (case-insensitive)
# From the playbook's launch_train.sh:
ACCUM, EPOCHS, SAVE_EVERY, OUTPUT_NAME = 4, 100, 25, "flux_dreambooth"
REF_HOURS = 4.0                                             # playbook: "A complete 100-epoch run takes about four hours"


def images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS) if folder.is_dir() else []


def resolve(image_dir: str) -> Path:
    """image_dir is relative to sd-scripts' working dir, where flux_data/ is mounted. Look in both copies."""
    for base in (MOD, PB):
        if (base / image_dir).is_dir():
            return base / image_dir
    return MOD / image_dir


def steps_per_epoch(cfg: dict) -> tuple[int, int]:
    """(images seen per epoch, optimizer steps per epoch) for one [[datasets]] block."""
    ds = cfg["datasets"][0]
    seen = sum(len(images_in(resolve(s["image_dir"]))) * s.get("num_repeats", 1) for s in ds["subsets"])
    return seen, math.ceil(math.ceil(seen / ds.get("batch_size", 1)) / ACCUM)


def validate(cfg: dict) -> list[str]:
    """The mistakes that stop (or quietly spoil) a Dreambooth LoRA run. Returns a list of problems."""
    problems, triggers = [], []
    keep = cfg.get("general", {}).get("keep_tokens", 0)
    for ds in cfg.get("datasets", []):
        if not isinstance(ds.get("resolution"), int):
            problems.append(f"resolution must be a number like 1024, got {ds.get('resolution')!r}")
        if not (isinstance(ds.get("batch_size"), int) and ds["batch_size"] >= 1):
            problems.append(f"batch_size must be an integer ≥ 1, got {ds.get('batch_size')!r}")
        for s in ds.get("subsets", []):
            imgs = images_in(resolve(s.get("image_dir", "")))
            tok = str(s.get("class_tokens", "")).split()
            if not imgs:
                problems.append(f"{s.get('image_dir')}: folder missing or has no images")
            elif not 5 <= len(imgs) <= 10:
                problems.append(f"{s['image_dir']}: {len(imgs)} images (the playbook suggests about 5–10 per concept)")
            if len(tok) != 2:
                problems.append(f"{s.get('image_dir')}: class_tokens {s.get('class_tokens')!r} should be "
                                "'<rare trigger word> <class word>', e.g. 'tjtoy toy'")
            elif keep and len(tok) != keep:
                problems.append(f"keep_tokens = {keep} but class_tokens has {len(tok)} words")
            triggers.append(tok[0] if tok else "")
            if not (isinstance(s.get("num_repeats", 1), int) and s.get("num_repeats", 1) >= 1):
                problems.append(f"{s.get('image_dir')}: num_repeats must be an integer ≥ 1")
    dup = {t for t in triggers if t and triggers.count(t) > 1}
    if dup:
        problems.append(f"trigger word(s) {sorted(dup)} used by more than one concept — the LoRA cannot tell them apart")
    return problems


def draw_lobby(rng: random.Random, size: int = 1024) -> Image.Image:
    """A SYNTHETIC stand-in 'hotel lobby': a wall, a floor, a desk, a round logo. It teaches FLUX nothing."""
    wall = (rng.randint(40, 70), rng.randint(70, 100), rng.randint(90, 120))
    im = Image.new("RGB", (size, size), wall)
    d = ImageDraw.Draw(im)
    d.rectangle([0, int(size * 0.62), size, size], fill=(rng.randint(180, 210), rng.randint(160, 180), 140))   # floor
    x = rng.randint(120, 380)
    d.rectangle([x, int(size * 0.48), x + 520, int(size * 0.66)], fill=(118, 185, 0), outline=(20, 20, 20), width=6)
    cx, cy, r = x + 260, int(size * 0.25), rng.randint(70, 110)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(240, 240, 240), width=12)                             # logo
    return im


banner("Lab 12-3 · FLUX.1 Dreambooth LoRA dataset — concepts, triggers, captions, data.toml",
       "the playbook's sample concepts + one hotel concept · validated on this laptop", status=False)

# ── STEP 1 ────────────────────────────────────────────────────────────────────
step(1, "the playbook's dataset: flux_data/data.toml and its images")
pb_toml = PB / "flux_data" / "data.toml"
if not pb_toml.is_file():
    sys.exit(f"✕ {pb_toml} not found — clone https://github.com/NVIDIA/dgx-spark-playbooks at the repo root.")
pb = tomllib.loads(pb_toml.read_text(encoding="utf-8"))
print(f"│ [general]  shuffle_caption = {pb['general']['shuffle_caption']} · keep_tokens = {pb['general']['keep_tokens']}")
print(f"│ [[datasets]] resolution = {pb['datasets'][0]['resolution']} · batch_size = {pb['datasets'][0]['batch_size']}")
rows = []
for s in pb["datasets"][0]["subsets"]:
    imgs = images_in(PB / s["image_dir"])
    sizes = [Image.open(p).size for p in imgs]
    caps = sum(1 for p in imgs if p.with_suffix(".caption").exists())
    rows.append([s["image_dir"], f"'{s['class_tokens']}'", len(imgs), s["num_repeats"], s["flip_aug"],
                 f"{min(w for w, _ in sizes)}–{max(w for w, _ in sizes)} px wide", caps])
table(rows, ["image_dir", "class_tokens", "images", "num_repeats", "flip_aug", "sizes", ".caption files"])
note("No caption files, so sd-scripts uses class_tokens as every image's caption. The images are not 1024×1024: "
     "the playbook's own set runs from 650 to 3018 px wide.")

# ── STEP 2 ────────────────────────────────────────────────────────────────────
step(2, "the step arithmetic for the playbook's run (launch_train.sh: accumulation 4, 100 epochs, save every 25)")
seen, per_epoch = steps_per_epoch(pb)
total = per_epoch * EPOCHS
print(f"│ images per epoch = 6 × 1 (tjtoy) + 7 × 2 (sparkgpu) = {seen} · batch 1 → {seen} forward passes")
print(f"│ optimizer steps  = ⌈{seen} ÷ {ACCUM}⌉ = {per_epoch} per epoch × {EPOCHS} epochs = {total} steps")
print(f"│ time per step    ≈ {REF_HOURS:.0f} h ÷ {total} = {REF_HOURS * 3600 / total:.0f} s   (derived from the playbook's "
      "'about four hours', not a measurement)")
print("│ LoRA files       = " + ", ".join(f"{OUTPUT_NAME}-{e:06d}.safetensors" for e in range(SAVE_EVERY, EPOCHS, SAVE_EVERY))
      + f", {OUTPUT_NAME}.safetensors (final)")
pace = REF_HOURS * 3600 / total                             # seconds per optimizer step, derived
note("File names follow sd-scripts' EPOCH_FILE_NAME '{}-{:06d}' (sd3 branch, commit b8d1eb0). The playbook: usable "
     f"concepts often appear within ~90 minutes, i.e. around epoch {round(90 * 60 / pace / per_epoch)}. The saves land at "
     f"≈ {SAVE_EVERY * per_epoch * pace / 3600:.0f} h, {2 * SAVE_EVERY * per_epoch * pace / 3600:.0f} h, "
     f"{3 * SAVE_EVERY * per_epoch * pace / 3600:.0f} h and {REF_HOURS:.0f} h — so the epoch-25 file is your first one to try.")

# ── STEP 3 ────────────────────────────────────────────────────────────────────
step(3, "add a hotel concept: flux_data/sparkhotel/ + one .caption per image (SYNTHETIC stand-ins)")
folder = OURS / "sparkhotel"
if folder.exists():
    shutil.rmtree(folder)                            # only ever this lab's own output folder
folder.mkdir(parents=True)
rng = random.Random(12)
views = ["wide shot", "view from the entrance", "evening light", "close to the desk", "from the stairs", "morning light"]
for i, view in enumerate(views, 1):
    draw_lobby(rng).save(folder / f"{i}.png", optimize=True)
    (folder / f"{i}.caption").write_text(f"sparkhotel lobby, {view}", encoding="utf-8")
print(f"→ wrote {len(views)} PNGs (1024×1024) + {len(views)} .caption files to {folder.relative_to(ROOT)}")
print("◈ SYNTHETIC — flat PIL drawings so the folder, captions and config can be checked. For a real LoRA, "
      "replace them with 5–10 real photos of the lobby from different angles and lights.")
toml_text = pb_toml.read_text(encoding="utf-8").split("[general]", 1)[1]
toml_text = ("# Week 25 · Module 12 — the playbook's flux_data/data.toml plus a third concept.\n"
             "# Written by labs/lab03_flux_dataset.py. Lab 04 uploads it to assets/flux_data/data.toml on the Spark.\n"
             "[general]" + toml_text.rstrip() + """

    [[datasets.subsets]]
    image_dir = "flux_data/sparkhotel"
    class_tokens = "sparkhotel lobby"
    num_repeats = 1
    is_reg = false
    flip_aug = false                # a mirrored logo is a different logo: no flipping for this concept
""")
(OURS / "data.toml").write_text(toml_text, encoding="utf-8")
print(f"→ wrote {(OURS / 'data.toml').relative_to(ROOT)}")

# ── STEP 4 ────────────────────────────────────────────────────────────────────
step(4, "validate both data.toml files")
good = True
for name, path in (("playbook", pb_toml), ("course", OURS / "data.toml")):
    cfg = tomllib.loads(path.read_text(encoding="utf-8"))
    problems = validate(cfg)
    for p in problems:
        print("✕ " + p)
    seen, per_epoch = steps_per_epoch(cfg)
    good &= check(not problems, f"{name} data.toml: {len(cfg['datasets'][0]['subsets'])} concepts, every folder has 5–10 "
                  f"images, two-word class_tokens, unique triggers · {seen} images/epoch → {per_epoch * EPOCHS} steps "
                  f"for {EPOCHS} epochs", f"{name} data.toml has {len(problems)} problem(s)")
hours = per_epoch * EPOCHS * REF_HOURS * 3600 / total / 3600
note(f"with the hotel concept: ≈ {hours:.1f} h for {EPOCHS} epochs at the playbook-derived pace, or ≈ {hours / 4:.1f} h "
     "with --max_train_epochs=25 (lab 04 --epochs 25).")
if not good:
    result("✕ fix data.toml before uploading.")
    sys.exit(1)
result("dataset and data.toml ready. Lab 04 uploads them and trains on the Spark (--hotel), or trains the "
       "playbook's two concepts as they are.")

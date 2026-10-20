#!/usr/bin/env python3
"""Lab 12-1 · Get a vision-language dataset into exactly the shape the playbook's trainers expect.

Image recipe (Qwen2.5-VL + GRPO): train_image_vlm.py calls load_dataset("data")["train"] and reads the
label from the FOLDER NAME ('nowildfire' → answer "No", anything else → "Yes"). Step 1 inspects the
playbook's own sample images. Step 2 builds a hotel version — "is this room ready for the next
guest?" — with small SYNTHETIC stand-in images drawn by PIL (they teach a model nothing; they let you
practise the layout). Step 3 validates both folders with the rules Hugging Face's imagefolder loader
applies, and prints the three-line change train_image_vlm.py needs for the hotel labels.

Video recipe (InternVL3 + SFT): step 4 probes the playbook's three sample videos with ffprobe (frames,
fps, duration), and step 5 validates a metadata.jsonl against the enums in the notebook's prompt —
tested on a clearly labelled EXAMPLE file — and shows how the notebook resolves each video path.

Runs on this laptop (PIL; ffprobe if installed). No Spark, no network.

Run: .venv/bin/python week25/12_vlm_flux_finetune/labs/lab01_vlm_datasets.py
"""
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import ROOT, banner, check, note, result, step, table, warn  # noqa: E402

MOD = Path(__file__).resolve().parents[1]
PB = ROOT / "dgx-spark-playbooks" / "nvidia" / "playbook-vlm-finetuning" / "assets"
SAMPLES = PB / "ui_image" / "assets" / "image_vlm" / "images"
VIDEOS = PB / "ui_video" / "assets" / "video_vlm" / "videos"
ROOMS = MOD / "data" / "room_check"
SPLITS = {"train", "test", "validation", "valid", "dev"}         # split folder names imagefolder recognises
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Enums copied from the user_prompt in ui_video/train/video_vlm.ipynb
VIDEO_ENUMS = {"event_type": {"collision", "near_miss", "no_incident"},
               "intended_action": {"turn_left", "turn_right", "change_lanes"},
               "traffic_density": {"low", "high"}, "visibility": {"good", "bad"},
               "scene": {"Urban", "Sub-urban", "Rural", "Highway"}}
VIOLATIONS = {"speeding", "failure_to_yield", "ignoring_traffic_signs"}


def validate_imagefolder(root: Path) -> tuple[bool, dict]:
    """Apply imagefolder's rules: split dirs (optional) → class dirs → images. Returns (ok, {split: {class: n}})."""
    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    splits = {p.name: p for p in subdirs} if subdirs and all(p.name in SPLITS for p in subdirs) else {"train": root}
    counts, bad = {}, []
    for split, sp in splits.items():
        counts[split] = {}
        for cls in sorted(p for p in sp.iterdir() if p.is_dir()):
            files = [f for f in cls.iterdir() if f.suffix.lower() in EXTS]
            for f in files:
                try:
                    with Image.open(f) as im:
                        im.convert("RGB")
                except Exception as e:  # noqa: BLE001
                    bad.append(f"{f.name}: {e}")
            counts[split][cls.name] = len(files)
    classes = [set(c) for c in counts.values()]
    ok = check(all(c == classes[0] for c in classes) and classes[0],
               f"{root.name}: splits {sorted(counts)} · the same classes in every split {sorted(classes[0])}",
               f"{root.name}: class folders differ between splits {counts}")
    ok &= check(not bad, f"{root.name}: every image opens and converts to RGB", f"{root.name}: unreadable images {bad[:3]}")
    return ok, counts


def draw_room(rng: random.Random, messy: bool, size: int = 256) -> Image.Image:
    """A SYNTHETIC stand-in: floor, a bed with a pillow; 'needs_attention' adds towels and a stain."""
    im = Image.new("RGB", (size, size), (rng.randint(170, 200), rng.randint(150, 175), rng.randint(120, 140)))
    d = ImageDraw.Draw(im)
    x0, y0 = rng.randint(30, 60), rng.randint(40, 70)
    d.rectangle([x0, y0, x0 + 150, y0 + 110], fill=(245, 245, 240), outline=(90, 90, 90), width=2)   # bed
    d.rectangle([x0 + 10, y0 + 8, x0 + 60, y0 + 30], fill=(255, 255, 255), outline=(150, 150, 150))  # pillow
    if messy:
        for _ in range(rng.randint(2, 4)):                                                          # towels on the floor
            tx, ty = rng.randint(10, size - 50), rng.randint(size - 70, size - 20)
            d.rectangle([tx, ty, tx + 40, ty + 14], fill=(250, 250, 250), outline=(120, 120, 120))
        sx, sy = rng.randint(x0 + 50, x0 + 120), rng.randint(y0 + 40, y0 + 90)
        d.ellipse([sx, sy, sx + 28, sy + 18], fill=(150, 110, 60))                                  # a stain
    return im


banner("Lab 12-1 · vision-language datasets — image folders and video metadata",
       "the layout the playbook's trainers expect · validated on this laptop", status=False)

# ── STEP 1 ────────────────────────────────────────────────────────────────────
step(1, "the playbook's own sample images (ui_image/assets/image_vlm/images/)")
if SAMPLES.is_dir():
    rows = []
    for f in sorted(SAMPLES.glob("*/*")):
        with Image.open(f) as im:
            rows.append([f"{f.parent.name}/{f.name}", f"{im.size[0]}×{im.size[1]}", im.mode, f"{f.stat().st_size // 1024} KB"])
    table(rows, ["file", "size", "mode", "bytes"])
    note("These are the demo's gallery images. Training uses the Kaggle Wildfire Prediction Dataset, which you "
         "download into ui_image/data/ on the Spark (lab 02).")
else:
    warn(f"playbook clone not found at {SAMPLES} — clone https://github.com/NVIDIA/dgx-spark-playbooks at the repo root")

# ── STEP 2 ────────────────────────────────────────────────────────────────────
step(2, "build the hotel version: room_check/{train,test}/{ready,needs_attention}/ (SYNTHETIC stand-ins)")
if ROOMS.exists():
    shutil.rmtree(ROOMS)                               # only ever this lab's own output folder
rng = random.Random(12)
for split, n in (("train", 8), ("test", 2)):
    for cls in ("ready", "needs_attention"):
        out = ROOMS / split / cls
        out.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            draw_room(rng, cls == "needs_attention").save(out / f"{i + 1:02d}.png")
print(f"→ wrote {sum(1 for _ in ROOMS.rglob('*.png'))} PNGs under {ROOMS.relative_to(ROOT)}")
print("◈ SYNTHETIC — flat PIL drawings to practise the folder layout. Replace them with real room photos "
      "(a few hundred per class) before training anything.")

# ── STEP 3 ────────────────────────────────────────────────────────────────────
step(3, "validate with imagefolder's rules — and what the trainer does with the folder names")
good = True
if SAMPLES.is_dir():
    ok_s, c_s = validate_imagefolder(SAMPLES)
    good &= ok_s
    note(f"playbook samples: no split folders, so everything is 'train': {c_s['train']}")
ok_r, c_r = validate_imagefolder(ROOMS)
good &= ok_r
table([[s, c, n] for s, cs in c_r.items() for c, n in cs.items()], ["split", "class (= label)", "images"])
note("train_image_vlm.py maps labels by NAME: `if label == \"nowildfire\": answer = \"No\" else \"Yes\"`. "
     "With the hotel folders, 'ready' would silently become \"Yes\" — the wrong question. Change three lines:")
print("""  --- ui_image/src/train_image_vlm.py (format_instruction)
  -    if label == "nowildfire":
  +    if label == "ready":
  -    prompt = "Identify if this region has been affected by a wildfire"
  +    prompt = "Does this hotel room need attention before the next guest arrives"
  (and the same prompt text in Image_VLM.py if you use the Streamlit demo)""")

# ── STEP 4 ────────────────────────────────────────────────────────────────────
step(4, "the playbook's sample videos (ui_video/assets/video_vlm/videos/), probed with ffprobe")
if shutil.which("ffprobe") and VIDEOS.is_dir():
    rows = []
    for v in sorted(VIDEOS.glob("*.mp4")):
        p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                            "stream=width,height,r_frame_rate,nb_read_packets:format=duration", "-of", "json", str(v)],
                           capture_output=True, text=True, timeout=60)
        d = json.loads(p.stdout or "{}")
        s, fmt = (d.get("streams") or [{}])[0], d.get("format", {})
        num, den = (s.get("r_frame_rate", "0/1").split("/") + ["1"])[:2]
        frames = int(s.get("nb_read_packets", 0))
        rows.append([v.name, f"{s.get('width')}×{s.get('height')}", f"{int(num) / int(den):.1f}",
                     f"{float(fmt.get('duration', 0)):.1f} s", frames, f"every {frames // 12}th"])
    table(rows, ["video", "size", "fps", "duration", "frames", "12 frames = "])
    note("The notebook's load_video() samples a random 8–32 evenly spaced frames per clip; the demo's inference config "
         "uses num_frames: 12. A clip of a few seconds becomes a dozen still images the model sees at once.")
else:
    print("◆ ffprobe not installed — skipping (brew install ffmpeg, or read the sizes on the Spark).")

# ── STEP 5 ────────────────────────────────────────────────────────────────────
step(5, "validate a video metadata.jsonl against the notebook's enums (on an EXAMPLE file)")
print("◈ EXAMPLE — a synthetic metadata.jsonl written to test the validator. It does not describe any real video.")
example = [
    {"video": "videos/video1.mp4", "caption": "The ego vehicle waits at a junction, then turns left behind a bus.",
     "event_type": "no_incident", "rule_violations": [], "intended_action": "turn_left",
     "traffic_density": "high", "scene": "Urban", "visibility": "good"},
    {"video": "videos/video2.mp4", "caption": "A car cuts in front of the ego vehicle, which brakes hard.",
     "event_type": "near_miss", "rule_violations": ["failure_to_yield"], "intended_action": "change_lanes",
     "traffic_density": "low", "scene": "Highway", "visibility": "bad"},
    {"video": "clips/video3.mp4", "caption": "", "event_type": "crash", "rule_violations": ["tailgating"],
     "intended_action": "turn_right", "traffic_density": "high", "scene": "urban", "visibility": "good"},
]
problems = []
for i, rec in enumerate(example, 1):
    for k, allowed in VIDEO_ENUMS.items():
        if rec.get(k) not in allowed:
            problems.append(f"line {i}: {k}={rec.get(k)!r} not in {sorted(allowed)}")
    if not set(rec.get("rule_violations", [])) <= VIOLATIONS:
        problems.append(f"line {i}: rule_violations {rec['rule_violations']} not ⊆ {sorted(VIOLATIONS)}")
    if not rec.get("caption"):
        problems.append(f"line {i}: empty caption")
    if not rec.get("video", "").startswith("videos/"):
        problems.append(f"line {i}: video path {rec['video']!r} is not under videos/")
for p in problems:
    print("✕ " + p)
good &= check(len(problems) == 5 and all("line 3" in p for p in problems),
              "validator: lines 1–2 pass; line 3's five mistakes (enum, case, violation, caption, path) are all caught",
              f"validator found {problems}")
note("How the notebook finds a clip: get_video_path() joins the PARENT of dataset_path with the record's 'video'. "
     "So point dataset_path at the metadata file itself — dataset_path = \"/data/dataset/metadata.jsonl\" — and "
     "write 'video' relative to the dataset folder: \"videos/video1.mp4\". (Read from the notebook's code; the "
     "playbook text only shows the folder layout.)")

if not good:
    result("✕ a validation failed above — fix it before you move data to the Spark.")
    sys.exit(1)
result("image folders and video metadata validated. Lab 02 checks the real Kaggle dataset on the Spark and runs the "
       "image recipe headless.")

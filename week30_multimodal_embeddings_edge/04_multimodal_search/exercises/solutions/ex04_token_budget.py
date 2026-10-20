#!/usr/bin/env python3
"""Exercise 04 · Token budget and interleaved records — reference solution.

Fill in the four TODOs, save, then run:
    week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/04_multimodal_search/exercises/ex04_token_budget.py

The checker is free and offline. Stuck? Compare with exercises/solutions/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common"))
from embedkit import banner, check  # noqa: E402

COST = {"image": 280, "video_per_s": 140, "audio_per_s": 25}   # model card, "Context Limits" (video at 1 fps)
CONTEXT = 8192
PLACEHOLDER = {"image": "<|image|>", "video": "<|video|>", "audio": "<|audio|>"}


# ── TODO 1 ── tokens one interleaved record costs
def record_tokens(text_tokens: int, images: int, video_s: float, audio_s: float) -> int:
    return int(text_tokens + images * COST["image"] + video_s * COST["video_per_s"] + audio_s * COST["audio_per_s"])


# ── TODO 2 ── does it fit the shared context? and how many more seconds of video would?
def fits(tokens: int) -> bool:
    return tokens <= CONTEXT


def seconds_of_video_left(tokens: int) -> int:
    return max(0, (CONTEXT - tokens) // COST["video_per_s"])


# ── TODO 3 ── list the problems in an interleaved record
def check_record(record: dict) -> list:
    problems = []
    text = record.get("text", "")
    for key, tag in PLACEHOLDER.items():
        if key not in record:
            continue
        items = record[key]
        n_items = 1 if isinstance(items, str) else len(items)
        n_tags = text.count(tag)
        if n_tags != n_items:
            problems.append(f"{key}: {n_tags} placeholder{'s' if n_tags != 1 else ''} for {n_items} items")
    return problems


# ── TODO 4 ── which task prefix goes in front of an image, a clip or a voicemail? (a string, or None)
PREFIX_FOR_MEDIA = None


# ─────────────────────────── checker — no need to edit below ────────────────
def main() -> None:
    banner("Exercise 04 · token budget + interleaved records", "offline checker · no model needed")
    good = True
    try:
        t = [record_tokens(text_tokens=30, images=1, video_s=10, audio_s=0),
             record_tokens(text_tokens=0, images=0, video_s=0, audio_s=60),
             record_tokens(text_tokens=100, images=2, video_s=0, audio_s=12)]
    except Exception as e:  # noqa: BLE001
        t = f"error: {e}"
    good &= check(t == [1710, 1500, 960], "record_tokens: 30+280+10×140 = 1710 · 60 s audio = 1500 · 100+560+300 = 960",
                  f"TODO 1: record_tokens gave {t} — image 280, video 140 per second (1 fps), audio 25 per second")
    try:
        f = [fits(1710), fits(8192), fits(8193)]
        left = seconds_of_video_left(1710)
    except Exception as e:  # noqa: BLE001
        f, left = f"error: {e}", None
    good &= check(f == [True, True, False] and left == 46,
                  "fits(): ≤ 8,192 tokens · 1,710 used leaves room for 46 more seconds of video",
                  f"TODO 2: fits → {f}, seconds_of_video_left(1710) → {left} (want [True, True, False] and 46)")
    ok_rec = {"text": "Leak at AHU-3 <|image|> and again <|image|>", "image": ["a.jpg", "b.jpg"]}
    bad_rec = {"text": "CCTV <|video|> photo <|image|>", "image": ["a.jpg", "b.jpg"], "video": "c.mp4"}
    no_tag = {"text": "Leak at AHU-3", "image": ["a.jpg"]}
    try:
        v = [check_record(ok_rec), check_record(bad_rec), check_record(no_tag)]
    except Exception as e:  # noqa: BLE001
        v = f"error: {e}"
    good &= check(isinstance(v, list) and v[0] == [] and len(v[1]) == 1 and "image" in v[1][0] and len(v[2]) == 1,
                  "check_record: one placeholder per media item, per modality — problems listed, none for a good record",
                  f"TODO 3: check_record gave {v}")
    good &= check(PREFIX_FOR_MEDIA is None,
                  "media items get NO task prefix (prefixes are for text only)",
                  f"TODO 4: PREFIX_FOR_MEDIA = {PREFIX_FOR_MEDIA!r} — what does the model card say about prefixes and media?")
    if not good:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)
    print("\n═ Done. check_record() is worth keeping in any ingest pipeline: a missing <|image|> fails silently.")


if __name__ == "__main__":
    main()

# Week 23 Media Manifest

Generated 2026-08-04 by media-producer (team week23-media, task #2) via Higgsfield MCP.
All 15 images: model requested `nano_banana_pro` (backend executed/reported as `nano_banana_2`), prompts taken **verbatim** from `MEDIA_PLAN.md`.
Video: `kling3_0_turbo`, 16:9, 8 s, 1080p.

Note on the prompt column: every image prompt begins with the shared verbatim style prefix
("Cinematic tech-noir 3D render on a near-black background, dark …"), so the first 10 words shown
below are the first 10 words of each prompt's **scene description** (the part after the prefix) to
keep rows distinguishable.

> **Filenames below are the generated `.png` masters, now in `originals/` (gitignored).** What ships
> in `media/` — and what the README/SLIDES embed — are web-optimized `.jpg` copies with the same
> basenames (see the Optimization section at the bottom).

| Filename | Module | Prompt (first 10 words of scene) | Generation ID | Deviations |
|---|---|---|---|---|
| `hero-banner.png` | Hero banner (21:9, 4k, 3168×1344) | An ultra-wide panorama of a four-layer floating stack of… | `f8dbc9d7-8977-4703-9677-c841155bb143` | none |
| `00-stack-navigator.png` | 00 · Stack Navigator hub | An isometric dark control hub: a central glowing green node… | `b0ccc762-8bfb-4de6-ae8b-03f625c0430c` | none |
| `01-nemotron-models.png` | 01 · Nemotron 3 model family | Three glowing green crystalline polyhedral cores of increasing size floating… | `4c6dc34f-4202-4a25-a659-7b9590d5d545` | none |
| `02-nim-microservices.png` | 02 · NIM signed containers | A sleek dark shipping-container-like cube with softly glowing green seams… | `35b555ff-8189-40dc-834f-182e38b28987` | none |
| `03-dynamo-serving.png` | 03 · Dynamo disaggregated serving | Two separate banks of dark server towers on opposite sides… | `b4513bd4-fac8-4d4f-aca7-dfd69765b419` | none |
| `04-agent-skills.png` | 04 · Portable Agent Skills | A set of identical dark modular cartridges with green glowing… | `545000c3-a9d3-468b-8993-b61cdb066dfa` | none |
| `05-aiq-research-lab.png` | 05 · AI-Q deep-research lab | An isometric dark research laboratory: a central green analytical lens… | `65102b6a-ecaa-445d-9a6f-8ab825b50f6f` | none |
| `06-nemoclaw.png` | 06 · NemoClaw agent fleet | A fleet of five distinct dark robotic units in formation… | `47842ee6-4cf7-4c87-8017-18c681c0ea00` | none |
| `07-guardrails-openshell.png` | 07 · OpenShell guardrails & sandbox | A glowing green agent orb working inside a translucent hexagonal… | `9b24b9f5-8503-48c5-8887-7e3d0871cc94` | none |
| `08-nemo-relay.png` | 08 · NeMo Relay observability | A dark observation deck where a cascade of luminous trace… | `8bbef5ad-a62f-4b2c-97c0-9248cc070810` | none |
| `09-inference-economics.png` | 09 · Inference economics | A dark industrial balance scale: on one side a towering… | `fe87d8b7-8516-40da-8675-2bfd84577e76` | none |
| `10-nemo-gym-rl.png` | 10 · NeMo Gym RL training | An isometric dark training gymnasium of parallel obstacle-course lanes, each… | `16eeb99f-4088-4c47-9f23-c5dd8fdfbd95` | none |
| `11-data-flywheel.png` | 11 · Data flywheel loop | A massive dark flywheel ring spinning in space, its rim… | `0774dd63-b957-4fd7-8e6d-a3e1d19b5d65` | none |
| `12-capstone-smart-hotel.png` | 12 · Sovereign smart hotel, Bangkok | An isometric cutaway of a sleek high-rise hotel tower at… | `fcfe2eff-26a5-47f7-8689-c0f99c34c9c4` | none |
| `capstone-cinematic.png` | Capstone cinematic (16:9, 4k, 5504×3072) | A dramatic low-angle night shot of a futuristic hotel tower… | `05dcc7e9-7ec3-4207-96fe-ab2ed66887a8` | faint distant pink/magenta neon street glow (illegible, no readable signage); acceptable for a Bangkok street scene |
| `intro-video.mp4` | Intro video (16:9, 8 s, 1080p) | Cinematic tech-noir 3D animation, near-black world with NVIDIA-green (#76B900) and… | `d4ae66e6-b7f3-45eb-be89-62cef619e678` | none |

## Production notes

- **Model deviation (all images):** requests were submitted with `model: nano_banana_pro` per plan quality guidance; the Higgsfield backend executed and reported the jobs under model id `nano_banana_2` (Google Nano Banana family). Output quality, aspect ratios (21:9 / 16:9) and resolutions (2k cards, 4k hero + capstone) all match the plan.
- **Hero banner:** generated natively at 21:9 (no outpaint/reframe fallback needed), 3168×1344 — exceeds the 2K-width requirement of §7.1.
- **Quality gate (§7.4):** all 15 images visually inspected; zero legible text, letters, numbers, or logos found. **No regenerations were required** (0 of the 1-allowed retries used per asset; 16 total generations, well under the 20-image budget).
- **File formats:** every URL served true PNG (verified with `file`); video is ISO MP4. All filenames match §2 exactly.
- All files verified present and > 50 KB; total payload ≈ 123 MB.

## Optimization

Web optimization performed 2026-08-04 by integrator (task #3), using macOS `sips`:

- All 15 master `.png` files (≈ 110 MB, 2K-4K) were **moved to `originals/`**, which is
  **gitignored** (`media/.gitignore`) so the heavy masters never enter git.
- Each master was resampled and re-encoded as a web `.jpg` in `media/` with the **same basename**
  (`sips -s format jpeg -s formatOptions 72 --resampleWidth 1600` — hero banner at width 2000).
  Every JPG landed between 154 KB and 283 KB (< 500 KB target); no quality reduction below
  formatOptions 72 was needed.
- `intro-video.mp4` is 13.8 MB (< 25 MB threshold), so it stays in `media/` unmoved.
- Result: web payload in git ≈ **16.8 MB** (≈ 3.2 MB images + 13.8 MB video), down from ≈ 123 MB.
- **Course content embeds the `.jpg` web copies only**; the PNG masters remain available locally
  in `media/originals/` for print/slides re-export.

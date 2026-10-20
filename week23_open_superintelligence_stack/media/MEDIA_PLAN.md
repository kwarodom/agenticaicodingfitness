# Week 23 — Media Plan & Style Guide
### "The Open Superintelligence Stack" · Agent = Model + Harness

Authored by: art-director (team week23-media) · For: media-producer (task #2)
Scope: 1 hero banner · 13 module cards (apps 00–12) · 1 cinematic capstone image · 1 intro video.
All prompts below are **exact, self-contained Higgsfield prompts** — copy verbatim, no edits needed.

---

## 1 · STYLE GUIDE — one visual language for every asset

The course apps share a "sovereign terminal" identity (see `week23/SLIDES.md` §2 and every
`static/guide.html`): near-black `#0c0f0a` background, `#13160f` panels with 1px `#2c3322`
hairlines, NVIDIA-green `#76b900` as the single dominant accent, cyan `#56d4dd` for data-in-motion,
amber `#d29922` for warnings, violet `#bc8cff` for advanced topics, and inline-SVG schematic
diagrams (rounded panel boxes, labeled arrows, glowing accents). Media must feel like those
diagrams came to life in 3D.

### Visual language rules

| Rule | Decision |
|---|---|
| **Mood** | Cinematic tech-noir. A sovereign machine room at night — calm, precise, powerful. Not busy, not "AI slop collage". |
| **Palette** | Near-black `#0c0f0a` base · NVIDIA green `#76B900` primary glow · cyan `#56d4dd` secondary (data flows only) · amber sparingly for heat/warning motifs. No other hues. |
| **Form language** | Isometric / schematic 3D: dark matte panels, rounded-corner slabs, hairline light edges, circuit traces, glass-and-graphite server hardware. Mirrors the guide.html diagram culture (panels + arrows + glow). |
| **Lighting** | Volumetric green under-glow and edge light against deep black; thin light beams as "data"; shallow depth of field on cards. |
| **NO TEXT** | Absolutely no text, letters, numbers, words, labels, logos, or UI type baked into any image — generative text renders badly and all real labels are overlaid later in HTML/CSS. Every prompt carries an explicit negative clause. |
| **Consistency device** | Every image prompt starts with the same STYLE PREFIX (below), so the 15 stills read as one family. |
| **Composition** | One clear hero object per card, centered or rule-of-thirds, generous negative space at top (room for an HTML title overlay), floor reflection optional. |
| **Accent discipline** | Green = the component being taught (the "lit layer"). Cyan = data in motion. Follows the SLIDES.md diagram convention. |

### Reference screens (Mobbin, absorbed into the rules above)

- [Midday dashboard](https://mobbin.com/screens/973fa6a1-9e7f-4777-8429-66e9283ca007) — near-black minimalism, huge negative space, hairline dividers.
- [Twenty dashboard](https://mobbin.com/screens/c017a665-7d85-4ae7-a659-1dbde26c1882) — dark card grid, muted panels, restrained accent color.
- [Toggl Track admin](https://mobbin.com/screens/49e9fd09-eefc-4b2f-95aa-ca4fc97b8746) — single-accent discipline on a dark dashboard.

Takeaway applied: keep imagery quiet enough to sit behind/next to dense dashboard UI — one glowing
subject, black everywhere else.

### THE STYLE PREFIX (verbatim start of every image prompt)

> Cinematic tech-noir 3D render on a near-black background, dark matte graphite panels with hairline edge lighting, luminous NVIDIA-green (#76B900) accent glow with subtle cyan secondary light, isometric schematic style, volumetric light, ultra-detailed, photorealistic CGI.

### THE NEGATIVE CLAUSE (verbatim end of every image prompt)

> No text, no words, no letters, no numbers, no logos, no watermarks, no UI labels, no people.

---

## 2 · ASPECT RATIOS & FILENAMES

| Asset | Ratio | Target filename (in `week23/media/`) |
|---|---|---|
| Hero banner | **21:9** (fallback 16:9 if unsupported) | `hero-banner.png` |
| Module cards 00–12 | **16:9** | `00-stack-navigator.png` `01-nemotron-models.png` `02-nim-microservices.png` `03-dynamo-serving.png` `04-agent-skills.png` `05-aiq-research-lab.png` `06-nemoclaw.png` `07-guardrails-openshell.png` `08-nemo-relay.png` `09-inference-economics.png` `10-nemo-gym-rl.png` `11-data-flywheel.png` `12-capstone-smart-hotel.png` |
| Capstone cinematic | **16:9** | `capstone-cinematic.png` |
| Intro video | **16:9**, 5–10 s | `intro-video.mp4` |

---

## 3 · HERO BANNER (21:9) — `hero-banner.png`

**"The Open Superintelligence Stack — Agent = Model + Harness"**

> Cinematic tech-noir 3D render on a near-black background, dark matte graphite panels with hairline edge lighting, luminous NVIDIA-green (#76B900) accent glow with subtle cyan secondary light, isometric schematic style, volumetric light, ultra-detailed, photorealistic CGI. An ultra-wide panorama of a four-layer floating stack of dark slabs seen edge-on: a radiant green crystalline core (the model) suspended at the center, wrapped by an exoskeleton of dark mechanical harness rings, cables and orbiting tool modules connected by thin cyan light-traces, all hovering above a reflective black floor that recedes into darkness. Composition leaves clean empty black space along the top third for a title overlay. No text, no words, no letters, no numbers, no logos, no watermarks, no UI labels, no people.

---

## 4 · MODULE CARDS (16:9, one per app)

### 00 · Stack Navigator hub — `00-stack-navigator.png`
> Cinematic tech-noir 3D render on a near-black background, dark matte graphite panels with hairline edge lighting, luminous NVIDIA-green (#76B900) accent glow with subtle cyan secondary light, isometric schematic style, volumetric light, ultra-detailed, photorealistic CGI. An isometric dark control hub: a central glowing green node ringed by twelve smaller dark rounded tiles arranged in an orbit, each tile connected to the center by a thin cyan light-path, like a subway map of a technology stack floating over a black reflective floor. No text, no words, no letters, no numbers, no logos, no watermarks, no UI labels, no people.

### 01 · Nemotron 3 open model family — `01-nemotron-models.png`
> Cinematic tech-noir 3D render on a near-black background, dark matte graphite panels with hairline edge lighting, luminous NVIDIA-green (#76B900) accent glow with subtle cyan secondary light, isometric schematic style, volumetric light, ultra-detailed, photorealistic CGI. Three glowing green crystalline polyhedral cores of increasing size floating in a row — small, medium, and massive — each with a visible lattice of internal light filaments, hovering above dark hexagonal pedestals, representing an open family of AI models from compact to frontier scale. No text, no words, no letters, no numbers, no logos, no watermarks, no UI labels, no people.

### 02 · NIM signed containers — `02-nim-microservices.png`
> Cinematic tech-noir 3D render on a near-black background, dark matte graphite panels with hairline edge lighting, luminous NVIDIA-green (#76B900) accent glow with subtle cyan secondary light, isometric schematic style, volumetric light, ultra-detailed, photorealistic CGI. A sleek dark shipping-container-like cube with softly glowing green seams and an embossed circular wax-seal-style emblem of pure light on its face, a small crystalline model core visible through a glass window inside it, sitting on a dark loading dock with a single cyan conveyor light-path leading away. No text, no words, no letters, no numbers, no logos, no watermarks, no UI labels, no people.

### 03 · Dynamo disaggregated serving — `03-dynamo-serving.png`
> Cinematic tech-noir 3D render on a near-black background, dark matte graphite panels with hairline edge lighting, luminous NVIDIA-green (#76B900) accent glow with subtle cyan secondary light, isometric schematic style, volumetric light, ultra-detailed, photorealistic CGI. Two separate banks of dark server towers on opposite sides of the frame — one bank pulsing with dense green intake light, the other emitting fine streams of cyan output particles — bridged by thick luminous transfer conduits carrying packets of light between them, an isometric traffic-router node hovering above and directing the flow. No text, no words, no letters, no numbers, no logos, no watermarks, no UI labels, no people.

### 04 · Portable Agent Skills — `04-agent-skills.png`
> Cinematic tech-noir 3D render on a near-black background, dark matte graphite panels with hairline edge lighting, luminous NVIDIA-green (#76B900) accent glow with subtle cyan secondary light, isometric schematic style, volumetric light, ultra-detailed, photorealistic CGI. A set of identical dark modular cartridges with green glowing connector pins floating between three different dark robotic sockets, one cartridge mid-insertion with a satisfying snap of light, conveying plug-and-play capabilities that fit any machine. No text, no words, no letters, no numbers, no logos, no watermarks, no UI labels, no people.

### 05 · AI-Q deep-research lab — `05-aiq-research-lab.png`
> Cinematic tech-noir 3D render on a near-black background, dark matte graphite panels with hairline edge lighting, luminous NVIDIA-green (#76B900) accent glow with subtle cyan secondary light, isometric schematic style, volumetric light, ultra-detailed, photorealistic CGI. An isometric dark research laboratory: a central green analytical lens or magnifying optic hovering over a grid of floating translucent document panes, with several small drone-like sub-agent orbs fanning out along cyan search beams to gather glowing fragments and return them to the lens. No text, no words, no letters, no numbers, no logos, no watermarks, no UI labels, no people.

### 06 · NemoClaw specialized agent fleet — `06-nemoclaw.png`
> Cinematic tech-noir 3D render on a near-black background, dark matte graphite panels with hairline edge lighting, luminous NVIDIA-green (#76B900) accent glow with subtle cyan secondary light, isometric schematic style, volumetric light, ultra-detailed, photorealistic CGI. A fleet of five distinct dark robotic units in formation on a black reflective deck, each with a different green-lit tool appendage — a claw, a probe, an antenna, a shield, a lens — assembled from the same modular chassis, a larger command unit behind them, conveying specialized agents built from one base. No text, no words, no letters, no numbers, no logos, no watermarks, no UI labels, no people.

### 07 · OpenShell guardrails & sandbox — `07-guardrails-openshell.png`
> Cinematic tech-noir 3D render on a near-black background, dark matte graphite panels with hairline edge lighting, luminous NVIDIA-green (#76B900) accent glow with subtle cyan secondary light, isometric schematic style, volumetric light, ultra-detailed, photorealistic CGI. A glowing green agent orb working inside a translucent hexagonal containment shell with visible energy walls, a single gated aperture in the shell where a guarded beam of cyan light is allowed out through a ring of dark security checkpoints, everything else sealed, conveying safe sandboxed autonomy. No text, no words, no letters, no numbers, no logos, no watermarks, no UI labels, no people.

### 08 · NeMo Relay observability traces — `08-nemo-relay.png`
> Cinematic tech-noir 3D render on a near-black background, dark matte graphite panels with hairline edge lighting, luminous NVIDIA-green (#76B900) accent glow with subtle cyan secondary light, isometric schematic style, volumetric light, ultra-detailed, photorealistic CGI. A dark observation deck where a cascade of luminous trace threads fans out like a horizontal waterfall of light — each thread splitting into finer branching spans of green and cyan — passing through a floating glass prism relay that renders them visible, conveying every call being observed and traced. No text, no words, no letters, no numbers, no logos, no watermarks, no UI labels, no people.

### 09 · Inference economics — tokens per megawatt — `09-inference-economics.png`
> Cinematic tech-noir 3D render on a near-black background, dark matte graphite panels with hairline edge lighting, luminous NVIDIA-green (#76B900) accent glow with subtle cyan secondary light, isometric schematic style, volumetric light, ultra-detailed, photorealistic CGI. A dark industrial balance scale: on one side a towering column of tiny glowing green token-like coins of light, on the other a humming amber-tinged power turbine with electrical arcs, connected by circuit traces across a black factory floor, conveying the economics of intelligence measured in tokens per megawatt. No text, no words, no letters, no numbers, no logos, no watermarks, no UI labels, no people.

### 10 · NeMo Gym RL training — `10-nemo-gym-rl.png`
> Cinematic tech-noir 3D render on a near-black background, dark matte graphite panels with hairline edge lighting, luminous NVIDIA-green (#76B900) accent glow with subtle cyan secondary light, isometric schematic style, volumetric light, ultra-detailed, photorealistic CGI. An isometric dark training gymnasium of parallel obstacle-course lanes, each lane a glowing circuit track where a small agent orb runs trials, successful runs flaring bright green while failed runs fade dim, a reward beacon at the finish line emitting concentric rings of light, conveying reinforcement learning through repeated verifiable trials. No text, no words, no letters, no numbers, no logos, no watermarks, no UI labels, no people.

### 11 · Data flywheel loop — `11-data-flywheel.png`
> Cinematic tech-noir 3D render on a near-black background, dark matte graphite panels with hairline edge lighting, luminous NVIDIA-green (#76B900) accent glow with subtle cyan secondary light, isometric schematic style, volumetric light, ultra-detailed, photorealistic CGI. A massive dark flywheel ring spinning in space, its rim made of streaming data particles that pass through four glowing stations around the circle — a collector funnel, a refinery filter, a forge anvil striking sparks, and a judging lens — each revolution leaving the central model core brighter and denser, conveying a self-improving loop. No text, no words, no letters, no numbers, no logos, no watermarks, no UI labels, no people.

### 12 · Sovereign autonomous hotel, Bangkok — `12-capstone-smart-hotel.png`
> Cinematic tech-noir 3D render on a near-black background, dark matte graphite panels with hairline edge lighting, luminous NVIDIA-green (#76B900) accent glow with subtle cyan secondary light, isometric schematic style, volumetric light, ultra-detailed, photorealistic CGI. An isometric cutaway of a sleek high-rise hotel tower at night in a Bangkok-inspired skyline, its floors revealed as a living circuit board with green light pulsing through room grids, elevators and climate systems, a small glowing AI core in the basement plant room radiating control traces up the entire building, warm distant city lights far behind. No text, no words, no letters, no numbers, no logos, no watermarks, no UI labels, no people.

---

## 5 · CINEMATIC CAPSTONE IMAGE (16:9) — `capstone-cinematic.png`

> Cinematic tech-noir 3D render on a near-black background, dark matte graphite panels with hairline edge lighting, luminous NVIDIA-green (#76B900) accent glow with subtle cyan secondary light, isometric schematic style, volumetric light, ultra-detailed, photorealistic CGI. A dramatic low-angle night shot of a futuristic hotel tower rising over a rain-slicked Bangkok street, every window pulsing in a synchronized green wave as an unseen intelligence runs the building, a translucent holographic exoskeleton of harness rings and data conduits faintly visible wrapping the tower, reflections shimmering in the wet asphalt, epic cinematic composition with atmospheric haze. No text, no words, no letters, no numbers, no logos, no watermarks, no UI labels, no people.

---

## 6 · INTRO VIDEO (16:9, 5–10 s) — `intro-video.mp4`

> Cinematic tech-noir 3D animation, near-black world with NVIDIA-green (#76B900) and cyan accent light. A slow dolly-in on a radiant green crystalline core floating in darkness; as the camera approaches, dark mechanical harness rings, tool modules and cable conduits assemble around it piece by piece with precise magnetic snaps, cyan data-light beginning to circulate through the completed exoskeleton; the final frame holds on the finished agent — core plus harness — pulsing steadily like a heartbeat. Smooth cinematic camera, volumetric light, no text, no words, no logos, no people.

**Direction notes for media-producer:** 5–10 s, 16:9, one continuous shot (no cuts), slow and
deliberate motion; ambient low hum audio optional; the "snap-assembly" beats should read as
Model → Harness → Agent.

---

## 7 · PRODUCTION NOTES (for task #2)

1. Generate at the highest resolution available; upscale to at least 2K width for the hero.
2. If 21:9 is unsupported for the hero, generate 16:9 then outpaint/reframe to 21:9.
3. Batch the 13 module cards with `generate_image_batch` using the prompts verbatim — the shared
   style prefix is what keeps the set coherent; do not paraphrase it.
4. Reject and regenerate any output containing legible text, letters, or logos — the negative
   clause reduces but does not eliminate this.
5. Save with the exact filenames in §2, all inside `week23/media/`.

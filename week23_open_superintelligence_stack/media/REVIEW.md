# Week 23 Media Integration — Cold-Read Review

Reviewer: critic (team week23-media, task #4) · 2026-08-04
Scope: `week23/README.md`, `week23/SLIDES.md`, `week23/media/` (manifest, plan, assets, .gitignore, git status).

## Verdict: PASS

The media genuinely helps a first-time reader. The hero banner sets the "Agent = Model +
Harness" theme before a word is read; the 01→12 learning path is untouched and still obvious
(numbered tables first, thumbnails after); every thumbnail sits under the correct phase table
(Phase 0: 00 · Phase 1: 01 · Phase 2: 02–03 · Phase 3: 04–06 · Phase 4: 07 · Phase 5: 08–09 ·
Phase 6: 10–11 · Phase 7: 12 + cinematic). Nothing clutters — cards are 240px, linked to their
folders, and placed after (never inside) the tables they illustrate.

## Checks performed

| Check | Result |
|---|---|
| Every `src`/media `href` in README.md + SLIDES.md resolves from the file's own dir | PASS — 15 unique image paths, all exist under `week23/media/` |
| Relative-path correctness (both files live at `week23/`, all refs `media/...`) | PASS |
| HTML sanity | PASS — README: 8/8 `<p>` pairs, 15/15 `<img>` with alt, 13/13 `<a>` pairs; SLIDES: 1/1, 4/4 alt, no `<a>` |
| Embedded .jpg sizes | PASS — all 15 between 154 KB and 283 KB (< 500 KB); nothing embedded > 5 MB |
| Visual spot-check (Read tool on the actual pixels) | PASS — 03 dynamo: two server banks (green intake / cyan output) + luminous transfer conduits + router node above, exactly per caption & plan. 07 openshell: green agent core inside translucent hex containment shell, gated aperture beaming light out. hero-banner: green crystal core wrapped in harness rings over stacked slabs, clean negative space up top. 12 capstone: isometric hotel cutaway, basement AI core, Bangkok skyline. Zero legible/garbled baked-in text in any of the four. |
| `media/.gitignore` contains `originals/` | PASS — `git check-ignore -v` confirms `media/originals/` is ignored |
| manifest.md vs reality | PASS with one nit (see M2) — all 15 web `.jpg` + `intro-video.mp4` present in `media/`, 15 masters in `originals/`, sizes match the manifest's 154–283 KB claim |
| MEDIA_PLAN.md vs reality | PASS — all §2 basenames produced; extension divergence (.png→.jpg web copies) is documented in manifest §Optimization |

## Findings

### Minor

- **M1 — `media/intro-video.mp4` (13.8 MB) is committed but referenced by nothing.**
  `week23/README.md` and `week23/SLIDES.md` never embed or link it (grep: zero hits outside
  `media/manifest.md`). It will be the single largest object added to git this week while being
  invisible to learners. **Recommendation (structural, not fixed):** either add a one-line link
  in README's Media footer (`README.md:202-204`) — note GitHub won't inline-play a repo-relative
  mp4 from markdown, so a plain link is the honest option — or move it to `media/originals/` until
  something uses it.

- **M2 — `media/manifest.md:14-29` table lists `.png` filenames, but the files shipping in
  `media/` are `.jpg`.** The Optimization section (manifest.md:39-52) explains the conversion,
  but a reader scanning only the table will look for files that aren't there. **Recommendation:**
  retitle the column "Master filename (in `originals/`)" or list both names. (Not fixed — manifest
  is outside this task's allowed edit set.)

- **M3 — `git status --short week23/` shows far more than the expected README/SLIDES/media diff:**
  8 modified `tutorial_server.py` files, untracked `TUTORIAL.md`/`labs/` in every app,
  `00_stack_navigator/`, `AGENTS.md`, `issues/`. None of these are media-task changes — they
  appear to be parallel work by other teams/sessions. Not a media defect, but the media work
  cannot be committed in isolation without deliberate pathspec staging (`week23/README.md
  week23/SLIDES.md week23/media/`). `originals/` is correctly invisible (ignored inside the
  untracked `media/` dir).

### Nit

- **N1 — README.md:31 vs README.md:40:** Phase 0 is "recommended entry point" and Phase 1 is
  "start here". Both are defensible (hub first, then module 01), but a newcomer reads two
  competing "start" signals ~10 lines apart. Suggest "Phase 1 · The Model — the first module"
  or similar. (Wording judgment call, left to the author.)

- **N2 — README.md:110-113:** the 12-capstone card thumbnail and the 720px capstone-cinematic
  image appear back-to-back — two hotel renders in a row. It reads as intentional flourish and
  the cinematic differs clearly (street-level night shot vs isometric cutaway), so no change
  required; a one-line caption under the cinematic would remove any doubt.

## Fixes applied by reviewer

None required. All paths resolved, all alt attributes were present, and no caption typos were
found, so no edits were made to README.md or SLIDES.md.

## Blockers / Major

None.

# ✦ Agentic Quest

The whole **Agentic Coding Fitness** course, Week 2 → Week 29 plus the SO-ARM101 robotics track, as an animated
quest inside Claude Code. It is a mod: a panel in your terminal with a pixel-art banner (a little robot runs across a
city skyline), a metro map of the course, animated flow diagrams of each week's core idea, a lab guide with the exact
commands, quizzes, XP, levels and badges.

Needs Claude Code 2.1.287+ (built and tested on 2.1.292).

## Install

From a clone of this repo:

```bash
claude plugin marketplace add ./plugins/course-quest
claude plugin install course-quest@course-quest
```

From GitHub, once this folder is pushed:

```text
/plugin marketplace add kwarodom/agenticaicodingfitness
/plugin install course-quest@agentic-coding-fitness
```

Then type **`/quest`** (or `/reload-plugins` first in a session that was already open).

## The panel

| key | tab / action |
|---|---|
| **m** | 🗺 **Map**: one metro line per phase; ● done ◐ started ○ new; ◉ pulses where you are and a train runs on your line. Pick any week from the list. |
| **w** | 📖 **Week**: tagline, summary, an animated flow (a ◆ token runs through the boxes, and loops back for loops), concept cards, what you need, a fun fact |
| **l** | 🧪 **Labs**: the "start here" command, every lab with its command, port link and mode; **c** copy · **r** run with Claude · **d** mark done |
| **k** | 🧠 **Quiz**: three questions per week; press **1–4**, **j** for the next one; confetti when you get it right |
| **s** | 🏅 **Stats**: level, XP, shimmering progress bars per phase, badges, streak |
| **p / n** | previous / next week |
| **t** | **Teach me**: puts a lesson request in your prompt (Claude reads the week's docs and teaches it with examples) |
| **g** | **Guide my labs**: puts a step-by-step lab walkthrough request in your prompt (checks prerequisites first, asks before starting servers) |
| **x** / Esc | close |

`/quest [<week id>|map|labs|quiz|stats|next|prev|band on|off|reset]`, e.g. `/quest 15`, `/quest R1`.
A one-line band above the prompt shows your level and current week (`/quest band off` hides it).

XP: +5 for visiting a week, +20 per lab marked done, +10 per right answer, +15 for a perfect quiz.
Progress is kept on this machine in the plugin's store.

## Claude knows the course too

The mod registers two tools, so you can just ask Claude ("what's week 15 about?", "how do I run the week 24 labs?"):

- `course_week({ week })`: the week's facts, every lab's exact command and port, and your progress
- `course_overview()`: all weeks with phase, tagline and your state, plus XP and level

## What it does to your session

From `claude plugin validate`: it hooks `session.start`, its own `/quest` command and two tools, and draws a band and
its pane. It reads and writes only its own store, and never runs a lab itself: commands run only when you or Claude
run them. While the pane is open, a 110 ms clock repaints its pictures with `$.ui.blit`.

## Develop

```bash
claude plugin validate plugins/course-quest
claude plugin test plugins/course-quest      # 7 tests: every tab × every week, narrow and wide, quiz, labs, tools, animation
```

The course data (`hooks/course.ts`) was surveyed from each week's own docs; every lab path and readme was checked to
exist. Regenerate it rather than hand-edit when weeks change. The pictures are in `hooks/canvas.ts` (a small
true-color character canvas for the terminal's `Raster` element).

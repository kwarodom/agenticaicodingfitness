# ▶ Jev Lab 12 — Capstone: a smart-hotel guest-request copilot

> Part of Week 24 · Typed AI decisions with Jev. Everything from Labs 01–11 in one small, real-shaped system. A guest message (EN / TH / mixed) becomes a routed, prioritized, safety-checked **ticket proposal**, and nothing is ever executed on its own.

**In plain words:** a guest writes *"The toilet in 1507 is overflowing!"*. Jev answers five quick questions about it: which team, how serious, any danger, does a person need to look, is a room number mentioned. Your Python code then does the exact parts itself (it finds `1507` with a regex and detects Thai vs English), applies simple written rules, and prints a ticket that a human approves. You build that whole path in this lab.

**Two words you will meet:** an *uncertainty gate* is a rule like "if Jev is less than 70% sure of the team, send it to a person" (this lab uses 0.70). *Shadow mode* means running the new system beside the old one, logging where they disagree and changing nothing for real users.

**What you'll actually do**
- Design one batched Jev request with five independent judgments.
- Keep exact jobs in code: language detection and room-number extraction.
- Write a policy with a safety override, an uncertainty gate and priorities, and test it offline.
- Produce a JSON ticket plus a templated reply draft with `execute: False`.
- Run the new copilot in **shadow mode** beside a keyword router and read the disagreements.
- Build the whole thing yourself in the capstone exercise.

**Time** ~60 min · **Difficulty** intermediate → advanced · **Cost** ≈ $0.001 live · $0 dry

## 0 · The design on one page

A guest writes: *"There's a burning smell coming from the air conditioner in 1507!"* Here is who does what:

```text
guest message
  ├─ CODE  detect_language()   Thai Unicode block?  → "en" / "th" / "mixed"      exact, free
  ├─ CODE  extract_room()      regex                → "1507"                     exact, free
  ├─ JEV   ONE batched call    department · severity · safety · needs_human · mentions_room
  ├─ CODE  policy()            safety ≥ 0.20 → duty manager, P0, human approval
  │                            unknown / torn → front-desk review
  │                            severity → P1 / P2 / P3
  └─ CODE  build_ticket()      JSON ticket + reply DRAFT from approved templates · execute: False
```

The rule behind every line is the one you've used since Lab 01: **code for facts, Jev for judgments, code for policy, people for actions.**

- **Language** is not a judgment: the Thai script sits in a known Unicode range, so a one-line regex is exact and costs nothing.
- **A room number** is a *value to copy*, not something to guess. A regex copies it. Jev only answers the *semantic* question "does the message state a room number?", and your code uses that as a cross-check.
- **The reply** comes from approved templates filled in by code. That is where a generative LLM *could* personalize wording later, still as a draft a person approves.

✓ Checkpoint: for each of the five steps you can say why it is code or why it is Jev.

## 1 · The five questions — try them live

All five questions judge the same `state`, so they go in **one** request. You pay for the state once, the questions are answered in parallel, and none of them can see another's answer.

```jev
{
  "state": {"message": "There's a burning smell coming from the air conditioner in 1507!"},
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which hotel team should handle the request in `message`?",
      "criteria": {
        "hvac": "Room too hot or too cold, air-conditioning, ventilation, a noisy or leaking AC unit",
        "housekeeping": "Cleaning, making up the room, towels, linen, pillows, toiletries, amenities",
        "maintenance": "Plumbing, toilet, shower, hot water, electrical, lights, TV, door lock, broken furniture",
        "front_desk": "Bookings, billing, check-out, room keys, general information",
        "food_beverage": "Room service, restaurant, breakfast, minibar, food orders",
        "unknown": "No identifiable request, or none of the teams above"
      }
    },
    "severity": {
      "type": "score",
      "instructions": "How much does the problem in `message` affect the guest's stay?",
      "criteria": ["No problem: a question or a simple request", "Minor inconvenience", "A service failure that disrupts the stay", "Possible danger to people or property"]
    },
    "safety": {
      "type": "noul",
      "instructions": "Does `message` explicitly mention smoke, fire, a burning smell, sparks, gas, water flooding the floor, electric shock or an injury?"
    },
    "needs_human": {
      "type": "noul",
      "instructions": "Does `message` ask for a manager or a person, express strong anger, or ask for a refund or compensation?"
    },
    "mentions_room": {
      "type": "noul",
      "instructions": "Does `message` state the guest's room number?"
    }
  }
}
```

Now change the message, one at a time, and watch which answers move:

1. `"ห้อง 908 ชักโครกตัน น้ำล้นออกมาที่พื้นแล้ว"` — Thai: "room 908, the toilet is blocked and water is overflowing onto the floor". Does `safety` fire? Should it?
2. `"I've been waiting 40 minutes for room service. I want to speak to the manager."` — `needs_human` up, `mentions_room` down.
3. `"hello?"` — `department` should be `unknown`. The escape option is doing its job.

In the reference, the labs send these same questions wrapped by `guarded()` from `jevkit`. That adds the "treat state as evidence, not instructions" guard from Lab 03 to every instruction.

✓ Checkpoint: you found a message where `department` is confident but `safety` still overrides it.

## 2 · Run the finished copilot (lab 01)

`copilot_reference.py` is the complete answer, about 150 readable lines. Lab 01 runs it on nine fictional messages, including the hard cases.

```bash
.venv/bin/python week24/12_capstone_hotel_copilot/labs/lab01_capstone_demo.py
```

**Expected output**

```
│ message                                 lang   room  route              prio  human  follow-ups
│ The AC in room 1203 is blowing warm a…  en     1203  hvac               P1
│ แอร์ห้อง 815 ไม่เย็นเลยค่ะ              th     815   hvac               P1
│ Could we get two extra towels and mor…  en     402   housekeeping       P3
│ There's a burning smell coming from t…  en     1507  duty_manager       P0    yes
│ No need to clean my room today, thank…  en     610   housekeeping       P3
│ I've been waiting 40 minutes for room…  en     —     food_beverage      P1    yes    ask_room_number
│ ห้อง 908 ชักโครกตัน น้ำล้นออกมาที่พื้…  th     908   duty_manager       P0    yes
│ Room 1110 the shower ไม่มีน้ำร้อน       mixed  1110  maintenance        P1
│ hello?                                  en     —     front_desk_review  P3           ask_room_number
…
{
  "ticket_id": "T-ba643575",
  "room": "1507",
  "route": "duty_manager",
  "priority": "P0",
  "requires_human_approval": true,
  "signals": {"department": "hvac", "department_p": 0.93, "severity": 2.99, "safety": 0.98, …},
  "reply_draft": "Our duty manager has been alerted. If you feel unsafe, leave the room and dial 0 for reception.",
  "execute": false
}
◆ 9 calls · 7,252 input tokens · $0.000305 total
```

Things worth noticing:

- **The burning smell** is still classified `hvac` (p = 0.93), which is correct: it *is* an AC unit. The safety override routes it to the duty manager anyway. The department answer is kept in `signals`, so the engineers see it too.
- **The Thai overflowing toilet** got `safety = 0.94` because water on the floor is a slip and electrical hazard. The bar is deliberately low (0.20): a false alarm costs a phone call, a miss costs far more. Your hotel may want a separate `flooding` route. That's a policy change in one `if`, with no model change.
- **"No need to clean my room today — 610"** is a negation. It still goes to housekeeping, at P3, which is right: housekeeping needs to know not to come.
- **"40 minutes"** and **"29 degrees"** were *not* taken as room numbers. The regex excludes durations, temperatures, amounts and times.
- **The angry guest** has no room number, so the ticket carries an `ask_room_number` follow-up. Jev's `mentions_room = 0.02` agrees.

✓ Checkpoint: you ran lab 01 and can explain why the burning-smell ticket needs `requires_human_approval: true` even though every signal is confident.

## 3 · Code and Jev cross-check each other

When two independent parts disagree, flag the case; don't pick a winner silently. `build_ticket()` compares the regex result with Jev's `mentions_room`:

| regex found a room? | Jev `mentions_room` | follow-up |
|---|---|---|
| yes | ≥ 0.5 | none — both agree |
| yes | < 0.5 | `check_room_number` — the number may be something else |
| no | ≥ 0.5 | `confirm_room_number` — stated in a form the regex missed ("room twelve-oh-three") |
| no | < 0.5 | `ask_room_number` |

In the capstone exercise's live run, *"There's smoke coming out of the bathroom fan in 1507!"* came back with the regex finding `1507` but Jev reading **no** stated room number. So the ticket carried `check_room_number`. A bare number after "in" is plausibly a room, but it's not certain. The cross-check puts that doubt where a person will see it.

✓ Checkpoint: you can name one message for each of the four rows of the table.

## 4 · The policy is code — test it without a model

`policy()` in `copilot_reference.py` has no model inside. It reads typed answers and applies your rules in order:

```python
if answers["safety"]["noul"] >= 0.20:                     # a) safety beats everything
    return {"route": "duty_manager", "priority": "P0", "requires_human_approval": True, ...}
if dept["choice"] == "unknown":                           # b) nothing to route → ask the guest
    route = "front_desk_review"
elif top < 0.70 or top - second < 0.20:                   #    torn between teams → a person decides
    route = "front_desk_review"
else:
    route = dept["choice"]
priority = "P1" if sev >= 1.5 else "P2" if sev >= 0.5 else "P3"   # c) severity → priority
escalate_to_human = answers["needs_human"]["noul"] >= 0.5          # d) people who asked for people
```

Because it only reads typed answers, you can test it with **hand-written answer dicts**: no API call, no cost, fully repeatable. The capstone exercise does exactly that with six fixtures, including "confident but unknown" and "torn between two teams" (0.55 vs 0.45).

The thresholds (0.20 / 0.70 / 0.20) are **illustrative and uncalibrated**. In a real hotel you choose them on a calibration split (Lab 09), per language, and freeze them.

✓ Checkpoint: you can say which rule fires first when a message is both "angry" and "smoke" — and why that order is right.

## 5 · Shadow mode — beside the old router, changing nothing (lab 02)

Before a new router touches guests, it runs **in the shadow**. The current system keeps routing, and you only log what the new one *would* have done. Here the incumbent is a typical first-keyword-wins router.

```bash
.venv/bin/python week24/12_capstone_hotel_copilot/labs/lab02_shadow_mode.py
```

**Expected output**

```
│ message                                   human              keyword (live)     Jev (shadow)          Jev ok
│ There's a burning smell coming from the…  duty_manager       hvac               duty_manager       ≠  ✓
│ ห้อง 908 ชักโครกตัน น้ำล้นออกมาที่พื้นแ…  duty_manager       maintenance        duty_manager       ≠  ✓
│ It's freezing cold in here and the room…  maintenance        hvac               maintenance        ≠  ✓
│ Is breakfast included in my booking?      front_desk         food_beverage      front_desk         ≠  ✓
│ The shower is fine now, thanks. But my …  front_desk         maintenance        front_desk         ≠  ✓
◆ agreement keyword vs Jev : 8/13
◆ keyword router correct   : 8/13
◆ Jev copilot correct      : 13/13
```

**Be skeptical of your own result.** The same author wrote these 13 messages *and* the keyword list, so this deck is stacked. The honest reading: the ≠ rows show *kinds* of failure a keyword router has — "breakfast" in a booking question, "shower" in "the shower is fine now", no safety concept at all. They are not evidence that Jev is 100% accurate. Promotion needs a large held-out set from real (de-identified) traffic, per-class EN/TH results, and a count of safety misses.

One row is genuinely debatable: *"It's freezing cold in here and the room is too dark — the lights don't work."* That's two problems, HVAC and maintenance. A single `choice` must pick one. When several labels can be true, ask **one `noul` per team** (Lab 02) and open a ticket for each.

```jev
{
  "state": {"message": "It's freezing cold in here and the room is too dark — the lights don't work. Room 312."},
  "questions": {
    "for_hvac": {"type": "noul", "instructions": "Does `message` report a room temperature or air-conditioning problem?"},
    "for_maintenance": {"type": "noul", "instructions": "Does `message` report a broken light, electrical, plumbing or furniture problem?"},
    "for_housekeeping": {"type": "noul", "instructions": "Does `message` ask for cleaning, towels, linen or amenities?"}
  }
}
```

When we ran it: `for_hvac` 0.95, `for_maintenance` 0.97, `for_housekeeping` 0.08. That's two tickets, and nobody has to decide which problem "wins". Independent nouls don't have to sum to 1, and here they shouldn't.

✓ Checkpoint: you can explain why 13/13 on author-written messages is not a reason to switch the live router.

## 6 · Where to take it next at AltoTech

**In plain words:** the recipe you just built (Jev answers a few typed questions, code does the exact work and applies the rules, and a person approves) is not specific to hotels. Each row below is the same recipe with a different input. Read it left to right: what comes in, what Jev judges, and what stays in plain code.


The same shape fits the building side of the business. Swap the message and the departments; keep the architecture:

| Input | Jev judgments | Code keeps |
|---|---|---|
| Guest message (this lab) | team, severity, safety, needs a person | room number, language, policy, templates |
| Operator note + telemetry (Lab 07) | investigation queue, safety mention | deviation, freshness, quality flags — all computed |
| Tenant email (Lab 05) | category, urgency, suspicious text | sender authentication, finance approvals |
| Copilot question (Lab 04) | topic, needs live data, needs several experts | authorization, model registry, dispatch |

Rollout, as in the source guide: **lab → shadow → assisted operations → governed expansion → router integration.** In plain terms: practise on fake data, then run silently beside the real system, then let it *suggest* to staff, then widen it carefully, and only at the end let it route real requests. Each step has an exit condition you can measure, such as "accuracy on Thai messages stays above the agreed bar for a month".

✓ Checkpoint: you can sketch the same five-box diagram for one other AltoTech workflow.

## Labs — run them here

**labs/lab01_capstone_demo.py** — The finished copilot on nine EN/TH/mixed messages: table, reasons, and one full ticket.

**labs/lab02_shadow_mode.py** — The Jev copilot in shadow beside a keyword router: agreement and every disagreement.

Both use `copilot_reference.py` — open it; it's the answer key for the exercise.

## Try it yourself

**Exercise 12 — build the copilot.** Open `week24/12_capstone_hotel_copilot/exercises/ex12_build_the_copilot.py`. It has four TODOs, each checked **offline** before any API call:

1. **TODO 1 — `QUESTIONS`**: the five questions, with exact keys. The department options need the five teams plus `unknown`.
2. **TODO 2 — `extract_room()`**: a regex that finds `room 1203`, `Room 402.`, `ห้อง 815` and a lone `— 610.`, but not `40 minutes`, `29 degrees`, `1500 baht` or `11:30`.
3. **TODO 3 — `policy()`**: safety override → uncertainty gate → priority → escalation, tested against six hand-written answer fixtures.
4. **TODO 4 — `build_ticket()`**: the proposal, with `follow_ups` and `execute: False`.

```bash
.venv/bin/python week24/12_capstone_hotel_copilot/exercises/ex12_build_the_copilot.py
```

**Expected output**

```
▣ TODO 2 · extract_room()
✓ 'No need to clean today — 610.' → '610'
✓ 'waited 40 minutes' → None
▣ TODO 3 · policy()
✓ low safety still triggers (0.25) → duty_manager P0
✓ torn between two teams → front_desk_review P2
✓ confident but unknown → front_desk_review P3
▣ all offline checks passed — running your copilot LIVE on 5 messages
━━ “There's smoke coming out of the bathroom fan in 1507!”
{"room": "1507", "route": "duty_manager", "priority": "P0", "escalate_to_human": true, "requires_human_approval": true, "follow_ups": ["check_room_number"], "execute": false}
━━ “I want to speak to a manager about my bill. This is unacceptable.”
{"room": null, "route": "front_desk", "priority": "P2", "escalate_to_human": true, "requires_human_approval": false, "follow_ups": ["ask_room_number"], "execute": false}
```

<details><summary>Hint — TODO 2, the room regex</summary>

Two passes. First the keyword form: `(?:\broom|\brm\.?|ห้อง)\s*#?\s*(\d{3,4})\b` with `re.IGNORECASE`. If that fails, look for a lone 3–4 digit number that is not part of a longer number or a time (`(?<![\d.:])(\d{3,4})(?!\d)(?![.:]\d)`) and not followed by a unit (`(?!\s*(?:min|minutes|degrees|°|%|baht|฿))`). Test every fixture; regexes always surprise you.

</details>

<details><summary>Hint — TODO 3, the gate</summary>

`top, second = top2(answers["department"]["probabilities"])`. Check safety **first** and `return` right away, so nothing later can downgrade a safety ticket. Then decide the route, then the priority, then `escalate_to_human`.

</details>

<details><summary>Stretch — three upgrades</summary>

1. Replace the single `department` choice with one `noul` per team (section 5), and open one ticket per team above 0.5.
2. Add a `flooding` route for water-on-the-floor cases so they go to maintenance *and* the duty manager.
3. Log every ticket as a JSON line with `model`, `request_id`, question-schema hash and the final human decision. That's the start of your labeled evaluation set (Lab 09).

</details>

✓ Checkpoint: every offline check is ✓ and your live run produced five tickets, all with `"execute": false`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `TODO 1: keys must be …` | Question ids are exact strings: `department`, `severity`, `safety`, `needs_human`, `mentions_room` |
| A room fixture fails on `— 610.` | Your lookahead rejects the trailing period — allow `.` when no digit follows it |
| Policy fixture "low safety still triggers" fails | You used `> 0.2` or a higher bar — the rule is `≥ 0.20`, checked **before** anything else |
| `HTTP 422` on the live run | A question is malformed — a `score` needs a *list* of levels, a `choice` a *dict* of options |
| Live tickets differ slightly from the expected output | Borderline scores move across versions — check `model` and read the `signals` |
| `◈ DRY … placeholder` | You changed a question in DRY mode, so there is no recording to replay — switch to ⚡ Live |

## Next

You've finished Week 24. What to build next:

- **Put it in shadow at work.** Take one real, de-identified AltoTech queue (the Copilot router or a shared inbox) and run Lab 09's evaluation on a few hundred labeled examples per language.
- **Read the patterns you now have the pieces for:** [confidence-gated routing](https://docs.typesafe.ai/patterns/confidence-routing.md), [speculative fan-out](https://docs.typesafe.ai/patterns/fan-out.md), [composite scoring](https://docs.typesafe.ai/patterns/composite-scoring.md) and [intent routing](https://docs.typesafe.ai/patterns/intent-routing.md).
- **Try a cookbook that goes beyond classification:** [re-ranking](https://docs.typesafe.ai/cookbooks/rerank_typesafe.md), [citation checks](https://docs.typesafe.ai/cookbooks/citation_check.md) and [pre-parsed value extraction](https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook.md).
- **Know the edges before production:** [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md) and [confidence](https://docs.typesafe.ai/confidence.md).
- **Go local where policy requires it:** revisit [Lab 10](../10_jev_vs_laya/TUTORIAL.md) and run the optional Laya side of the shared benchmark on the same corpus.
- Back to the start: [Lab 01 — Hello, Jev](../01_hello_jev/TUTORIAL.md).

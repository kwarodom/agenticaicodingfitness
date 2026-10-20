# ▶ Jev Lab 02 — The three primitives in depth: choice, noul, score

> Part of Week 24 · Typed AI decisions with Jev. Lab 01 showed you *that* Jev returns typed answers. This lab shows you *how to read them* — and how to pick the right question type for every need.

**What you'll actually do**
- Watch a `choice` split its probability when a message has two needs — and learn to spot it.
- Use `noul` for yes/no propositions, including several labels at once.
- Build a `score` with concrete levels and turn it into a threshold in code.
- Recompute `confidence` yourself and discover where Score confidence differs.
- Decide, for any new need, which of the three primitives fits.

**Time** ~30 min · **Difficulty** beginner · **Cost** ≈ $0.0003 live · $0 dry

## 0 · Recap in 30 seconds

Every Jev request is `state` (the facts) + `questions` (a named map). Every answer has a `type` that matches its question:

| Primitive | Asks | Answer fields | Has `confidence`? |
|---|---|---|---|
| `choice` | Which ONE option fits? | `choice`, `probabilities` (per option, sum ≈ 1) | yes |
| `noul` | Is this proposition true? | `noul` (0 → no, 1 → yes) | no |
| `score` | Where on my ordered scale? | `score` (expected level), `probabilities` (per level), `legend` | yes |

Three rules that hold for all of them:

- **Question ids are yours.** `"department"` is a label for your code — Jev never sees it. Put the whole meaning in `instructions` and `criteria`.
- **Questions are independent.** In one request, each question is answered on its own against the same state. One cannot read another's answer.
- **Typed ≠ true.** A perfectly shaped answer can still be wrong. Your code checks, gates and decides.

✓ Checkpoint: you can say, without looking, which primitive has no `confidence` field (noul).

## 1 · choice — one winner, but read the whole distribution

A Choice picks one option from a set you define (up to 255). You get the winner **and** a probability for every option. The winner alone hides the most useful information.

Lab 01 asks the same department question about three messages:

```bash
.venv/bin/python week24/02_three_primitives/labs/lab01_choice.py
```

**Expected output**

```
▣ STEP 2 · two_needs: “Our room is too hot and we also need fresh towels, please.”
» department             choice  → hvac   (confidence 0.34)
      hvac                       0.51  ████████████░░░░░░░░░░░░ ◀
      housekeeping               0.46  ███████████░░░░░░░░░░░░░
      unknown                    0.02  ░░░░░░░░░░░░░░░░░░░░░░░░
…
▣ STEP 4 · side by side — the margin between the top two options is the tell
│ message     winner   top p  2nd p  margin  confidence
│ clear       hvac     1.00   0.00   1.00    1.00
│ two_needs   hvac     0.51   0.46   0.05    0.34
│ no_request  unknown  0.99   0.01   0.98    0.99
```

Look at `two_needs`: HVAC "wins" with 0.51, but housekeeping has 0.46. That is not a mistake — the message genuinely needs both teams. A program that only reads `choice` would silently drop the towels. **When the margin between the top two options is small, your code should notice.** (Section 2 shows the better tool for "several can be true".)

Try it yourself — edit the message or the options, then **⚡ Ask Jev**:

```jev
{
  "state": {"message": "Our room is too hot and we also need fresh towels, please."},
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which hotel team should handle `message`?",
      "criteria": {
        "hvac": "Heating, cooling, ventilation or air-conditioning problems",
        "housekeeping": "Cleaning, towels, linen, amenities, room tidiness",
        "front_desk": "Bookings, billing, check-in, check-out and general questions",
        "unknown": "The message contains no request for any of these teams"
      }
    }
  }
}
```

Experiments:

1. Add an option `"multiple": "The message needs two or more different teams"`. Does it win?
2. Change the message to `"Thanks, we had a lovely stay!"` — the escape option `unknown` should take almost everything.
3. Delete the `unknown` option and ask about the thank-you message again. Jev must now pick a real team — watch it do so.

✓ Checkpoint: you ran lab 01 and can explain why `hvac 0.51 / housekeeping 0.46` is a signal, not noise.

## 2 · noul — the probability that a statement is true

A Noul (say "noo-ul") returns one number: the probability that your yes/no proposition holds. You can optionally describe what *true* and *false* mean with `criteria`.

The three things beginners get wrong:

| Myth | Truth |
|---|---|
| "0.5 means medium" | 0.5 means **unsure** — the model cannot tell yes from no |
| "Nouls about one message add up to 1" | Each noul is independent. Four labels can all be 0.9 |
| "p(X) + p(not X) = 1" | Not guaranteed. Ask each decision **one way** and enforce logic in code |

```bash
.venv/bin/python week24/02_three_primitives/labs/lab02_noul.py
```

**Expected output**

```
▣ STEP 1 · same question, three messages: clear yes, clear no, unsure
│ The TV remote is not working.                     0.98  yes
│ What time is breakfast?                           0.01  no
│ The room is fine I guess, but it smells a bit d…  0.50  UNSURE → review

▣ STEP 2 · multi-label — a message can need SEVERAL teams: one noul per label
» needs_hvac             noul    0.97  ███████████████████████░  likely YES
» needs_housekeeping     noul    0.87  █████████████████████░░░  likely YES
» needs_billing          noul    0.98  ████████████████████████  likely YES
» needs_security         noul    0.22  █████░░░░░░░░░░░░░░░░░░░  uncertain
◆ sum of the four nouls = 3.04 — independent yes/no questions do NOT sum to 1

▣ STEP 3 · ask a statement AND its negation — are they complementary?
» leaving_early          noul    0.36  █████████░░░░░░░░░░░░░░░  uncertain
» not_leaving_early      noul    0.33  ████████░░░░░░░░░░░░░░░░  uncertain
◆ p(yes) + p(negation) = 0.69 — nothing guarantees exactly 1.00
```

Two real surprises from this run are worth remembering:

- "The room is fine I guess, but it smells a bit different" scored **exactly 0.50**. That is honest uncertainty — route it to a human, do not round it to yes or no.
- For "I might be checking out a day early, not sure yet", the proposition got 0.36 **and its negation got 0.33**. Both are "unsure", and together they sum to 0.69, not 1. If your code needs `p(no) = 1 − p(yes)`, compute it in code from *one* question.

Multi-label, live — one noul per label:

```jev
{
  "state": {"message": "The AC is dripping water onto the carpet and the minibar bill looks wrong."},
  "questions": {
    "needs_hvac": {"type": "noul", "instructions": "Does `message` describe a heating or air-conditioning problem?"},
    "needs_housekeeping": {"type": "noul", "instructions": "Does `message` require cleaning or drying something in the room?"},
    "needs_billing": {"type": "noul", "instructions": "Does `message` raise a question or complaint about a bill or charge?"},
    "needs_security": {"type": "noul", "instructions": "Does `message` report a safety or security threat?"}
  }
}
```

Experiments:

1. Remove the minibar sentence. Which noul drops, and do the others move?
2. Add `"criteria": {"true": "…", "false": "…"}` to `needs_security` saying water near electrics **is** a safety threat. Does the number move?

✓ Checkpoint: you can explain why a multi-label problem needs one noul per label rather than one choice.

## 3 · score — a position on your ordered scale

A Score rates the state against levels you list **lowest first** (2 to 10 levels). You get:

- `score` — the expected level, e.g. `1.6` = "mostly between level 1 and level 2". Levels start at **0**.
- `probabilities` — one per level (`"0"`, `"1"`, …).
- `legend` — your level texts echoed back, so logs are self-explanatory.

Write levels as **concrete situations** a person could recognise ("guest cannot use part of the room"), not vague words ("medium"). And use the number for **thresholds** ("≥ 2.5 → page the manager"), never to reconstruct a magnitude — 1.6 is not "1.6 hours".

```bash
.venv/bin/python week24/02_three_primitives/labs/lab03_score.py
```

**Expected output**

```
▣ STEP 1 · the full answer for one message
» urgency                score   2.00 on 0…3   (confidence 1.00)
      2 Needs action within the hour: guest c…  1.00  ████████████████
◆ Σ level×p = 2.00  vs API score 2.00  (same idea, rounded)

▣ STEP 2 · four messages across the scale
│ message                                           score  top level  conf  → policy (in code)
│ Could you recommend a restaurant for tomorrow n…  0.49   0          0.51  normal queue
│ The bedside lamp bulb is out, whenever you get …  0.11   0          0.89  normal queue
│ There's no hot water and I have a meeting in 40…  2.00   2          1.00  same-hour ticket
│ Water is pouring from the ceiling onto the elec…  3.00   3          1.00  page duty manager
```

Notice the restaurant question: `0.49` with confidence `0.51` — split between level 0 ("no time pressure") and level 1 ("handled today"), because "tomorrow night" is a real time reference. The policy still does the right thing because the **threshold lives in code**.

```jev
{
  "state": {"message": "There's no hot water and I have a meeting in 40 minutes."},
  "questions": {
    "urgency": {
      "type": "score",
      "instructions": "How urgent is the request in `message`?",
      "criteria": [
        "No time pressure: a question or a request for later",
        "Should be handled today, guest is mildly inconvenienced",
        "Needs action within the hour: guest cannot use part of the room",
        "Emergency now: safety risk, flooding, fire, or someone hurt"
      ]
    }
  }
}
```

Experiments:

1. Replace the four levels with `["low", "medium", "high"]`. Does the answer get less certain? Vague levels usually spread probability.
2. Change the message to `"Water is pouring from the ceiling onto the electrical sockets!"` — it should jump to level 3.

✓ Checkpoint: you can compute a score by hand: Σ (level × probability).

## 4 · What `confidence` really measures

`confidence` (choice and score only) summarises how **concentrated** the probabilities are: 1.0 = all on one option, 0.0 = flat. For a **Choice**, TypeSafe's docs use:

```text
confidence = (n · peak − 1) / (n − 1)        n = number of options, peak = highest probability
```

So the same peak means more with more options: 0.70 over 4 options → 0.60; 0.70 over 2 options → 0.40.

For a **Score**, TypeSafe uses a different statistic that it does not publish. Lab 04 measures both:

```bash
.venv/bin/python week24/02_three_primitives/labs/lab04_confidence_math.py
```

**Expected output**

```
│ message                                           type    answer  peak  API conf  peak formula  match    score p(0 1 2)
│ The chiller tripped twice last night.             choice  hvac    0.88  0.84      0.84          ✓ same   —
│ The chiller tripped twice last night.             score   0.88    0.38  0.07      0.07          ✓ same   0.37 0.38 0.25
│ Why did our electricity bill jump after the chi…  choice  energy  0.94  0.92      0.92          ✓ same   —
│ Why did our electricity bill jump after the chi…  score   0.94    0.62  0.44      0.43          ✓ same   0.22 0.62 0.16
│ Can you look into it?                             choice  other   0.96  0.95      0.95          ✓ same   —
│ Can you look into it?                             score   0.52    0.66  0.21      0.49          differs  0.66 0.17 0.17
◆ choice: the peak formula reproduces the API confidence
◆ score : often close, but NOT the same statistic — look at the rows marked 'differs'
```

We found this live while building the course: for "Can you look into it?" the score probabilities `0.66 / 0.17 / 0.17` give a peak formula of 0.49, but the API says **0.21**. The Score statistic appears to care about *order*: probability on a **far** level (level 2, two steps from the peak) costs more confidence than probability on an adjacent level. (Compare the restaurant row in lab 03 above: `0.51 / 0.49 / 0 / 0` — an adjacent split — got confidence 0.51, higher than the peak formula's 0.35.)

The lesson is bigger than the formula: **the probabilities are the ground truth; confidence is a convenient summary.** And neither one is "the chance the answer is right on your data" — only evaluation tells you that (Lab 09).

✓ Checkpoint: you recomputed a Choice confidence by hand and it matched the API.

## 5 · Which primitive? A decision table

| Your need sounds like… | Use | Why |
|---|---|---|
| "Which one of these …?" (exclusive set) | `choice` | One winner + a distribution; add an escape option |
| "Is it true that …?" | `noul` | A single probability, no forced alternatives |
| "Which of these tags apply?" (several can be true) | one `noul` per tag | A choice would force one winner |
| "How much / how severe / how likely-to-buy …?" | `score` | Ordered levels, a threshold in code |
| "How many …?" / "Which date is earlier?" / "What is 28 − 24?" | **code** | Counting, dates and maths are not judgments (Lab 03) |
| "Write a reply" | a generative LLM | Jev does not generate text |

✓ Checkpoint: you can place each row of this table in your own words.

## Labs — run them here

**labs/lab01_choice.py** — One Choice, three messages: clear, torn between two, and none-of-these.

**labs/lab02_noul.py** — Yes/no/unsure, multi-label with one noul per label, and a proposition vs its negation.

**labs/lab03_score.py** — Concrete ordered levels, the expected value by hand, and a threshold policy in code.

**labs/lab04_confidence_math.py** — Recompute confidence yourself; see where Score confidence differs from the Choice formula.

## Try it yourself

**Exercise 02 — pick the primitive.** Open `week24/02_three_primitives/exercises/ex02_pick_the_primitive.py`.

- **Part A (offline, free):** a six-question quiz — for each need write `"choice"`, `"noul"` or `"score"`.
- **Part B:** write three questions for guest messages — `department` (choice with an escape option), `mentions_refund` (noul) and `anger` (score, calmest level first).
- **Part C (automatic):** the checker asks Jev about three test messages.

```bash
.venv/bin/python week24/02_three_primitives/exercises/ex02_pick_the_primitive.py
```

**Expected output**

```
▣ PART A · quiz (offline, $0)
✓ q1_which_language: choice
…
◆ quiz: 6/6
▣ PART B · question shapes (offline, $0)
✓ department is a choice with 4 options incl. an escape option
✓ mentions_refund is a noul
✓ anger is a score with 4 levels
▣ PART C · asking Jev about 3 messages (one batched call each)
━━ “I've been waiting 45 minutes for room service. If it isn't here in 10 I want my money back.”
» department             choice  → room_service   (confidence 1.00)
» mentions_refund        noul    0.99  ████████████████████████  likely YES
» anger                  score   2.93 on 0…3   (confidence 0.93)
```

<details><summary>Hint — the tricky quiz item (q4)</summary>

"A review can praise food, staff AND location" — several tags can be true together. A `choice` would force exactly one winner, so the answer is **one noul per tag**.

</details>

<details><summary>Hint — a refund that is only threatened</summary>

"If it isn't here in 10 I want my money back" is a *conditional* demand. Decide whether that counts, then say so in the noul's `criteria` — e.g. `true: "A refund is requested or demanded, even conditionally"`. Jev reads literally; the criteria are where your intent goes.

</details>

✓ Checkpoint: the quiz says 6/6 and all three shape checks are ✓.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `HTTP 422` on a score | `criteria` must be a **list** (ordered levels), not a dict — and have 2–10 levels |
| `HTTP 422` on a choice | `criteria` must be a **dict** `{option: description}`; use `null` if an option needs no description |
| Score keys are strings | `probabilities` and `legend` use `"0"`, `"1"`… — index with `str(i)` |
| A noul sits near 0.5 | That is the model saying "unsure". Sharpen the proposition, add `criteria`, or send to review |
| Choice confidence looks low but the winner is obvious | Several options may be genuinely acceptable — check the top two before discarding it |

## Next

Continue to [Lab 03 — writing questions Jev answers well](../03_question_design/TUTORIAL.md): literal reading, escape options, fan-out, prompt injection, and why maths stays in code.

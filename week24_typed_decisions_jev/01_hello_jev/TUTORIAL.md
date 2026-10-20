# ▶ Jev Lab 01 — Hello, Jev: your first typed decision

> Part of Week 24 · Typed AI decisions with Jev. You type the commands, you see the real output. Every lab also runs in **DRY** mode (no key, no network, $0) by replaying recorded answers.

**What you'll actually do**
- Understand in one picture what Jev is — and what it is *not*.
- Make your first Jev call three ways: `curl`, plain Python, and the `jevkit` helper.
- Read every field of a Jev answer: `choice`, `probabilities`, `confidence`, `noul`, `score`, `usage`.
- Edit a live request inside this page and watch the probabilities move.

**Time** ~20 min · **Difficulty** beginner · **Cost** ≈ $0.0001 live · $0 dry

## 0 · Before you start

You need three things. The Lab Runner already checked them for you — look at the status pill in the header.

| Need | Check | Already done? |
|---|---|---|
| Python 3.10+ | `.venv/bin/python --version` | The repo `.venv` is 3.13 |
| A TypeSafe API key | `grep -c '^TYPESAFE_API_KEY=' .env` prints `1` | Yes — it is in the repo-root `.env` |
| No `pip install` | the labs use only the standard library | — |

```bash
cd /Users/altodev/Desktop/agenticaicodingfitness
.venv/bin/python --version
grep -c '^TYPESAFE_API_KEY=' .env
```

**Expected output**

```
Python 3.13.13
1
```

> 🔐 The key stays on this machine. The Lab Runner reads it server-side and passes it to your lab scripts — it is never sent to the browser, printed, or written to a results file. Never paste it into code, a notebook or a screenshot.

✓ Checkpoint: the header pill says **LIVE · jev-1.13.0**. (If it says DRY, everything still works — you will see recorded answers instead of fresh ones.)

## 1 · What Jev is, in one picture

Most AI models you have used so far *generate text*: you ask, they write a paragraph. **Jev is different.** It is a *System One* model — it makes a **fast, typed judgment** that your code can use directly, like calling a function.

You send it two things:

- **`state`** — the facts to look at (a message, an email, a JSON record).
- **`questions`** — a named map of typed questions about that state.

It sends back one **typed answer per question**. There are only three question types:

| Type | Plain-English meaning | You get back |
|---|---|---|
| `choice` | "Which ONE of these options fits best?" | the chosen option + a probability for every option + confidence |
| `noul` | "How likely is it that this statement is true?" | one number from 0 (no) to 1 (yes) |
| `score` | "Where does this fall on my ordered scale?" | a number like 1.3 on levels 0…n + probabilities + confidence |

Think of Jev as **a very fast colleague who can only tick boxes you designed** — never write you a letter. That limitation is the point: the answer is always machine-readable, so *your code stays in control*.

```text
your code ──►  state + typed questions  ──►  Jev  ──►  typed answers  ──►  your code decides what to do
               "Room 1203 is too hot"                   intent = hvac (0.97)      route to the HVAC queue
               intent? urgent? how bad?                 urgent = 0.88             page the duty engineer? (policy!)
```

**What Jev will not do** (keep these in code or in another model):

- ✗ write replies, summaries or reports → use a generative LLM *after* Jev routes the request
- ✗ arithmetic, counting, date comparison → compute in Python, give Jev the result
- ✗ fetch data, browse, scrape → another component collects authorized data first
- ✗ grant permissions or take actions → authorization and approvals stay in your code

✓ Checkpoint: you can explain to a colleague the difference between `state` and `questions`, and name the three question types.

## 2 · Your first call with curl

This is the *entire* API: one `POST` with a JSON body. Load the key into your shell (the built-in ⌨ Terminal already has it), then send one question.

```bash
set -a; source <(grep '^TYPESAFE_API_KEY=' .env); set +a
curl -s https://api.typesafe.ai/v1/systemone \
  -H "Authorization: Bearer $TYPESAFE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"jev-1.13.0",
       "state":"Room 1203 is too warm and the AC is blowing warm air.",
       "questions":{"is_hvac":{"type":"noul","instructions":"Is this message about air conditioning or cooling?"}}}'
```

**Expected output**

```
{"model":"jev-1.13.0","answers":{"is_hvac":{"type":"noul","noul":0.99}},"usage":{"input_tokens":292,"output_tokens":22}}
```

Read it left to right:

- `model` — the exact version that answered. Always log it; the alias `jev-latest` can move to a new version later.
- `answers.is_hvac` — comes back under **the key you chose**. The key is just a label for your code; Jev never sees it.
- `noul: 0.99` — Jev is 99% sure the statement is true.
- `usage.input_tokens` — what you pay for: 292 tokens × $0.042 per million ≈ **$0.000012**. Output tokens are free. (The exact count shifts a little with the wording of the request.)

✓ Checkpoint: you got HTTP 200 JSON with `"noul"` close to 1.

## 3 · The same call in plain Python

Lab 01 does exactly the curl call above with nothing but the Python standard library — no helper, no SDK — and prints both the request and the raw response so you can see every byte. Open it with **view source** in the 🧪 Labs section, or run it here:

```bash
.venv/bin/python week24/01_hello_jev/labs/lab01_first_call_raw.py
```

**Expected output**

```
━━ STEP 1 · the request we send (this is ALL of it)
{
  "model": "jev-1.13.0",
  "state": {
    "message": "Room 1203 is too warm and the AC is blowing warm air."
  },
  "questions": {
    "is_hvac": {
      "type": "noul",
      "instructions": "Is `message` about air conditioning or cooling?"
    }
  }
}

━━ STEP 2 · POST https://api.typesafe.ai/v1/systemone
✓ HTTP 200 in 462 ms

━━ STEP 3 · the raw response
{"model": "jev-1.13.0", "answers": {"is_hvac": {"type": "noul", "noul": 0.99}}, "usage": {"input_tokens": 299, "output_tokens": 22}}

═ is_hvac = 0.99 → the model is 99% sure the message is about air conditioning.
◆ you paid for 299 input tokens × $0.042/M = $0.0000126 (output is free)
◆ answered by jev-1.13.0 — log this so you know which version made each decision
```

Notice the backticks in ``Is `message` about…``. When `state` is a JSON object, you point a question at a field by writing its name in backticks — that tells Jev *which part* of the state to judge.

✓ Checkpoint: you ran lab 01 and can point to the three parts of the request: `model`, `state`, `questions`.

## 4 · All three question types in one call

Real workflows ask several questions at once. They are evaluated **independently and in parallel** against the same state — one question cannot see another's answer. You pay for the state once, so batching is cheaper *and* faster than separate calls.

Try it right here. The block below is **live**: edit anything — the message, an option, a level — and press **⚡ Ask Jev**.

```jev
{
  "state": {"message": "The lobby has been freezing since this morning and guests are complaining. Please fix it before the 6pm wedding!"},
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle `message`?",
      "criteria": {
        "hvac": "Heating, cooling, ventilation or air-conditioning problems",
        "housekeeping": "Cleaning, towels, linen, room tidiness",
        "front_desk": "Bookings, billing, check-in and general requests",
        "unknown": "No clear request"
      }
    },
    "has_deadline": {
      "type": "noul",
      "instructions": "Does `message` state an explicit deadline or time limit?"
    },
    "severity": {
      "type": "score",
      "instructions": "How severe is the impact described in `message`?",
      "criteria": ["No impact stated", "Minor inconvenience for one guest", "Many guests or an event affected", "Safety risk"]
    }
  }
}
```

How to read what comes back:

- **choice** → `department` is the winner; the bars show how the probability is shared. If two bars are close, the model is torn — your code should notice that.
- **noul** → `has_deadline` near 1 means "yes, clearly". Near **0.5 means unsure**, not "half a deadline".
- **score** → `severity` like `2.1` means "mostly level 2, a little toward 3". Level numbers start at **0**.
- **confidence** (choice and score only) → how concentrated the probabilities are: 1.0 = all on one option, 0.0 = perfectly flat. It is *not* a promise of correctness.

Now experiment — one change at a time, then ask again:

1. Delete `"before the 6pm wedding"` from the message. What happens to `has_deadline`?
2. Change the message to `"Can I get extra towels?"`. Which department wins, and how confident is it?
3. Remove the `"unknown"` option and send `"hello"`. Where does the probability go now? (Lesson: always give the model a way to say *none of these*.)

✓ Checkpoint: you changed the state at least twice and saw `choice`, `noul` and `score` all respond.

## 5 · Read the answer like a program — lab 02

Lab 02 makes the same three-question call through `jevkit` (the tiny helper all later labs use), prints it with probability bars, then shows how **your code** turns the typed answer into a decision — a routing rule you can read, test and change without touching the model.

```bash
.venv/bin/python week24/01_hello_jev/labs/lab02_read_the_answer.py
```

**Expected output**

```
▣ STEP 1 · ask three typed questions about one guest message
» department             choice  → hvac   (confidence 1.00)
      hvac                       1.00  ████████████████████████ ◀
      unknown                    0.00  ░░░░░░░░░░░░░░░░░░░░░░░░
      housekeeping               0.00  ░░░░░░░░░░░░░░░░░░░░░░░░
      front_desk                 0.00  ░░░░░░░░░░░░░░░░░░░░░░░░
» has_deadline           noul    0.98  ████████████████████████  likely YES
» severity               score   2.00 on 0…3   (confidence 1.00)
      0 No impact stated                        0.00  ░░░░░░░░░░░░░░░░
      1 Minor inconvenience for one guest       0.00  ░░░░░░░░░░░░░░░░
      2 Many guests or an event affected        1.00  ████████████████
      3 Safety risk                             0.00  ░░░░░░░░░░░░░░░░
◆ jev-1.13.0 · LIVE · 478 input tok · $0.000020 · 454 ms
▣ STEP 2 · your code decides — Jev only supplied the judgments
→ route to hvac queue · priority P1 (event affected + explicit deadline)
═ execute: False — this lab never pages anyone; it prints the proposal.
```

`jevkit` is ~350 lines of plain Python in `week24/common/jevkit.py`: `ask()` sends the request, `show()` prints the bars, `choice()/noul()/score()` build questions. Nothing magic — open it.

✓ Checkpoint: you can point to the exact `if` statement in lab 02 that turns Jev's answers into a priority.

## Labs — run them here

**labs/lab01_first_call_raw.py** — Your first call with only the standard library: request, raw response, one number.

**labs/lab02_read_the_answer.py** — All three primitives in one call, pretty-printed, then turned into a routing decision by plain code.

Both run LIVE with the key, or DRY (recorded answers) with the mode switch in the header.

## Try it yourself

**Exercise 01 — your first questions.** Open `week24/01_hello_jev/exercises/ex01_my_first_questions.py` in your editor. It has three `TODO`s:

1. Write a `noul` asking whether a guest message is a **complaint**.
2. Write a `choice` with at least three **room areas** (`bathroom`, `bedroom`, `balcony`, plus an escape option).
3. Write a `score` for **how polite** the message is, with at least three ordered levels.

Save, then press **▶ Run** on the exercise card below (or run the command). The built-in checker validates your questions *before* spending a cent, then asks Jev about three test messages and prints the answers.

```bash
.venv/bin/python week24/01_hello_jev/exercises/ex01_my_first_questions.py
```

**Expected output**

```
✓ is_complaint is a noul with instructions
✓ room_area is a choice with 4 options (includes an escape option)
✓ politeness is a score with 4 ordered levels

▣ asking Jev about 3 test messages …

━━ “What time does breakfast start?”
» is_complaint           noul    0.02  ░░░░░░░░░░░░░░░░░░░░░░░░  likely NO
» room_area              choice  → not_stated   (confidence 1.00)
» politeness             score   2.05 on 0…3   (confidence 0.92)
```

<details><summary>Hint — what makes a good escape option?</summary>

Name it `unknown` or `not_stated` and describe *when* to pick it: "The message does not mention a specific area of the room." Without it, Jev must force every message into a real area — and it will, with confidence.

</details>

<details><summary>Stretch — batch vs separate calls</summary>

Call `ask()` three times with one question each, then once with all three. Compare total `input_tokens`. Why is the batched call cheaper? (Hint: the state is ingested once.)

</details>

✓ Checkpoint: all three checker lines are ✓ and Jev returned an answer for every test message.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `HTTP 401` | The key is missing or wrong. Check `grep -c '^TYPESAFE_API_KEY=' .env` and that you sourced it in *this* shell |
| `HTTP 422` | The JSON shape is wrong — e.g. a `choice` without `criteria`, or a `score` with `criteria` that is not an array. The error body names the field |
| `HTTP 429` / `529` | Rate-limited or overloaded — `jevkit` backs off and retries up to 3 times; wait a minute |
| `◈ DRY … placeholder` | You are in DRY mode and changed the request, so there is no recording. Switch the header to ⚡ Live |
| `curl: command not found` in the terminal | Use lab 01 instead — it is the same call in Python |

## Next

Continue to [Lab 02 — the three primitives in depth](../02_three_primitives/TUTORIAL.md): when to use `choice` vs `noul` vs `score`, and what `confidence` really measures.

# ▶ Jev Lab 11 — Jev + your choice of LLM: Jev decides, your LLM writes

> Part of Week 24 · Typed AI decisions with Jev. Jev never writes prose, so every real product pairs it with a generative LLM. In this lab you choose that LLM (Claude, ChatGPT, Gemini, DeepSeek, Kimi, GLM, OpenRouter or a local Ollama model), wire it behind Jev, compare them on the same task, and let Jev check what the LLM wrote before any person sees it.

**In plain words:** Jev is the fast colleague who ticks boxes ("this is an HVAC problem, urgent, no refund requested"). The LLM is the colleague who writes the email. You will let each do only what it is good at, and keep the rules in your own code.

**What you'll actually do**
- See which LLM providers your machine can use, and pick one for the whole course from the runner's **LLM** picker.
- Build the pipeline: Jev routes → code picks an approved prompt → your LLM drafts a reply.
- Run the same task on every model you have a key for, in English and Thai.
- Make Jev check the LLM's draft (promises a refund? invents a technician?) and escalate to a person.
- Learn the self-identity trap: a model's claim about its own name is not evidence.

**Time** ~45 min · **Difficulty** intermediate · **Cost** ≈ $0.0005 Jev + a few cents of LLM tokens live · $0 dry

## 0 · Why two models?

Jev is a **System One** model: a fast, typed judgment your code can branch on. A generative LLM is a **System Two** model: it writes, explains and reasons in free text. They are good at different things, and they cost different amounts of time.

| Job | Give it to | Why |
|---|---|---|
| "Which team? How urgent? Refund asked?" | **Jev** | typed answer + probabilities, ~0.5 s, a fraction of a cent |
| Extract the room number, detect Thai script | **code** | exact facts: a regex never guesses |
| Choose the system prompt | **code** | only approved prompts, stored in a registry |
| Write the reply to the guest | **your LLM** | fluent prose in the guest's language |
| "Does the draft promise a refund? Invent a name?" | **Jev** | typed checks on the LLM's output |
| Send it? | **a person** | `execute: False`, always |

Measured in lab 02 on this machine: Jev's routing decision averaged **660 ms**; Claude's reply draft averaged **2,765 ms**. The typed decision is the cheap, fast step, and writing is the slow one.

```text
guest message ─► Jev: department · urgency · wants refund?     (typed, ~0.5 s)
              ─► code: language · room number · APPROVED prompt  (exact)
              ─► your LLM: reply draft                          (prose, seconds)
              ─► Jev: checks the DRAFT                           (typed, ~0.5 s)
              ─► code: review policy → staff approval | human review   execute: False
```

✓ Checkpoint: you can say which step of the pipeline is Jev, which is code, and which is the LLM, and why the prompt is picked by code.

## 1 · Meet the providers

Every provider below works with the same `llmkit.generate()` call in `week24/common/llmkit.py`. Claude uses Anthropic's Messages API; the others speak the OpenAI-compatible `/chat/completions` API, so switching is one word.

| id | Provider | Example models (checked live 2026-09-27) | Key variable | Get a key |
|---|---|---|---|---|
| `claude` | Anthropic | `claude-sonnet-5`, `claude-opus-5-5`, `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/settings/keys) |
| `chatgpt` | OpenAI | `gpt-5.4-mini`, `gpt-5.5`, `gpt-5.4-nano` | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/api-keys) |
| `gemini` | Google | `gemini-3.5-flash`, `gemini-3.1-pro-preview` | `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com/apikey) |
| `deepseek` | DeepSeek | `deepseek-flash`, `deepseek-v4-pro` | `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com/api_keys) |
| `kimi` | Moonshot AI | `kimi-k3`, `kimi-k2.6` | `KIMI_API_KEY` | [platform.moonshot.ai](https://platform.moonshot.ai/console/api-keys) |
| `glm` | Z.ai / Zhipu | `glm-5.3`, `glm-5.3-flash` | `ZAI_API_KEY` | [z.ai](https://z.ai/manage-apikey/apikey-list) |
| `openrouter` | OpenRouter | `openai/gpt-5.4-mini`, `anthropic/claude-sonnet-5`, … | `OPENROUTER_API_KEY` | [openrouter.ai](https://openrouter.ai/settings/keys) |
| `ollama` | Ollama (local) | `gemma4:latest`, any model you pulled | none | [ollama.com](https://ollama.com/download) |

Notes that matter in practice:

- **Ollama** runs on your own machine: $0, no key, data stays local. It is also the slowest here (a small laptop GPU).
- **OpenRouter** is one key for hundreds of models, useful for trying many without many accounts.
- **China-region accounts:** set `KIMI_BASE_URL=https://api.moonshot.cn/v1` or `GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4`.
- **Thai quality differs by model.** Measure it on *your* messages (step 4). Don't assume.
- **Thinking models** (DeepSeek, Kimi, GLM-5.x, GPT-5.x, Gemini) spend hidden "reasoning" tokens before the reply. If `max_tokens` is too small, the reply can come back empty or cut off.
- **Prices** change often and differ per provider, so this course does not hardcode them. Check each provider's pricing page, and compare the tokens that the labs print.

Lab 01 prints this table for *your* machine, showing only where each key was found, never the key itself:

```bash
.venv/bin/python week24/11_jev_plus_llm/labs/lab01_pick_your_llm.py
```

**Expected output**

```
▣ STEP 1 · which providers can this machine use? (key SOURCE only — values are never shown)
│    id          provider                default model        key
│ ─  ──────────  ──────────────────────  ───────────────────  ──────────
│ ✓  claude      Claude (Anthropic)      claude-sonnet-5      repo .env
│ ✓  chatgpt     ChatGPT (OpenAI)        gpt-5.4-mini         repo .env
│ ✓  gemini      Gemini (Google)         gemini-3.5-flash     repo .env
│ ✓  deepseek    DeepSeek                deepseek-flash       repo .env
│ ✓  kimi        Kimi (Moonshot AI)      kimi-k3              repo .env
│ ✓  glm         GLM (Z.ai / Zhipu)      glm-5.3              repo .env
│ ✓  openrouter  OpenRouter (any model)  openai/gpt-5.4-mini  repo .env
│ ✓  ollama      Ollama (local, $0)      gemma4:latest        not needed
◆ 8 of 8 providers ready: claude, chatgpt, gemini, deepseek, kimi, glm, openrouter, ollama

▣ STEP 2 · your choice (the runner's LLM picker, or JEV_LLM_PROVIDER / JEV_LLM_MODEL)
→ provider claude · model claude-sonnet-5

▣ STEP 3 · one hello — then compare what the model SAYS with what the API REPORTS
   │ hello from Claude
◆ Claude (Anthropic) · claude-sonnet-5 · LIVE · 22 in / 8 out tok · 1840 ms
```

A fresh clone shows `·` and `missing — set ANTHROPIC_API_KEY` on every row until you add keys (next step). Everything still runs in DRY mode.

✓ Checkpoint: lab 01 shows at least one ✓ provider (Ollama counts if it is running), or you understand why every row says missing.

## 2 · Keys: never in code, and how to choose your model

**Never paste a key into a `.py` file, a notebook or a tutorial.** The labs look it up, in this order:

1. the **environment** of the process (e.g. `export OPENAI_API_KEY=…` in your shell)
2. **`week24/.env.local`**, which the Lab Runner's **🔑 Keys** dialog writes (gitignored, readable only by you)
3. the **repo-root `.env`** (gitignored)

**If you cloned this repo**, pick one way:

- open the runner's **🔑 Keys** dialog, paste a key, and press Save, or
- copy the template and fill it in:

```bash
cp -n week24/.env.example .env   # -n = never overwrite an .env you already have
# then edit .env — set only the keys you have, leave the rest empty
# already have a .env? copy just the lines you need from week24/.env.example into it
```

Both files are in `.gitignore`, so a key can never be committed by accident. You only need `TYPESAFE_API_KEY` for Jev plus **one** LLM key; Ollama needs none.

**Choosing the LLM.** The Lab Runner header has an **LLM** picker (provider + model). Every ▶ Run passes your choice to the lab as `JEV_LLM_PROVIDER` and `JEV_LLM_MODEL`. In a terminal, set them yourself:

```bash
JEV_LLM_PROVIDER=gemini .venv/bin/python week24/11_jev_plus_llm/labs/lab01_pick_your_llm.py
JEV_LLM_PROVIDER=glm JEV_LLM_MODEL=glm-5.3-flash .venv/bin/python week24/11_jev_plus_llm/labs/lab02_route_then_write.py
```

Try the LLM right here. The block below uses the provider picked in the header:

```llm
{
  "system": "You write short replies to hotel guests for staff to review. 2 sentences. Never promise refunds.",
  "user": "Guest in room 1203: the air conditioning is freezing and won't turn off. It's 1am. Write the reply.",
  "max_tokens": 400
}
```

Experiment: switch the LLM picker to another provider and ask again. Then change the user text to Thai (`ห้อง 1203 แอร์เย็นมาก ปิดไม่ได้`) and see whether the reply follows the guest's language.

✓ Checkpoint: you can name the three places a key is looked up, and you ran one lab with a provider other than the default.

## 3 · Jev routes → code picks the approved prompt → your LLM writes

`week24/11_jev_plus_llm/pipeline.py` holds the whole pipeline in about 150 readable lines. First Jev routes the guest message. Try that call here:

```jev
{
  "state": {"message": "Room 1203 is freezing and the AC won't turn off. It's 1am and my kids can't sleep."},
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which hotel team should handle `message`?",
      "criteria": {
        "hvac": "Air conditioning, heating, ventilation, room too hot or too cold",
        "housekeeping": "Cleaning, towels, linen, amenities, room tidiness",
        "maintenance": "Plumbing, electrical, broken furniture, doors, lights, TV",
        "front_desk": "Bookings, billing, check-in/out, general questions",
        "unknown": "No clear request, or none of the above"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is the request in `message`?",
      "criteria": ["Routine: no time pressure stated", "Should be handled today", "Needs attention within the hour", "Possible danger or a total outage right now"]
    },
    "wants_compensation": {
      "type": "noul",
      "instructions": "Does `message` ask for a refund, discount, free night or other compensation?"
    }
  }
}
```

Then **code** (not the LLM) does three things:

1. **Exact facts:** language from Thai script (regex), room number (regex). Nothing is guessed.
2. **Uncertainty gate:** if Jev is unsure of the team (top probability < 0.70, or two teams close), the request goes to the front desk.
3. **Approved prompt:** the system prompt comes from `APPROVED_PROMPTS[team]`. Its house rules forbid promising refunds or inventing names, times or prices. The LLM only ever sees an approved prompt plus a short list of facts.

Lab 02 runs three messages (English, Thai, and an angry refund demand) through the whole path with your chosen LLM:

```bash
.venv/bin/python week24/11_jev_plus_llm/labs/lab02_route_then_write.py
```

**Expected output**

```
▣ STEP 1 · “Room 1203 is freezing and the AC won't turn off. It's 1am and my kids can't sleep.”
» department  hvac         (confidence 1.00)  → team: hvac
» urgency     2.07 on 0…3   · wants compensation 0.03
» code facts  language=en · room=1203
   │ Thank you for letting us know, and we're sorry for the trouble this is causing your family so
   │ late at night. We've asked our HVAC team to check the air conditioning in Room 1203 within the
   │ hour. …
   │ Guest Services
◆ Claude (Anthropic) · claude-sonnet-5 · LIVE · 233 in / 95 out tok · 2219 ms

▣ STEP 2 · “ห้อง 815 ยังไม่มีผ้าเช็ดตัวเลยค่ะ รบกวนเอามาให้หน่อยนะคะ”
» department  housekeeping (confidence 1.00)  → team: housekeeping
» code facts  language=th · room=815
   │ … ทางเราได้แจ้งทีมแม่บ้านให้รีบนำผ้าเช็ดตัวไปให้ภายในวันนี้แล้วค่ะ …

▣ STEP 3 · “I was charged twice for my stay. I want a refund today or I'm leaving a review.”
» department  front_desk   (confidence 1.00)  → team: front_desk
» urgency     1.02 on 0…3   · wants compensation 0.98
→ compensation requested: the draft must NOT promise it — a manager is flagged separately
   │ Thank you for letting us know about the duplicate charge — … Our front desk team has been
   │ alerted and will follow up with you today to look into this. …

◆ Jev decision avg 660 ms · claude draft avg 2765 ms → the typed decision is the cheap, fast step; writing is the slow one
═ execute: False — every draft goes to a staff member for approval; nothing is sent.
```

Look at step 3: the guest demands a refund (`wants_compensation` 0.98), and the draft **acknowledges without promising**. The house rule in the approved prompt did its job. The refund decision belongs to a manager, not to a model.

✓ Checkpoint: you can point to the line in `pipeline.py` where code, not the LLM, chooses the system prompt.

## 4 · Same task, every model you have

Lab 03 routes each message **once** with Jev, then asks every provider that has a key to write the same reply, in parallel so the lab stays fast. It prints latency, output tokens, length, whether a Thai guest got a Thai answer, and whether the draft ended cleanly.

```bash
.venv/bin/python week24/11_jev_plus_llm/labs/lab03_compare_models.py
```

**Expected output**

```
▣ STEP 1 · “Room 1203 is freezing and the AC won't turn off. It's 1am and my kids can't sleep.”
» Jev → team hvac · language en (code) · 520 ms
│ provider    model (API says)         ms     out tok  chars  reply language  complete?
│ claude      claude-sonnet-5          2304   108      279    ✓ English       ✓
│ chatgpt     gpt-5.4-mini-2026-03-17  1622   123      223    ✓ English       ✓
│ gemini      gemini-3.5-flash         2350   76       341    ✓ English       ✓
│ deepseek    deepseek-flash           3865   583      225    ✓ English       ✓
│ kimi        kimi-k3                  18649  562      408    ✓ English       ✓
│ glm         glm-5.3                  2355   79       346    ✓ English       ✓
│ openrouter  openai/gpt-5.4-mini      1616   44       180    ✓ English       ✓
│ ollama      gemma4:latest            21694  594      256    ✓ English       ✓

▣ STEP 2 · “ห้อง 815 ยังไม่มีผ้าเช็ดตัวเลยค่ะ รบกวนเอามาให้หน่อยนะคะ”
» Jev → team housekeeping · language th (code) · 586 ms
│ claude      claude-sonnet-5          4494   173      198    ✓ Thai          ✓
│ chatgpt     gpt-5.4-mini-2026-03-17  1452   102      201    ✓ Thai          ✓
│ gemini      gemini-3.5-flash         1854   58       192    ✓ Thai          ✓
│ kimi        kimi-k3                  12662  490      219    ✓ Thai          ✓
│ …
```

How to read it honestly:

- **These are single observations, not a benchmark.** Run it twice and the numbers move. Kimi ranged 12–23 s across our runs.
- **"out tok" includes hidden thinking** for reasoning models. DeepSeek used 583 tokens for a 225-character reply; Gemini used 76. That changes what you pay, even at the same price per token.
- **"complete?"** is an exact code check (does the draft end like a finished message?). While building this lab, Kimi once spent its whole 700-token budget thinking and returned no reply, and Gemini once stopped mid-sentence ("…check the air"). The lab now uses a roomy budget, and the column catches it when it happens.
- **"model (API says)"** is the version that actually answered: OpenAI returned the dated `gpt-5.4-mini-2026-03-17` for the alias `gpt-5.4-mini`. Log that value.
- All eight answered Thai guests in Thai. Whether the Thai *sounds right* to a Thai guest is for a Thai-speaking reviewer to judge on a few dozen real messages.

✓ Checkpoint: you can name one model you would try first for Thai replies and one you would avoid for a 2-second response budget, based on your own table.

## 5 · Jev checks the LLM's draft

LLMs write fluently, which makes their mistakes easy to miss. A draft can promise a refund nobody approved, or name a technician who does not exist. So after the LLM writes, Jev answers four typed questions about the **draft**, and code adds one exact check:

| Check | Who | Fails when |
|---|---|---|
| `promises_compensation` | Jev noul | ≥ 0.5: offers a refund, discount, free night, upgrade… |
| `invents_facts` | Jev noul | ≥ 0.5: a name, time, price or cause found in neither the message nor the facts |
| `reply_language` | Jev choice | differs from the guest's language (computed in code) |
| `addresses_request` | Jev noul | < 0.5: the draft does not answer what was asked |
| room number present | **code** | `facts["room"]` is set but missing from the draft (exact string check) |

Any failure → **human review**. No failure → **staff approval** (still a person's click). Try the check on a bad draft:

```jev
{
  "state": {
    "guest_message": "Room 1203 is freezing and the AC won't turn off. It's 1am and my kids can't sleep.",
    "facts": {"room": "1203", "team_asked": "hvac", "urgency": "within the hour"},
    "draft": "So sorry about the cold room! We will refund tonight's stay in full, and our technician Somchai will be at your door by 1:15am. Guest Services"
  },
  "questions": {
    "promises_compensation": {"type": "noul", "instructions": "Does `draft` promise or offer a refund, discount, free night, upgrade or other compensation?"},
    "invents_facts": {"type": "noul", "instructions": "Does `draft` state a specific fact — a person's name, a time or deadline, a price, or a cause of the problem — that appears in neither `guest_message` nor `facts`?"},
    "addresses_request": {"type": "noul", "instructions": "Does `draft` respond to what the guest actually asked for in `guest_message`?"}
  }
}
```

Experiment: delete the refund sentence and ask again. Which noul drops? Then remove "Somchai" and "by 1:15am".

Lab 04 runs a real draft from your LLM and the planted bad one through the checks:

```bash
.venv/bin/python week24/11_jev_plus_llm/labs/lab04_jev_checks_the_draft.py
```

**Expected output**

```
▣ STEP 3 · Jev checks A · LLM draft
» promises_compensation  noul    0.03  █░░░░░░░░░░░░░░░░░░░░░░░  likely NO
» invents_facts          noul    0.15  ████░░░░░░░░░░░░░░░░░░░░  likely NO
» reply_language         choice  → en   (confidence 1.00)
» addresses_request      noul    0.90  ██████████████████████░░  likely YES
✓ passes every check → goes to a staff member for one-click approval

▣ STEP 4 · Jev checks B · planted draft
» promises_compensation  noul    0.98  ████████████████████████  likely YES
» invents_facts          noul    0.98  ████████████████████████  likely YES
» addresses_request      noul    0.73  ██████████████████░░░░░░  uncertain
→ HUMAN REVIEW — 3 problem(s):
   ✕ promises compensation — only a manager may offer that
   ✕ states facts that are not in the message or the facts list
   ✕ room 1203 is missing from the draft

═ execute: False — Jev never sends; it only decides whether a person must look first.
```

The room-number check is plain code on purpose: "is `1203` in the text?" has an exact answer, so a model adds nothing. A caution from our own runs: on Claude's **Thai** draft, `invents_facts` came out at **0.64**, because it read "ภายในวันนี้" (within today) as invented, although `urgency: today` was in the facts. That is a false alarm in the safe direction (a person looks), but it shows that checks on Thai text need their own threshold tuning on real Thai drafts.

✓ Checkpoint: you can explain why the room-number check is code while the "invents facts" check is Jev.

## 6 · The self-identity trap

While building this lab we asked every provider to "reply with exactly: hello from <your model name>". Two answers:

- **DeepSeek** (`deepseek-flash`) replied **"hello from ChatGPT"**.
- **GLM** (`glm-5.3`) replied **"hello from GLM-4.6"**.

Neither is lying on purpose. Models learn from text written about other models, and they often don't know their own version. The lesson for production:

- **Log the API's `model` field**, never the model's self-description. Lab 01 prints both side by side.
- The same holds for any claim an LLM makes about itself ("I checked the database", "I am certain"). Verify with code or with a typed Jev check. Don't trust the prose.

✓ Checkpoint: you can say which field you would store in an audit log to prove which model wrote a reply.

## Labs — run them here

**labs/lab01_pick_your_llm.py** — Which providers have a key (source only, never the value), your choice, and one hello: what the model says vs what the API reports.

**labs/lab02_route_then_write.py** — Jev routes three guest messages, code picks the approved prompt, your LLM writes each reply.

**labs/lab03_compare_models.py** — The same routed task on every provider you have, in English and Thai: latency, tokens, language and complete-reply checks.

**labs/lab04_jev_checks_the_draft.py** — Jev checks a real LLM draft and a planted bad draft; the policy sends the bad one to a human.

Choose the writer with the header's **LLM** picker before ▶ Run. Every lab also runs in DRY mode from recorded answers.

## Try it yourself

**Exercise 11 — guard the reply yourself.** Open `week24/11_jev_plus_llm/exercises/ex11_guarded_reply.py`. Two TODOs:

1. `MY_CHECKS`: the four Jev questions about the draft (three nouls, one choice), wrapped in `guarded({...})`.
2. `my_review()`: the policy. One reason per failed rule, the room-number rule in plain code, and `execute` always `False`.

The checker runs five offline cases first (free). When they all pass, your chosen LLM writes two real drafts, the planted bad draft is added, and your checks decide each one.

```bash
.venv/bin/python week24/11_jev_plus_llm/exercises/ex11_guarded_reply.py
```

**Expected output**

```
✓ MY_CHECKS has exactly the four keys
✓ question types are right (three nouls + one choice)
✓ reply_language offers en / th / mixed / other
✓ every question carries the injection guard
✓ policy: clean English draft → staff_approval (0 reason(s))
✓ policy: promises a refund → human_review (1 reason(s))
✓ policy: invented technician + missing room → human_review (2 reason(s))
✓ policy: English reply to a Thai guest → human_review (1 reason(s))
✓ policy: off-topic, everything else fine → human_review (1 reason(s))

▣ offline checks passed — now live: claude (claude-sonnet-5) writes, your checks decide
━━ planted bad draft: So sorry! We will refund tonight's stay, and Somchai will be there by 1:15am.
» compensation 0.98 · invents 0.98 · language en · on-topic 0.63
→ HUMAN REVIEW: promises compensation; invents facts; room 1203 missing
```

<details><summary>Hint — why check the room number in code?</summary>

"Is `1203` in the draft?" is `facts["room"] in draft`: exact, free and never wrong. Asking Jev would add cost and a small chance of error to a question that has no judgment in it. Keep models for meaning; keep code for facts.

</details>

<details><summary>Stretch — a second writer as a fallback</summary>

When the review sends a draft to a human, try once more with a *different* provider (e.g. Claude → Gemini) and re-check. Count how often the second draft passes. Remember that a person still approves every send.

</details>

✓ Checkpoint: all offline checks are ✓ and the planted draft ends in human review.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `HTTP 401 (key missing or invalid)` | The key for that provider is wrong or expired. Re-paste it in 🔑 Keys or `.env`, or pick another provider |
| `HTTP 404 (unknown model id…)` | The model id changed. Pick one from the runner's model list (it asks the provider live) |
| `[the model used its whole token budget thinking …]` | A reasoning model spent `max_tokens` on hidden thinking. Raise `max_tokens` or choose a `-flash`/`-mini` model |
| Reply ends mid-sentence / `complete? ⚠` | Same cause, so give it more tokens |
| Ollama: `network/timeout` | Start it with `ollama serve` and pull a model (`ollama pull gemma4`), or set `OLLAMA_BASE_URL` |
| Kimi / GLM `401` from China | Use `KIMI_BASE_URL=https://api.moonshot.cn/v1` or `GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4` |
| `◈ no recording (DRY)` for a provider | DRY replays only what was recorded. Switch to ⚡ Live with that provider's key |
| Thai reply comes back in English | Check the house rule `reply in {language}` reached the prompt, then try another model, since Thai ability varies |

## Next

Continue to [Lab 12 — Capstone: a smart-hotel guest-request copilot](../12_capstone_hotel_copilot/TUTORIAL.md). You'll put everything together, and the reply-draft step you built here is exactly where your chosen LLM plugs in.

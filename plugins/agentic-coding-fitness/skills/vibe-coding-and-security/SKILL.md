---
name: vibe-coding-and-security
description: "Teach how to work WITH a coding agent without shipping garbage — context engineering (CLAUDE.md as the new README, the six-layer framework, Preferred/Avoid pairs), test-driven vibe coding (spec → test → implement so 'done' is verifiable), the ~45% AI-code vulnerability problem plus a self-reflection security-review loop, and a hand-off-vs-take-control decision table (partner not boss). Use when someone asks 'how do I write a good CLAUDE.md?', 'why is my AI-generated code buggy/insecure?', 'should I let the agent do this or do it myself?', mentions vibe coding / context engineering / prompt-as-source / agentic engineering / self-review security, or is reviewing Week 6."
when_to_use: "Learner is coding alongside an AI agent and wants to do it safely and fast — writing a CLAUDE.md/context file, spec-then-test-then-implement, security-reviewing AI-generated diffs, deciding what to delegate vs keep human, or catching up on Week 6."
---

# Vibe Coding & Security — Strategy is Human, Execution is AI (Week 6)

> **The one idea:** Vibe coding is fast because the *human* does **strategy** (the spec, the tests, the architecture) and the *AI* does **execution** (typing the code). Skip the strategy and you get speed with no steering — and roughly **45% of AI-generated code ships with a vulnerability**. The win isn't "let the AI cook." It's *partner, not boss*: precise context in, every diff reviewed out.

```
HUMAN: what + why (spec, tests, boundaries)  →  AI: how (implementation)  →  HUMAN: review + verify
```

---

## Part A — Context engineering is the job

> **"An AI agent is only as smart as the last time its context was reviewed."** Raw prompting skill matters less than feeding the right context at the right time. `CLAUDE.md` is the **new README**: it's the file the agent actually reads before it touches your code.

The course research frames context as **six layers**, each one inheriting from the one above:

| Layer | What lives here | Where it comes from |
|---|---|---|
| 1. **System instructions** | role, behavioral rules, output format, tool policy | your prompt / system message |
| 2. **Long-term memory** | architectural decisions, recurring constraints, preferences | `CLAUDE.md`, `agent-memory-graphs` |
| 3. **Retrieved docs (RAG)** | relevant chunks for *this* query | `rag-knowledge-agents` |
| 4. **Tool definitions** | names, params, usage examples | `tool-use`, `mcp-and-skills` |
| 5. **Conversation history** | last ~5–10 turns; older turns summarized | the loop, `agent-loops` |
| 6. **Current task** | the immediate request | placed **last** for recency |

`CLAUDE.md` is your handle on **Layer 2**. The repo's `week06_fullstack_agent_app/CLAUDE.md` is a clean, minimal example of the shape:

```markdown
## Project Overview      ## Code Conventions          ## Important Context
- Name, Tech Stack       - functional components       - JWT auth
- Database, API Base     - async/await over promises   - migrations on startup
                         - Write tests for all          - Stripe for payments
                           API endpoints

## File Structure        ## Common Commands
/src/components ...       - npm run dev / test / build / migrate
```

H2 sections, bullet rules, the commands the agent needs to run itself. A focused **~400-token file beats a sprawling 4,000-token one** — precision beats volume when you're directing attention.

> 📁 Class repo: `week06_fullstack_agent_app/CLAUDE.md` — the canonical minimal context file (Overview / Conventions / Context / File Structure / Commands).

### Nested layering — the real power move

Don't cram everything into one root file. **Layer it down the tree**, each level deepening specifics:

```
/CLAUDE.md                  → project overview, global conventions
/backend/CLAUDE.md          → backend stack, patterns, anti-patterns
/frontend/CLAUDE.md         → component conventions, state management
/infrastructure/CLAUDE.md   → deployment, environment, tooling
```

Claude Code reads `CLAUDE.md`/`AGENTS.md` with nested directory scoping, so a file in `/backend` only loads when work happens there. (For where this connects to skills and MCP servers, see the `CLAUDE.md` section in `mcp-and-skills`.)

### A real-repo gotcha: copied context drifts

> ⚠️ **Teachable failure (real in this repo):** `AgentMemoryDemo/CLAUDE.md` is a **DJI Tello drone dashboard** project — but its body still says JWT auth, *Stripe for payments*, and "Redis cache on port 6379," because it was copy-pasted from `week06_fullstack_agent_app/CLAUDE.md` (an e-commerce dashboard) and never re-grounded. A drifted context file is *worse than none*: it confidently lies to the agent. **Audit your `CLAUDE.md` whenever the project changes.**

> 📁 Class repo: `AgentMemoryDemo/CLAUDE.md` — same template as Week 6, **stale** for its actual project. Use it to teach "context drift."

### Teach the model with Preferred/Avoid pairs

The single most effective thing you can put in a context file is a **code pair** — the model learns your house style faster from a contrast than from prose:

```markdown
## Auth patterns
Preferred:
    user = db.query("SELECT * FROM users WHERE id = ?", [user_id])   # parameterized
Avoid:
    user = db.query(f"SELECT * FROM users WHERE id = {user_id}")      # SQL injection
```

Name sections in the natural language of the task — `## Auth patterns`, not `## Security considerations (misc.)`. Each rule should stand on its own.

---

## Part B — Test-driven vibe coding (Spec → Test → Implement)

> Pure vibe coding's core vulnerability is **deploying code you don't understand**. TDVC inverts the order so "done" is *verifiable* before any implementation exists: the **human writes the spec and the tests; the AI writes the code to pass them.**

```
1. HUMAN  writes the spec in plain language (inputs, outputs, edge cases)
2. AI     generates test cases from the spec
3. HUMAN  reviews & approves the tests        ← the contract is now locked
4. AI     writes implementation to pass them
5. HUMAN  verifies & iterates
```

Why this works: **AI excels at following detailed instructions but is poor at inferring intent from vague prompts.** A failing test is an unambiguous instruction. Writing the test first also:

- **catches hallucinated functions immediately** — a call to a non-existent function fails the test, loudly;
- **defines edge cases** the AI would otherwise skip;
- **freezes the interface** so a refactor can't silently change behavior;
- **nudges toward small, focused functions** because testable code is smaller code.

The trade-off is honest: TDVC is *slower to first output* than "Accept All," but it pays back in less debugging and less review burden. This is the same Reflection/Evaluator-Optimizer loop you'll meet in `models-and-patterns` — only here the *test suite* is the evaluator, and it never gets tired or flatters the code.

---

## Part C — The 45% problem + the self-reflection security loop

> ⚠️ **Conceptual / from the blueprint:** the security statistics in this section come from the course research (Second Talent, Databricks, Escape.tech), **not** measured in this repo. The `CLAUDE.md` examples in Part A *are* real repo files. Treat the numbers as motivation, not as data we collected.

**The headline:** ~**45% of AI-generated code fails basic security testing**, and AI code carries materially more privilege-escalation paths, secrets exposure, and design flaws than human-written code. AI is *worst* exactly where it matters most:

| AI does this **poorly** (keep human) | AI does this **acceptably** (safe to delegate) |
|---|---|
| authentication flows | routine boilerplate |
| authorization logic | test generation |
| cryptography | non-security refactoring |
| input validation | UI component creation |
| secrets management | documentation |

### The common AI-code vulns to grep for

| Vuln | Looks like | Fix |
|---|---|---|
| **Injection** (SQL/command) | f-string / concatenated query or shell | parameterize / use an allowlist |
| **Secrets in code** | `API_KEY = "sk-..."`, passwords in source | env vars / a secrets manager |
| **Missing authz** | endpoint with no "is this user allowed?" check | explicit ownership/role check |
| **Unsafe eval** | `eval()` / `exec()` / `pickle.loads` on input | parse explicitly; never `eval` user data |
| **Weak crypto** | `md5`, hand-rolled tokens | vetted libs, `secrets` module |

### The mitigation: make the agent review its own diff

The most effective single tactic from the research is **self-reflection prompting** — ~**48–50% vulnerability reduction**. After the AI writes code, you make it security-review *its own diff* against named vulnerability classes:

```
Review the diff you just wrote for security issues. Check specifically for:
SQL/command injection, XSS, insecure deserialization, hardcoded secrets,
missing input validation, missing authorization checks, and weak cryptography.
For each issue: show the line, fix it, and explain what changed in one sentence.
```

Two rules that make it actually work:

- **Be language-specific.** Naming the concrete classes (above) cuts vulns **24–37%**; a vague "make it secure" only manages **8–16%** and isn't worth typing.
- **Pair it with a gate, don't trust it alone.** Self-review is a filter, not a guarantee. Back it with quality gates — **>80% branch coverage, zero critical vulnerabilities, SAST on every commit** — and *mandatory human review* for anything touching auth, authz, crypto, or input validation, no matter how confident the agent sounds. (The CI-gate machinery lives in `agent-evaluation`.)

---

## Part D — Hand off vs take control (partner, not boss)

> The most productive developers abandoned both extremes — neither "Accept All" nor "type every line." Treat the AI like an **eager junior developer**: precise instructions, review of every change, clear feedback. Strategy stays human; execution delegates.

### The delegation table

| ✅ Hand off to AI (well-bounded, clear acceptance criteria) | 🛑 Keep human (consequences compound across the system) |
|---|---|
| CRUD operations | authentication / authorization |
| boilerplate | data-model design |
| test scaffolding | API contracts |
| UI scaffolding | anything touching **money** |
| documentation | deploy / infra config |
| refactors **with tests** | anything **destructive / irreversible** |

The rule of thumb: hand off where AI reliably gets to ~70% complete and the last 30% is mechanical; take control where getting it wrong is silent, expensive, or unrecoverable. AI-generated code carries notably **more major issues at the boundaries where systems interact** — which is exactly the right column.

### The delegation ladder (the workflow)

1. Write a **detailed spec** — human owns *what* and *why*, AI figures out *how*.
2. Set **explicit acceptance criteria** before any code is generated.
3. **Review the plan** before it implements, not after.
4. **Run the tests** before accepting any generated code (Part B).
5. **Commit AI changes separately** with clear attribution → easy rollback.

A planning artifact makes step 1–2 concrete. In this repo, `GSD_vs_PlanMD_Analysis.md` compares spec-driven systems (GSD, Plan.md, OpenSpec, Taskmaster) on autonomy, context management, and cost control — a real example of *deciding how you'll hand work to agents* before you hand it. The matching hand-off decision tree (rung-by-rung "is one model call enough? does it need tools?") lives in `models-and-patterns`.

> 📁 Class repo: `GSD_vs_PlanMD_Analysis.md` — a real comparison of spec-driven / context-engineering planning systems for an agentic team.

---

## The one-screen recap

```
CONTEXT  →  CLAUDE.md is the new README; layer it, keep it true, use Preferred/Avoid pairs
SPEC     →  write the test (or success criteria) BEFORE the agent writes code
SECURITY →  self-review the diff against named vuln classes + back it with a gate
DELEGATE →  CRUD/tests/UI/boilerplate to AI; auth/data/money/deploy stay human
```

Frameworks and models churn; *strategy, context, and review* are what you keep.

---

## 🧪 Guided lab (offer this)

### Warm-up (5–10 min, pass/fail)
Write **one `Preferred:` / `Avoid:` code pair** for a real rule in a project the learner cares about (e.g. parameterized queries, env-var secrets, a naming convention). Drop it into a `## <task-named> patterns` section of a `CLAUDE.md`.
**Pass =** the pair is a genuine contrast (the `Avoid` line is something they've actually seen), the section title names the *task* (`## Auth patterns`, not `## Misc`), and it stands on its own without surrounding prose.

### Skill drill (15–30 min, runnable, $0 — no API key)
Do one full **Spec → Test → Implement → Self-review** loop on a tiny function, using a `MockLLM` stub so it runs offline.

```python
# tdvc_drill.py — runs with: python tdvc_drill.py   (no API key, $0)

# ---- A MockLLM that "writes" two candidate implementations on demand ----
class MockLLM:
    """Stands in for a coding agent. Returns canned code so the lab is free & offline."""
    INSECURE = (
        "def find_user(db, user_id):\n"
        "    return db.query(f\"SELECT * FROM users WHERE id = {user_id}\")\n"  # injection!
    )
    SECURE = (
        "def find_user(db, user_id):\n"
        "    if not isinstance(user_id, int):\n"
        "        raise ValueError('user_id must be an int')\n"
        "    return db.query('SELECT * FROM users WHERE id = ?', [user_id])\n"
    )
    def implement(self, spec):   return self.INSECURE   # AI's first cut
    def self_review(self, code): return self.SECURE     # after the security prompt

# ---- STEP 1+2: the human's spec, expressed as a failing TEST (the contract) ----
class FakeDB:
    def __init__(self): self.last_sql, self.last_params = None, None
    def query(self, sql, params=None):
        self.last_sql, self.last_params = sql, params
        return [{"id": params[0] if params else None}]

def run_tests(find_user):
    db = FakeDB()
    find_user(db, 7)
    assert db.last_params == [7], "must pass user_id as a bound PARAMETER, not inline"
    assert "{" not in db.last_sql and "7" not in db.last_sql, "no value interpolated into SQL"
    try:
        find_user(db, "7 OR 1=1"); raise AssertionError("must reject non-int user_id")
    except ValueError:
        pass
    return True

# ---- STEP 3+4: AI implements; tests gate it; security self-review fixes it ----
llm = MockLLM()
for label, code in [("first cut", llm.implement("find_user(db, user_id)")),
                    ("after self-review", llm.self_review(llm.implement("...")))]:
    ns = {}
    exec(code, ns)                      # load the candidate implementation
    try:
        run_tests(ns["find_user"]); print(f"[{label}] PASS ✅")
    except AssertionError as e:
        print(f"[{label}] FAIL ❌ — {e}")
```

Expected output: the **first cut FAILS** (string-interpolated SQL), the **after-self-review version PASSES**. The learner *sees* the security prompt earn its keep.

**Weighted evaluation criteria (5):**
1. **(weight 2 — required) The test is written BEFORE looking at any implementation** and encodes the spec (parameterized query, rejects bad input). A test written to fit existing code fails this criterion.
2. The first/insecure implementation **fails** the suite and the learner can name *why* (injection via f-string).
3. The self-review step is invoked with a **language-specific** vuln list (named classes), not "make it secure."
4. The secure version **passes** and the diff is explained in one sentence ("parameterized the query + validated the type").
5. The learner places this function in the **right column** of the Part D table (input validation → keep-human-reviewed) and says so.

**Pass = 4/5 criteria, and criterion #1 must be one of them** (test-first is the whole point — clearing the others while writing the test last is a fail).

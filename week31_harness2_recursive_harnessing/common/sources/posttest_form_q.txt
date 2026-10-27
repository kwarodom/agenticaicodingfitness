# Harness² Lesson — Post-Test (Form Q)

**Course:** Agentic AI Coding Fitness · **Lesson:** Harness²: agents that improve their own harness · **Time:** 15 minutes · **Items:** 20 multiple choice, one best answer each · **Score:** ___ / 20

**Name:** ______________________ **Team:** ____________ **Date:** ____________

**Instructions.** Answer every question; there is no penalty for guessing. Choose the single best answer and mark it on the answer sheet at the end. Questions 1 to 17 are about the method and the published evidence; questions 18 to 20 are about the always-on operations agent design presented in class, which is an example design, not a deployed system. No notes, slides or internet.

---

**1.** Which statement matches the paper's framing "Agent = Model + Harness"?

- A. The harness is the benchmark container the agent runs in, with its timeouts and network rules
- B. The harness is the anonymised selector that picks the winning candidate at k > 1
- C. The harness is the set of editable components composed with a frozen model; the model's weights never change
- D. The harness is the data used to fine-tune the model for the benchmark domain

**2.** Which list names the eight Harness² components?

- A. System prompt, guardrails, memory, tool guidance, subagents, skills, scripts, plugins
- B. Model, tokenizer, prompt, retriever, cache, judge, scheduler, logger
- C. Prompt, temperature, top-p, max tokens, stop words, seed, tools, output format
- D. Planner, executor, critic, verifier, memory, retriever, router, interface

**3.** Which components are paid for on every model call because they enter the context on every call?

- A. Skills and scripts, because they are the largest files in the tree
- B. System prompt, guardrails, memory and tool guidance
- C. Subagents and plugins, because they run on every tool call
- D. Only the system prompt; everything else loads on demand

**4.** Which component is the only one that enforces behaviour rather than instructing the model?

- A. Guardrails (guardrails.md), because they are written as rules
- B. Tool guidance (tool_guidance.md), because it is read before every tool call
- C. Skills (SKILL.md), because they contain step-by-step procedures
- D. Plugins (hooks around tool calls, e.g. plugin/<name>.ts)

**5.** Which sequence describes one Propose–Probe–Compose step?

- A. Reference run with the current harness → two scoped editors propose candidates → both probed on the same task → compose (merge, revise, adopt one or keep base) → select if k > 1 → reflect
- B. Compose a harness from the library → probe it on the task → propose edits to fix what failed → select the best candidate → run the reference last to confirm the gain
- C. Select the best past harness from the library → reflect on it → propose one edit → run the task once and adopt
- D. Probe the task with the base harness → write a rubric from the result → compose a harness that satisfies it → judge

**6.** The composer concludes KEEP_BASE. According to the editor prompt in prompts.py, this is:

- A. A failure state that automatically triggers k more probes until an edit is found
- B. A signal that the run should be discarded and the task retried with a new seed
- C. "A real answer, not a failure: adaptation the evidence does not support costs context and buys nothing"
- D. Allowed only when the selector agrees and the library already holds an entry for the task

**7.** Which of these edits is an invariant-scope edit (process level) rather than a domain-scope edit?

- A. A skill fixing the aggregation grain to CHANNEL_NAME by QUARTER
- B. A tool-guidance line: do all work through native tool calls, never as text
- C. A redline skill that makes Word markup survive as text
- D. A reconciliation skill for schedule, roster and budget figures

**8.** The reflector records a verdict for every edit row. What is the vocabulary, and which verdict should be preferred when a candidate changed several components at once?

- A. pass, fail, skip; prefer skip when the outcome is unclear
- B. accept, reject, defer; prefer defer when several components changed
- C. green, amber, red; prefer amber when the evidence is mixed
- D. helped, hurt, no observed effect, confounded; prefer confounded

**9.** In an EXPERIENCE.md entry, what should the cue describe?

- A. The shape of situation the lesson applies to, so it can fire on a different matter, not this task's facts
- B. The file names, column mappings and thresholds of this task, so they can be reused directly
- C. The score the harness achieved on the task, so later tasks can pick the best
- D. The model that served as executor, so the lesson is applied only to that model

**10.** On WorkBuddy-Bench with the OpenCode harness at k = 1, Gemini 3.5 Flash went from 72.4 to 77.5 and GLM-5 from 64.2 to 75.7. Which reading is correct?

- A. Gemini 3.5 Flash gained more points than GLM-5 because it is the stronger executor
- B. Both executors gained about the same, as expected from a shared method
- C. The weaker executor gained most (+11.5 vs +5.1); GLM-5 refined (75.7) beats Gemini 3.5 Flash base (72.4)
- D. GLM-5 with its base harness already beat Gemini 3.5 Flash with its refined harness

**11.** Standard deviations across the three task-order streams are typically 1 to 3 points. How should a reported 1.5-point difference on a single benchmark be treated?

- A. As a confirmed gain, because the method is expected to improve every benchmark
- B. As proof that the method failed on that benchmark
- C. By multiplying it by k to recover the true effect
- D. As within noise unless it is consistent across streams and configurations

**12.** In Appendix E.1 (k = 5), removing components from the improved harness cost: skills −5.3, scripts −3.7, prompt-related text −1.5. What does this imply for an edge deployment with slow prefill?

- A. Move all guidance into the system prompt so it is always available
- B. Most gain sits in on-demand components, so quality can rise without growing the always-on prefix or breaking the cache
- C. Remove skills to save tokens, since they are the largest component
- D. Prompt text carries most of the gain, so a larger always-on prefix and slower prefill are unavoidable

**13.** Figure 12 (cross-model transfer): a harness refined with Gemini 3.5 Flash gave GLM-5 +17.3 on WB-code but Gemini 3.7 Flash only +0.3. What is the design conclusion for an agent that runs on more than one model?

- A. Keep one library per executor model and treat transfer as an experiment arm to be measured
- B. Share a single library across all models, since the harness is model-independent text
- C. Never reuse a harness across models; always start from the empty tree
- D. Transfer only the prompt text, because skills and scripts are model-specific

**14.** In Harness², when does rubric-based judging run relative to selection?

- A. Before selection, so the selector can use the scores to pick the winner
- B. During selection, as a tie-breaker between candidates the selector cannot separate
- C. Never; the pipeline has no judging step at all
- D. Only after selection has committed; the selector sees anonymised deliverables, never rubrics or scores

**15.** Figure 14: on a WB-code task the selector chose a candidate that scored 55 over one that scored 95. What is the paper's cause and remedy?

- A. An edit that capped the pool; the remedy is to generate more candidates per step from the same editors
- B. Attribution noise; the remedy is to retrain the executor on the failed task
- C. Blind selection: the selector preferred a plainer deliverable; remedy is selection balancing coverage and precision
- D. A grading-data leak; the remedy is to block the benchmark hosts

**16.** A "literal-key" rule written into a shared harness made all four candidates score 42. Which failure class is this, and what is the remedy?

- A. Blind selection; the remedy is pairwise human review of the deliverables
- B. An edit that capped the pool: a wrong premise inherited by every candidate; probes that challenge shared assumptions
- C. Attribution noise; the remedy is repeated executions to separate the edit from variance
- D. A measurement leak; the remedy is the spec.py measurement-word ban

**17.** Which option most completely describes the measurement-hygiene controls used in the Harness² release?

- A. spec.py bans measurement words in task prompts; library.py blocks grader verdicts from stored trees; EXPERIMENTS.md advises blocking huggingface.co, hf.co and github.com for the executor
- B. A longer system prompt telling the agent not to look for grading data, rubrics or benchmark repositories online, repeated in every candidate harness
- C. Fixed random seeds for every rollout and a frozen task order, so that any leaked grading data cannot change the measured outcome
- D. Running only at k = 1 with a single editor, so there is no selector that could be influenced by leaked scores or rubrics

**18.** In the always-on agent design from class, which items sit outside the editable harness surface?

- A. Skills and scripts, which only a named human may edit after review
- B. EXPERIENCE.md and the library index, which are rebuilt nightly from the edit rows and verdicts
- C. Propose-only mode, named-human approval and the legal gate, enforced in code with a validator refusing edits
- D. Tool guidance and memory, which are frozen at each deployment boundary

**19.** How does the always-on agent design from class use staff approvals and rejections?

- A. As numeric scores inside the anonymised selector so that human preference drives selection
- B. To fine-tune the local model on approved proposals each week
- C. They are ignored because signals that are not scores cannot inform the loop
- D. To choose which tasks to harness (weak-signal wakes) and to gate promotion, never inside the selector

**20.** Which statement about the pre-registered evaluation design is correct?

- A. pass@3 is used because one success in three trials is enough for a notification agent that staff will check anyway
- B. pass^3 is used where consistency matters (75 percent per trial gives about 42 percent over three trials); thresholds are fixed before the first run
- C. Thresholds are adjusted after the first run so that they match the observed results and avoid false alarms
- D. Safety graders may be skipped in the Shadow arm because nothing is actuated there and no staff see the output

---

## Answer sheet (Form Q)

| Q | A | B | C | D | | Q | A | B | C | D |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ☐ | ☐ | ☐ | ☐ | | 11 | ☐ | ☐ | ☐ | ☐ |
| 2 | ☐ | ☐ | ☐ | ☐ | | 12 | ☐ | ☐ | ☐ | ☐ |
| 3 | ☐ | ☐ | ☐ | ☐ | | 13 | ☐ | ☐ | ☐ | ☐ |
| 4 | ☐ | ☐ | ☐ | ☐ | | 14 | ☐ | ☐ | ☐ | ☐ |
| 5 | ☐ | ☐ | ☐ | ☐ | | 15 | ☐ | ☐ | ☐ | ☐ |
| 6 | ☐ | ☐ | ☐ | ☐ | | 16 | ☐ | ☐ | ☐ | ☐ |
| 7 | ☐ | ☐ | ☐ | ☐ | | 17 | ☐ | ☐ | ☐ | ☐ |
| 8 | ☐ | ☐ | ☐ | ☐ | | 18 | ☐ | ☐ | ☐ | ☐ |
| 9 | ☐ | ☐ | ☐ | ☐ | | 19 | ☐ | ☐ | ☐ | ☐ |
| 10 | ☐ | ☐ | ☐ | ☐ | | 20 | ☐ | ☐ | ☐ | ☐ |

Based on the [Harness² paper, arXiv 2610.14224](https://arxiv.org/html/2610.14224), the [google-research/harness2 repository](https://github.com/google-research/harness2) and, for questions 18 to 20, the example agent design presented in class. Solutions are in the separate file "Harness² Quiz Solutions".

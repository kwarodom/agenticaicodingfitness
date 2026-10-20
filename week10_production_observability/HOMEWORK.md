# Week 10 Homework

Deadline: **before Week 11 session**.

## Required — pick 3 of 4

### 1. Persistence upgrade
Swap `SqliteSaver` for `PostgresSaver` or a Mongo checkpointer in Ex 3. Commit a working branch.

### 2. Escalation node
Already implemented in `solutions/02_supervisor_solution.py`. Re-implement yourself and add a unit test that proves cancellation tickets route to `escalation_specialist`.

### 3. Parallel fan-out
In Ex 2, call all 3 specialists in parallel and let the supervisor pick the best reply. Use LangGraph's `Send` API.

### 4. Graph visualization
For any of your graphs, render the mermaid with `app.get_graph().draw_mermaid_png()` and add the PNG to your repo.

## Stretch — pick 1, optional

- Port Ex 5 to actual Claude Managed Agents (public beta, no waitlist).
- Build a LangSmith eval dataset of 50 labeled tickets and measure classify accuracy across Gemini 2.5 Flash-Lite, GLM-5.1, Qwen3 Coder 480B (free), DeepSeek R1 (free), and Claude Sonnet 4.6. Report accuracy, latency, and tokens per ticket.

## Submit

Push to your fork, open a PR titled `[W10-HW] <your-name>`.

# 02 · Tool Use (Week 3)

> Skill: `agentic-coding-fitness:tool-use`
> **One idea:** the model can't run code — it *asks you* to via a `tool_use` block. You run it and return a `tool_result`.

Run in order:

```bash
python 01_calculator_only.py        # one tool; watch 🔧 fire
python 02_calculator_and_weather.py # two tools; one question needs both
python 03_add_your_own_tool.py      # adds roll_dice — then YOU add a 4th
```

- `simple_math.py` is a **safe** calculator (AST-based, no raw `eval`) — the production fix the skill mentions.
- `get_weather` uses the free Open-Meteo API (no key needed).

### Things to try
- In `03_...`, add a 4th tool (`get_time`, `reverse_text`, `read_file`) by copying the **schema → implementation → register** pattern.
- Break mistake #2 on purpose: replace `call.id` with `"wrong-id"` in the `tool_result` and rerun → `tool_use_id` mismatch. Fix it. Now you'll never forget.

**Carry forward:** wrap this loop with a goal + a `max_iterations` cap and you get an *agent* → folder 03.

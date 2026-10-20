# 01 · LLM Fundamentals (Week 2)

> Skill: `agentic-coding-fitness:llm-fundamentals`
> **One idea:** an LLM call is just *text in → text out*. Everything else is built on this.

Run in order:

```bash
python 01_single_call.py      # one message → one reply + token count
python 02_streaming.py        # the reply types out live
python 03_chatbot_memory.py   # a real chatbot loop — memory via the messages list
```

### The lesson to actually feel
In `03_chatbot_memory.py`:
1. Run it. Say `My name is Aom.` then `What's my name?` — it remembers.
2. Open the file, set `BREAK_MEMORY = True`, rerun. Now it forgets — because we stopped appending the assistant's replies to `messages`.
3. Set it back to `False`. That broken run is *why* the `messages` list matters.

**Carry forward:** every agent in this course is this `messages` loop with extras bolted on — next we give it tools.

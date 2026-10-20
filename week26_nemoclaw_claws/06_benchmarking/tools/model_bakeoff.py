#!/usr/bin/env python3
"""Model bake-off · one OpenAI-compatible model server, the same five checks for every model.

Section 7 of Module 06. Point it at any vLLM / SGLang / Ollama endpoint and it measures:
  1. speed: time to first token and decode tok/s, one stream, thinking off (median of 3)
  2. HVAC knowledge: data/bakeoff_cases.json (or --cases), numbers graded, thinking off and on
  3. tool calls: 4 BMS-style requests, right tool + right arguments, thinking off
  4. vision (--vision): 3 synthetic plant images with known answers (data/vision/)
Every request is sent one at a time with temperature 0, so models are compared like for like.

    # on: spark
    python3 week26_nemoclaw_claws/06_benchmarking/tools/model_bakeoff.py --name "Nemotron 3.5 Super VL" \
        --url http://127.0.0.1:8000/v1 --model nemotron-3.5-super-vl --vision

Standard library only. Results append to --out as one JSON line per model.
"""
from __future__ import annotations

import argparse, base64, json, re, statistics, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def _post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def chat(url, model, messages, *, max_tokens, thinking, tools=None, stream=False, timeout=900):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if tools:
        body |= {"tools": tools, "tool_choice": "auto"}
    t0 = time.perf_counter()
    if not stream:
        with _post(f"{url}/chat/completions", body, timeout) as r:
            return json.load(r), time.perf_counter() - t0
    body |= {"stream": True, "stream_options": {"include_usage": True}}
    first, usage = None, {}
    with _post(f"{url}/chat/completions", body, timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ch = json.loads(line[6:])
            usage = ch.get("usage") or usage
            for c in ch.get("choices") or []:
                d = c.get("delta") or {}
                if first is None and (d.get("content") or d.get("reasoning") or d.get("reasoning_content")):
                    first = time.perf_counter()
    end = time.perf_counter()
    return {"ttft": (first or end) - t0, "total": end - t0, "tokens": usage.get("completion_tokens", 0)}, end - t0


def numbers_in(text):
    out = []
    for m in re.findall(r"-?\d+(?:[.,]\d+)*", text or ""):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def near(value, tol, found):
    return any(abs(x - value) <= tol for x in found)


TOOLS = [
    {"type": "function", "function": {"name": "read_point", "description": "Read the live value of a BMS point",
     "parameters": {"type": "object", "properties": {"equipment": {"type": "string"}, "point": {"type": "string"}},
                    "required": ["equipment", "point"]}}},
    {"type": "function", "function": {"name": "get_trend", "description": "Get the history of a BMS point",
     "parameters": {"type": "object", "properties": {"equipment": {"type": "string"}, "point": {"type": "string"},
                    "hours": {"type": "integer"}}, "required": ["equipment", "point", "hours"]}}},
    {"type": "function", "function": {"name": "list_alarms", "description": "List active alarms at a site",
     "parameters": {"type": "object", "properties": {"severity": {"type": "string", "enum": ["low", "medium", "high"]}},
                    "required": []}}},
]
TOOL_CASES = [
    ("What's the chilled water supply temperature on CH-2 right now?", "read_point", {"equipment": "CH-2"}),
    ("Show me CH-1's power consumption over the last 24 hours.", "get_trend", {"equipment": "CH-1", "hours": 24}),
    ("Are there any high-severity alarms right now?", "list_alarms", {"severity": "high"}),
    ("ตอนนี้ AHU-3 supply air temperature เท่าไร", "read_point", {"equipment": "AHU-3"}),
]
VISION = [  # synthetic images (data/vision/make_images.py), so the right answer is known exactly
    ("plant_table.png", "From this chiller plant screen, compute kW/RT for each chiller and say which chiller is the least efficient. Give the numbers.",
     lambda t: "CH-3" in t and near(0.95, 0.01, numbers_in(t)) and near(0.638, 0.01, numbers_in(t))),
    ("chws_trend.png", "At what hour does the chilled water supply temperature spike, and what is the peak value in °C?",
     lambda t: near(14, 0.3, numbers_in(t)) and near(9.5, 0.2, numbers_in(t))),
    ("nameplate.png", "Read this nameplate: capacity, refrigerant, rated input power. Then compute the rated kW/RT.",
     lambda t: "134" in t and near(1200, 0, numbers_in(t)) and near(680, 0, numbers_in(t)) and near(0.567, 0.005, numbers_in(t))),
]


def norm(x):  # CH-2, CH‑2 (non-breaking hyphen), "ch 2" all compare equal
    return re.sub(r"[\s‐-―−_-]", "", str(x)).upper()


def run(a):
    cases = json.loads(Path(a.cases).read_text(encoding="utf-8"))
    res = {"name": a.name, "model": a.model, "when": time.strftime("%Y-%m-%d %H:%M")}
    chat(a.url, a.model, [{"role": "user", "content": "Say ready."}], max_tokens=20, thinking=False)  # warm-up

    prompt = [{"role": "user", "content": "Explain chiller plant staging and sequencing to a new HVAC engineer in about 400 words."}]
    runs = [chat(a.url, a.model, prompt, max_tokens=600, thinking=False, stream=True)[0] for _ in range(3)]
    res["ttft_s"] = round(statistics.median(r["ttft"] for r in runs), 2)
    res["decode_tok_s"] = round(statistics.median(r["tokens"] / max(1e-6, r["total"] - r["ttft"]) for r in runs), 1)
    print(f"◆ speed: first token {res['ttft_s']} s · {res['decode_tok_s']} tok/s (one stream, median of 3)", flush=True)

    for think in (False, True):
        ok, secs = 0, 0.0
        for c in cases:
            j, dt = chat(a.url, a.model, [{"role": "user", "content": c["prompt"]}],
                         max_tokens=6000 if think else 1500, thinking=think)
            found = numbers_in(j["choices"][0]["message"].get("content") or "")
            ok += all(near(e["value"], e.get("tolerance", 0), found) for e in c["expect"]); secs += dt
        key = "knowledge_think" if think else "knowledge_nothink"
        res[key], res[key + "_avg_s"] = f"{ok}/{len(cases)}", round(secs / len(cases), 1)
        print(f"◆ knowledge, thinking {'on ' if think else 'off'}: {ok}/{len(cases)} · avg {res[key + '_avg_s']} s", flush=True)

    ok = 0
    for q, name, want in TOOL_CASES:
        j, _ = chat(a.url, a.model, [{"role": "user", "content": q}], max_tokens=800, thinking=False, tools=TOOLS)
        calls = j["choices"][0]["message"].get("tool_calls") or []
        good = bool(calls) and calls[0]["function"]["name"] == name
        if good:
            args = json.loads(calls[0]["function"]["arguments"] or "{}")
            good = all(norm(args.get(k, "")) == norm(v) for k, v in want.items())
        ok += good
    res["tools"] = f"{ok}/{len(TOOL_CASES)}"
    print(f"◆ tool calls: {res['tools']}", flush=True)

    if a.vision:
        ok = 0
        for img, q, check in VISION:
            b64 = base64.b64encode((HERE / "data" / "vision" / img).read_bytes()).decode()
            msg = [{"role": "user", "content": [{"type": "text", "text": q},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]
            j, dt = chat(a.url, a.model, msg, max_tokens=4000, thinking=True)
            text = j["choices"][0]["message"].get("content") or ""
            passed = bool(check(text)); ok += passed
            print(f"  {'✓' if passed else '✕'} {img} ({dt:.1f} s): {text.strip()[:90]!r}", flush=True)
        res["vision"] = f"{ok}/{len(VISION)}"
        print(f"◆ vision: {res['vision']}", flush=True)

    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--name", required=True)
    p.add_argument("--url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--vision", action="store_true", help="also run the three image checks")
    p.add_argument("--cases", default=str(HERE / "data" / "bakeoff_cases.json"))
    p.add_argument("--out", default="bakeoff_results.jsonl")
    run(p.parse_args())

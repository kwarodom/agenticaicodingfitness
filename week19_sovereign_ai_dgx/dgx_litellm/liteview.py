#!/usr/bin/env python3
"""Make the **LiteLLM gateway** VISIBLE — route → backend → answer.

Demos call this one module. Depending on what's reachable it either talks to a
real LiteLLM proxy, makes a real call to the underlying DGX backend (illustrating
the routing with the simulator), or fully simulates. Narration is plain text so
it renders in a terminal and the web app.

    GATEWAY  — one OpenAI-compatible URL + key in front of all DGX backends
    ROUTE    — which deployment / Spark the proxy picked, and why
    ANSWER   — the streamed completion
    METRIC   — tokens, latency, cloud cost $0
"""
from __future__ import annotations

import sys
import time

import config
import litesim

S_GW = "▣"
S_ROUTE = "  ⇄"
S_ANSWER = "  ·"
S_METRIC = "  ◆"


def _p(line: str = "") -> None:
    print(line, flush=True)


def is_sim() -> bool:
    return config.MODE != "real"


def banner(part: str, title: str, level: str) -> None:
    _p("━" * 64)
    _p(f"  {part} — {title}   [{level}]")
    _p("━" * 64)
    _p("")


def mode_line() -> None:
    s = config.SITUATION
    if s == "proxy":
        _p(f"{S_GW} MODE: REAL · PROXY — a LiteLLM proxy answers at {config.LITELLM_BASE_URL}.")
        _p("  calls go through the gateway; routing is real.")
    elif s == "direct":
        _p(f"{S_GW} MODE: REAL · DIRECT — no proxy, but a backend is up at {config.safe_backend_url()}.")
        _p(f"  backend connection: {config.CONN} ({config.conn_human()}).")
        _p("  generation is a real local call; routing/keys are shown via the simulator.")
    else:
        _p(f"{S_GW} MODE: SIM — no proxy or backend reachable; gateway fully simulated.")
        _p(f"  backend connection would be: {config.CONN} ({config.conn_human()}).")
        _p("  the litellm commands + config shown are exactly what you'd run for real.")
    _p("")


def _client(base_url: str, key: str):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=key, timeout=120.0)


def generate(prompt: str, *, alias: str = "dgx-fast", strategy: str = "least-busy",
             max_tokens: int = config.DEFAULT_MAX_TOKENS, show_route: bool = True) -> dict:
    """One gateway call. Returns {answer, tokens, elapsed_s, route}."""
    out = {"answer": "", "tokens": 0, "elapsed_s": 0.0, "route": None}
    started = time.time()

    if config.SITUATION == "proxy":
        # real proxy: it does the routing; we just call the alias
        _p(f"{S_ROUTE} GATEWAY: calling alias '{alias}' at {config.LITELLM_BASE_URL} (proxy routes it)")
        client = _client(config.LITELLM_BASE_URL + "/v1", config.LITELLM_KEY)
        resp = client.chat.completions.create(
            model=alias, messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.3)
        out["answer"] = (resp.choices[0].message.content or "").strip()
        _p(f"{S_ANSWER} {out['answer']}")
    else:
        # show the routing decision the proxy WOULD make
        dep, logs = litesim.route(alias, strategy)
        if show_route:
            for ln in logs:
                _p(f"{S_ROUTE} {ln}")
        out["route"] = dep
        if config.SITUATION == "direct":
            # make a REAL call to the backend the router picked
            client = _client(config.BACKEND_URL, config.BACKEND_KEY)
            resp = client.chat.completions.create(
                model=config.MODEL, messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=0.3)
            msg = resp.choices[0].message
            content = (msg.content or "").strip()
            if not content:  # thinking models can leave content empty
                extra = getattr(msg, "model_extra", None) or {}
                content = (getattr(msg, "reasoning", None) or extra.get("reasoning") or "").strip()
            out["answer"] = content
            _p(f"{S_ANSWER} {content}")
        else:
            _p(f"{S_ANSWER} ", )
            for chunk in litesim.stream_generate(prompt, alias):
                out["answer"] += chunk
                print(chunk, end="", flush=True)
            _p("")

    out["elapsed_s"] = time.time() - started
    out["tokens"] = max(1, round(len(out["answer"]) / 4))
    _p(f"{S_METRIC} ~{out['tokens']} tokens in {out['elapsed_s']:.1f}s · one OpenAI-"
       f"compatible call · stayed on-prem · $0.0000")
    return out


if __name__ == "__main__":
    _p("liteview.py is a helper imported by the demos in demos/.")
    sys.exit(0)

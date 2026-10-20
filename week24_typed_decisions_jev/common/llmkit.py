#!/usr/bin/env python3
"""llmkit — one tiny, standard-library client for the generative LLM you pair with Jev.

Jev DECIDES (typed answers), the LLM WRITES (prose). Pick any provider:

    from llmkit import generate, providers
    r = generate("claude", system="You are a hotel concierge.", user="Draft a 2-line apology.")
    print(r["text"])

Providers (all but Claude speak the OpenAI-compatible /chat/completions API):

    claude      Anthropic       ANTHROPIC_API_KEY
    chatgpt     OpenAI          OPENAI_API_KEY
    gemini      Google          GEMINI_API_KEY   (or GOOGLE_API_KEY)
    deepseek    DeepSeek        DEEPSEEK_API_KEY
    kimi        Moonshot AI     KIMI_API_KEY     (or MOONSHOT_API_KEY)
    glm         Z.ai / Zhipu    ZAI_API_KEY      (or ZHIPU_API_KEY)
    openrouter  OpenRouter      OPENROUTER_API_KEY  (one key, hundreds of models)
    ollama      local Ollama    (no key)  — $0, runs on your machine

Keys are looked up, in order: the process environment → week24/.env.local
(what the Lab Runner's 🔑 Keys dialog saves; gitignored) → the repo-root .env
(gitignored). Keys are never printed, logged, recorded or sent to a browser.

Modes follow jevkit: JEV_MODE=dry (or no key for that provider) replays a
recorded answer for the exact same request, else returns a clearly-labelled
placeholder. Override a base URL with e.g. KIMI_BASE_URL / GLM_BASE_URL
(China-region endpoints) or OLLAMA_BASE_URL.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

COMMON = Path(__file__).resolve().parent
WEEK24 = COMMON.parent
ROOT = WEEK24.parent
LOCAL_KEYS = WEEK24 / ".env.local"
RECORDED = COMMON / "recorded_llm"

# Default model ids were checked against each provider's live /models list on 2026-09-27.
# Models change fast: the Lab Runner can list live ids, and any id can be typed in.
PROVIDERS: dict[str, dict] = {
    "claude": {
        "label": "Claude (Anthropic)", "kind": "anthropic", "base": "https://api.anthropic.com/v1",
        "keys": ["ANTHROPIC_API_KEY"], "default": "claude-sonnet-5",
        "suggested": ["claude-sonnet-5", "claude-opus-5-5", "claude-haiku-4-5-20251001"],
        "get_key": "https://console.anthropic.com/settings/keys"},
    "chatgpt": {
        "label": "ChatGPT (OpenAI)", "kind": "openai", "base": "https://api.openai.com/v1",
        "keys": ["OPENAI_API_KEY"], "default": "gpt-5.4-mini",
        "suggested": ["gpt-5.4-mini", "gpt-5.5", "gpt-5.4-nano"],
        "tokens_field": "max_completion_tokens", "extra": {"reasoning_effort": "low"},
        "get_key": "https://platform.openai.com/api-keys"},
    "gemini": {
        "label": "Gemini (Google)", "kind": "openai",
        "base": "https://generativelanguage.googleapis.com/v1beta/openai",
        "keys": ["GEMINI_API_KEY", "GOOGLE_API_KEY"], "default": "gemini-3.5-flash",
        "suggested": ["gemini-3.5-flash", "gemini-3.1-pro-preview", "gemini-3.5-flash-lite"],
        "extra": {"reasoning_effort": "minimal"}, "strip_prefix": "models/",   # "low" still ate short budgets
        "get_key": "https://aistudio.google.com/apikey"},
    "deepseek": {
        "label": "DeepSeek", "kind": "openai", "base": "https://api.deepseek.com",
        "keys": ["DEEPSEEK_API_KEY"], "default": "deepseek-flash",
        "suggested": ["deepseek-flash", "deepseek-v4-pro"],
        "get_key": "https://platform.deepseek.com/api_keys"},
    "kimi": {
        "label": "Kimi (Moonshot AI)", "kind": "openai", "base": "https://api.moonshot.ai/v1",
        "base_env": "KIMI_BASE_URL", "keys": ["KIMI_API_KEY", "MOONSHOT_API_KEY"], "default": "kimi-k3",
        "suggested": ["kimi-k3", "kimi-k2.6"],
        "get_key": "https://platform.moonshot.ai/console/api-keys"},
    "glm": {
        "label": "GLM (Z.ai / Zhipu)", "kind": "openai", "base": "https://api.z.ai/api/paas/v4",
        "base_env": "GLM_BASE_URL", "keys": ["ZAI_API_KEY", "ZHIPU_API_KEY", "GLM_API_KEY"],
        "default": "glm-5.3", "suggested": ["glm-5.3", "glm-5.3-flash", "glm-5.2"],
        "extra": {"reasoning_effort": "low"},          # glm-5.x always thinks; keep it brief
        "get_key": "https://z.ai/manage-apikey/apikey-list"},
    "openrouter": {
        "label": "OpenRouter (any model)", "kind": "openai", "base": "https://openrouter.ai/api/v1",
        "keys": ["OPENROUTER_API_KEY"], "default": "openai/gpt-5.4-mini",
        "suggested": ["openai/gpt-5.4-mini", "anthropic/claude-sonnet-5", "google/gemini-3.5-flash"],
        "get_key": "https://openrouter.ai/settings/keys"},
    "ollama": {
        "label": "Ollama (local, $0)", "kind": "openai", "base": "http://localhost:11434/v1",
        "base_env": "OLLAMA_BASE_URL", "keys": [], "key_optional": True,
        "default_env": "OLLAMA_MODEL", "default": "gemma4:latest",
        "suggested": ["gemma4:latest", "nemotron-3-nano:latest"],
        "get_key": "https://ollama.com/download"},
}
DEFAULT_PROVIDER = "claude"
MODEL_RE = re.compile(r"^[A-Za-z0-9._:/@+-]{1,120}$")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class LLMError(RuntimeError):
    pass


# ── keys & configuration ──────────────────────────────────────────────────────
def _read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*(?:export\s+)?([A-Z0-9_]+)\s*=\s*(.*)$", line)
            if m and not line.lstrip().startswith("#"):
                v = m.group(2).strip()
                if v[:1] in ("'", '"'):
                    v = v[1:].split(v[0], 1)[0]            # quoted: take what is inside the quotes
                else:
                    v = re.split(r"\s+#", v, 1)[0].strip()  # unquoted: drop an inline "# comment"
                out[m.group(1)] = v
    except OSError:
        pass
    return out


def _lookup(name: str) -> tuple[str, str]:
    """(value, source) for one variable; source ∈ env | week24/.env.local | repo .env | ''."""
    if os.environ.get(name, "").strip():
        return os.environ[name].strip(), "env"
    v = _read_env_file(LOCAL_KEYS).get(name, "")
    if v:
        return v, "week24/.env.local"
    v = _read_env_file(ROOT / ".env").get(name, "")
    if v:
        return v, "repo .env"
    return "", ""


def api_key(provider: str) -> tuple[str, str]:
    for name in PROVIDERS[provider]["keys"]:
        v, src = _lookup(name)
        if v:
            return v, src
    return "", ""


def base_url(provider: str) -> str:
    p = PROVIDERS[provider]
    if p.get("base_env"):
        v, _ = _lookup(p["base_env"])
        if v:
            return v.rstrip("/")
    return p["base"]


def default_model(provider: str) -> str:
    p = PROVIDERS[provider]
    if p.get("default_env"):
        v, _ = _lookup(p["default_env"])
        if v:
            return v
    return p["default"]


def chosen() -> tuple[str, str]:
    """(provider, model) from JEV_LLM_PROVIDER / JEV_LLM_MODEL (the Lab Runner sets them)."""
    prov = os.environ.get("JEV_LLM_PROVIDER", "").strip() or DEFAULT_PROVIDER
    if prov not in PROVIDERS:
        prov = DEFAULT_PROVIDER
    model = os.environ.get("JEV_LLM_MODEL", "").strip()
    return prov, (model if MODEL_RE.match(model or "") else default_model(prov))


def available(provider: str) -> bool:
    return bool(api_key(provider)[0]) or bool(PROVIDERS[provider].get("key_optional"))


def providers() -> list[dict]:
    """Status for every provider — never includes a key value."""
    out = []
    for pid, p in PROVIDERS.items():
        _, src = api_key(pid)
        out.append({"id": pid, "label": p["label"], "default": default_model(pid),
                    "suggested": p["suggested"], "key_names": p["keys"], "has_key": bool(src),
                    "key_source": src or ("not needed" if p.get("key_optional") else ""),
                    "key_optional": bool(p.get("key_optional")), "get_key": p["get_key"],
                    "base_url": base_url(pid)})
    return out


def save_local_key(provider: str, value: str) -> str:
    """Write/replace this provider's first key name in week24/.env.local (0600). '' deletes it."""
    name = PROVIDERS[provider]["keys"][0] if PROVIDERS[provider]["keys"] else None
    if not name:
        raise LLMError(f"{provider} needs no key")
    value = value.strip()
    if value and not re.fullmatch(r"[A-Za-z0-9._\-:+/=]{8,400}", value):
        raise LLMError("that does not look like an API key (unexpected characters or length)")
    lines = []
    if LOCAL_KEYS.is_file():
        lines = [l for l in LOCAL_KEYS.read_text(encoding="utf-8").splitlines()
                 if not re.match(rf"\s*(?:export\s+)?{name}\s*=", l)]
    if value:
        lines.append(f"{name}={value}")
    header = "# Saved by the Jev Lab Runner 🔑 Keys dialog. Gitignored — never commit this file."
    body = [l for l in lines if l.strip() and not l.startswith("# Saved by")]
    LOCAL_KEYS.write_text("\n".join([header] + body) + "\n", encoding="utf-8")
    try:
        os.chmod(LOCAL_KEYS, 0o600)
    except OSError:
        pass
    return name


# ── the call ──────────────────────────────────────────────────────────────────
def _post(url: str, body: dict, headers: dict, timeout: float) -> dict:
    req = Request(url, data=json.dumps(body).encode(), method="POST",
                  headers={"Content-Type": "application/json", "User-Agent": "week24-jev-lab/1", **headers})
    for attempt in range(3):
        try:
            with urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 529) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            detail = exc.read().decode(errors="replace")[:300]
            hint = {401: "key missing or invalid", 403: "key lacks access to this model",
                    404: "unknown model id or endpoint", 402: "billing / credits",
                    429: "rate limited"}.get(exc.code, "provider error")
            raise LLMError(f"HTTP {exc.code} ({hint}) {detail}") from None
        except (URLError, TimeoutError) as exc:
            raise LLMError(f"network/timeout: {exc}") from None
    raise LLMError("gave up after 3 attempts")


def request_id(provider, model, system, user, max_tokens) -> str:
    blob = json.dumps([provider, model, system, user, max_tokens], ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def generate(provider: str | None = None, *, system: str = "", user: str, model: str | None = None,
             max_tokens: int = 600, timeout: float = 90, mode_override: str | None = None) -> dict:
    """Return {"text","provider","model","usage":{input_tokens,output_tokens},"latency_ms","_source"}."""
    if provider is None:
        provider, model = (lambda p, m: (p, model or m))(*chosen())
    if provider not in PROVIDERS:
        raise LLMError(f"unknown provider {provider!r}; choose one of {', '.join(PROVIDERS)}")
    p = PROVIDERS[provider]
    model = model or default_model(provider)
    if not MODEL_RE.match(model):
        raise LLMError("invalid model id")
    rid = request_id(provider, model, system, user, max_tokens)
    cache = RECORDED / f"{rid}.json"
    forced = mode_override or os.environ.get("JEV_MODE", "").lower()
    key, _ = api_key(provider)
    live = forced != "dry" and (bool(key) or p.get("key_optional"))

    if not live:
        if cache.is_file():
            r = json.loads(cache.read_text(encoding="utf-8"))["response"]
            r["_source"] = "recorded"
            return r
        why = "DRY mode" if forced == "dry" else f"no key for {p['label']} ({' / '.join(p['keys'])})"
        return {"text": f"[{why} — no recorded answer for this exact prompt. Add a key (🔑 Keys) "
                        f"and switch to ⚡ Live to generate for real.]",
                "provider": provider, "model": model, "usage": {"input_tokens": 0, "output_tokens": 0},
                "latency_ms": None, "_source": "placeholder"}

    t0 = time.perf_counter()
    if p["kind"] == "anthropic":
        body = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": user}]}
        if system:
            body["system"] = system
        d = _post(base_url(provider) + "/messages", body,
                  {"x-api-key": key, "anthropic-version": "2023-06-01"}, timeout)
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        if d.get("stop_reason") == "max_tokens":
            text += " …[cut off: token limit reached — raise max_tokens]"
        u = d.get("usage") or {}
        usage = {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0)}
        answered_by = d.get("model", model)
    else:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        body = {"model": model, "messages": msgs, p.get("tokens_field", "max_tokens"): max_tokens}
        body.update(p.get("extra") or {})
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        d = _post(base_url(provider) + "/chat/completions", body, headers, timeout)
        choice = (d.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        if isinstance(text, list):                     # some gateways return content parts
            text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
        if not text.strip() and (msg.get("reasoning_content") or msg.get("reasoning")):
            text = "[the model used its whole token budget thinking — raise max_tokens or pick a non-thinking model]"
        elif choice.get("finish_reason") == "length":
            text += " …[cut off: token limit reached — raise max_tokens]"
        u = d.get("usage") or {}
        usage = {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)}
        answered_by = d.get("model", model)
    r = {"text": text.strip(), "provider": provider, "model": answered_by, "usage": usage,
         "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "_source": "live"}
    if os.environ.get("JEV_RECORD") == "1":
        RECORDED.mkdir(exist_ok=True)
        keep = {k: v for k, v in r.items() if not k.startswith("_")}
        cache.write_text(json.dumps({"recorded_at": time.strftime("%Y-%m-%d"),
                                     "request": {"provider": provider, "model": model, "system": system,
                                                 "user": user, "max_tokens": max_tokens},
                                     "response": keep}, ensure_ascii=False, indent=1), encoding="utf-8")
    return r


def list_models(provider: str, timeout: float = 12) -> list[str]:
    """Live model ids from the provider (for the Lab Runner's model picker)."""
    p = PROVIDERS[provider]
    key, _ = api_key(provider)
    if not key and not p.get("key_optional"):
        return []
    if p["kind"] == "anthropic":
        req = Request(base_url(provider) + "/models?limit=100",
                      headers={"x-api-key": key, "anthropic-version": "2023-06-01", "User-Agent": "week24-jev-lab/1"})
    else:
        req = Request(base_url(provider) + "/models",
                      headers={"User-Agent": "week24-jev-lab/1", **({"Authorization": f"Bearer {key}"} if key else {})})
    with urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    ids = [(m.get("id") or m.get("name") or "") for m in (d.get("data") or d.get("models") or [])]
    pre = p.get("strip_prefix")
    ids = [i[len(pre):] if pre and i.startswith(pre) else i for i in ids]
    return sorted(i for i in ids if i and MODEL_RE.match(i))


def show(r: dict, width: int = 96) -> None:
    """Print a generation with the glyph conventions the Lab Runner colourises."""
    tag = {"live": "LIVE", "recorded": "RECORDED replay", "placeholder": "PLACEHOLDER"}.get(r.get("_source"), "?")
    for line in (r.get("text") or "").splitlines() or [""]:
        while len(line) > width:
            cut = line.rfind(" ", 0, width)
            cut = cut if cut > 20 else width
            print("   │ " + line[:cut])
            line = line[cut:].lstrip()
        print("   │ " + line)
    u = r.get("usage") or {}
    lat = f" · {r['latency_ms']:.0f} ms" if r.get("latency_ms") else ""
    print(f"◆ {PROVIDERS.get(r.get('provider'), {}).get('label', r.get('provider'))} · {r.get('model')} · {tag} · "
          f"{u.get('input_tokens', 0)} in / {u.get('output_tokens', 0)} out tok{lat}")


if __name__ == "__main__":                              # python llmkit.py → key status + one hello per provider
    for s in providers():
        mark = "✓" if s["has_key"] or s["key_optional"] else "·"
        print(f"{mark} {s['id']:<11} {s['label']:<24} default {s['default']:<26} key: {s['key_source'] or 'missing'}")
    if "--hello" in sys.argv:
        for s in providers():
            if not (s["has_key"] or s["key_optional"]):
                continue
            try:
                r = generate(s["id"], user="Reply with exactly: hello from <your model name>", max_tokens=200)
                print(f"\n━━ {s['id']}"); show(r)
            except LLMError as e:
                print(f"\n✕ {s['id']}: {e}")

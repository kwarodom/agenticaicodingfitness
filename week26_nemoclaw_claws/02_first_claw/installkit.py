#!/usr/bin/env python3
"""installkit — Module 02's small helper for the NemoClaw one-command installer line.

  build(agent=, provider=, sandbox=, tier=, web_search=, ...)  →  the non-interactive
        `curl -fsSL https://www.nvidia.com/nemoclaw.sh | VARS bash` line, with <placeholders> for keys
  parse(line)   →  which variables sit on the curl side and which on the bash side of the pipe
  lint(line)    →  (errors, warnings) against the rules the research tutorial's Part 1 documents

It never runs the installer. It only builds and checks a string; the learner pastes it into the ⌨ terminal.

Every value list below is copied from the research tutorial, Part 1 §1.1 / §1.3 / §1.4 (which cites the NemoClaw
OpenClaw + Hermes quickstarts and the network-policies reference) and the DGX Spark NemoClaw playbook's
provider mapping. Where a rule is the course's own (the sandbox-name regex, "looks like a real key"), the
message says "(course rule)".
"""
from __future__ import annotations

import re
import shlex

INSTALLER_URL = "https://www.nvidia.com/nemoclaw.sh"
CURL = f"curl -fsSL {INSTALLER_URL}"

AGENTS = {"openclaw": "nemoclaw", "hermes": "nemohermes", "langchain-deepagents-code": "nemo-deepagents"}

# provider → (what it is, key variable or None, runs on this Spark?)
PROVIDERS = {
    "build": ("NVIDIA Endpoints", "NVIDIA_INFERENCE_API_KEY", False),
    "openrouter": ("OpenRouter", "OPENROUTER_API_KEY", False),
    "openai": ("OpenAI", "OPENAI_API_KEY", False),
    "anthropic": ("Anthropic", "ANTHROPIC_API_KEY", False),
    "gemini": ("Google Gemini", "GEMINI_API_KEY", False),
    "routed": ("Model Router", "NVIDIA_INFERENCE_API_KEY", False),
    "custom": ("any /v1/chat/completions endpoint", "COMPATIBLE_API_KEY", False),
    "anthropicCompatible": ("any Anthropic-compatible endpoint", "COMPATIBLE_ANTHROPIC_API_KEY", False),
    "ollama": ("local Ollama (optional NEMOCLAW_MODEL)", None, True),
    "vllm": ("an already-running vLLM on localhost:${NEMOCLAW_VLLM_PORT:-8000}", None, True),
    "install-vllm": ("managed, Docker-backed vLLM (large download)", None, True),
    "hermes-provider": ("Hermes Provider (Hermes only)", None, False),
}
TIERS = ("restricted", "balanced", "open", "personal")
WEB_SEARCH = ("tavily", "none")
KNOWN_VARS = {
    "NEMOCLAW_NON_INTERACTIVE", "NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE", "NEMOCLAW_AGENT", "NEMOCLAW_PROVIDER",
    "NEMOCLAW_SANDBOX_NAME", "NEMOCLAW_MODEL", "NEMOCLAW_VLLM_PORT", "NEMOCLAW_POLICY_TIER",
    "NEMOCLAW_WEB_SEARCH_PROVIDER", "NEMOCLAW_NO_EXPRESS", "NEMOCLAW_GATEWAY_RUNTIME", "NEMOCLAW_INSTALL_REF",
    "NEMOCLAW_INSTALL_TAG", "NEMOCLAW_YES", "TAVILY_API_KEY", "BRAVE_API_KEY",
    *(v[1] for v in PROVIDERS.values() if v[1]),
}
SECRET_VARS = {v[1] for v in PROVIDERS.values() if v[1]} | {"TAVILY_API_KEY", "BRAVE_API_KEY"}
SANDBOX_RE = re.compile(r"^[a-z0-9-]{1,40}$")      # the Reef runner's name rule (spec §4.4) — course rule
PLACEHOLDER_RE = re.compile(r"<[A-Za-z0-9_.\- ]{1,40}>")
REAL_KEY_RE = re.compile(r"^(nvapi-|sk-|tvly-|hf_|BSA)[A-Za-z0-9_\-]{8,}")


def build(*, agent: str = "openclaw", provider: str = "", sandbox: str = "my-assistant", tier: str = "",
          web_search: str = "", model: str = "", vllm_port: int | None = None, non_interactive: bool = True,
          accept: bool = True, pin_tag: str = "") -> str:
    """The install line, one variable per continuation line. Keys become '<your-key>' placeholders (quoted,
    so a pasted-but-unedited line fails loudly instead of turning < and > into redirections)."""
    env: list[tuple[str, str]] = []
    if non_interactive:
        env.append(("NEMOCLAW_NON_INTERACTIVE", "1"))
    if accept:
        env.append(("NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE", "1"))
    if pin_tag:
        env += [("NEMOCLAW_INSTALL_REF", ""), ("NEMOCLAW_INSTALL_TAG", pin_tag)]
    if agent:
        env.append(("NEMOCLAW_AGENT", agent))
    if sandbox:
        env.append(("NEMOCLAW_SANDBOX_NAME", sandbox))
    if provider:
        env.append(("NEMOCLAW_PROVIDER", provider))
        key = PROVIDERS.get(provider, ("", None, False))[1]
        if key:
            env.append((key, "'<your-key>'"))
    if model:
        env.append(("NEMOCLAW_MODEL", model))
    if vllm_port is not None:
        env.append(("NEMOCLAW_VLLM_PORT", str(vllm_port)))
    if web_search:
        env.append(("NEMOCLAW_WEB_SEARCH_PROVIDER", web_search))
        if web_search == "tavily":
            env.append(("TAVILY_API_KEY", "'<your-tavily-key>'"))
    if tier:
        env.append(("NEMOCLAW_POLICY_TIER", tier))
    body = " \\\n  ".join(f"{k}={v}" for k, v in env)
    return f"{CURL} | \\\n  {body} \\\n  bash" if env else f"{CURL} | bash"


def parse(line: str) -> dict:
    """Split an install line at the pipe. Returns {curl: [...], bash_env: {VAR: value}, bash_args: [...],
    curl_env: {VAR: value}, placeholders_unquoted: [...], problems: [...]} — no shell is involved."""
    text = line.replace("\\\n", " ").strip()
    out = {"curl": [], "curl_env": {}, "bash_env": {}, "bash_args": [], "placeholders_unquoted": [],
           "problems": [], "bash_cmd": ""}
    # <placeholders> outside quotes are redirections to bash; note them, then parse them as plain words
    unquoted = []
    for m in PLACEHOLDER_RE.finditer(text):
        before = text[:m.start()]
        q = "'" if before.count("'") % 2 else ('"' if before.count('"') % 2 else "")
        if not q:
            unquoted.append(m.group(0))
    out["placeholders_unquoted"] = unquoted
    found: list[str] = []

    def _stash(m: re.Match) -> str:
        found.append(m.group(0))
        return f"@@PH{len(found) - 1}@@"

    def _restore(s: str) -> str:
        return re.sub(r"@@PH(\d+)@@", lambda m: found[int(m.group(1))], s)

    safe = PLACEHOLDER_RE.sub(_stash, text)
    try:
        lex = shlex.shlex(safe, posix=True, punctuation_chars="|")
        lex.whitespace_split = True
        toks = [_restore(t) for t in lex]
    except ValueError as e:
        out["problems"].append(f"cannot parse the line: {e}")
        return out
    if toks.count("|") != 1:
        out["problems"].append(f"expected exactly one pipe (curl … | … bash), found {toks.count('|')}")
        if "|" not in toks:
            return out
    left, right = toks[:toks.index("|")], toks[toks.index("|") + 1:]
    assign = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.S)
    j = 0
    while j < len(left) and assign.match(left[j]):
        k, v = assign.match(left[j]).groups()
        out["curl_env"][k] = v
        j += 1
    out["curl"] = left[j:]
    j = 0
    while j < len(right) and assign.match(right[j]):
        k, v = assign.match(right[j]).groups()
        out["bash_env"][k] = v
        j += 1
    rest = right[j:]
    out["bash_cmd"] = rest[0] if rest else ""
    out["bash_args"] = rest[1:]
    return out


def lint(line: str) -> tuple[list[str], list[str]]:
    """(errors, warnings) for one install line."""
    p = parse(line)
    errs, warns = list(p["problems"]), []
    if p["problems"] and not p["curl"]:
        return errs, warns
    if " ".join(p["curl"][:3]) != CURL:
        errs.append(f"the left side must be exactly `{CURL}` (got `{' '.join(p['curl'])}`)")
    if p["curl_env"]:
        errs.append(f"{', '.join(p['curl_env'])} set on the curl side of the pipe — curl only downloads the script; "
                    "the bash process that runs it never sees these. Move them after the `|`.")
    if p["bash_cmd"] != "bash":
        errs.append(f"the right side must end in `bash` (got `{p['bash_cmd'] or 'nothing'}`)")
    env, args = p["bash_env"], p["bash_args"]
    if args and args[:2] != ["-s", "--"]:
        errs.append(f"installer flags go after `bash -s --` (got `bash {' '.join(args)}`)")
    accepted = env.get("NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE") == "1" or "--yes-i-accept-third-party-software" in args
    if env.get("NEMOCLAW_NON_INTERACTIVE") == "1" and not accepted:
        errs.append("NEMOCLAW_NON_INTERACTIVE=1 without NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 — nobody is there to "
                    "accept the third-party software notice")
    for k in env:
        if k not in KNOWN_VARS:
            warns.append(f"{k} is not in the course's source list — check the quickstart before relying on it")
    agent = env.get("NEMOCLAW_AGENT", "openclaw")
    if agent not in AGENTS:
        errs.append(f"NEMOCLAW_AGENT={agent!r} — use one of {', '.join(AGENTS)}")
    prov = env.get("NEMOCLAW_PROVIDER", "")
    if prov and prov not in PROVIDERS:
        errs.append(f"NEMOCLAW_PROVIDER={prov!r} is not a documented value — one of {', '.join(PROVIDERS)}")
    if prov == "hermes-provider" and agent != "hermes":
        errs.append("hermes-provider works with NEMOCLAW_AGENT=hermes only")
    if prov == "ollama" and agent == "langchain-deepagents-code":
        warns.append("Local Ollama is not offered for Deep Agents (playbook starter prompt) unless the docs add support")
    if prov in PROVIDERS and PROVIDERS[prov][1] and env.get("NEMOCLAW_NON_INTERACTIVE") == "1":
        key = PROVIDERS[prov][1]
        if key not in env:
            errs.append(f"provider {prov} needs {key} in a non-interactive run (as a placeholder here)")
    if prov in ("custom", "anthropicCompatible"):
        warns.append(f"{prov} also needs an endpoint and a model — their variable names are not in the course "
                     "sources; use the wizard or check the quickstart")
    if "NEMOCLAW_VLLM_PORT" in env and prov != "vllm":
        warns.append("NEMOCLAW_VLLM_PORT only matters with NEMOCLAW_PROVIDER=vllm")
    if "NEMOCLAW_MODEL" in env and prov not in ("ollama", "install-vllm"):
        warns.append("NEMOCLAW_MODEL is documented for ollama; other providers pick the model elsewhere")
    tier = env.get("NEMOCLAW_POLICY_TIER", "")
    if tier and tier not in TIERS:
        errs.append(f"NEMOCLAW_POLICY_TIER={tier!r} — use one of {', '.join(TIERS)}")
    if tier == "personal":
        warns.append("personal tier: any binary may reach ports 80/443 at L4 — never on shared hardware")
    ws = env.get("NEMOCLAW_WEB_SEARCH_PROVIDER", "")
    if ws and ws not in WEB_SEARCH:
        errs.append(f"NEMOCLAW_WEB_SEARCH_PROVIDER={ws!r} — documented values are {' | '.join(WEB_SEARCH)} "
                    "(Brave uses BRAVE_API_KEY instead)")
    if ws == "tavily" and "TAVILY_API_KEY" not in env and env.get("NEMOCLAW_NON_INTERACTIVE") == "1":
        errs.append("tavily web search needs TAVILY_API_KEY in a non-interactive run (as a placeholder here)")
    name = env.get("NEMOCLAW_SANDBOX_NAME", "")
    if name and not SANDBOX_RE.match(name):
        errs.append(f"NEMOCLAW_SANDBOX_NAME={name!r} — use lowercase letters, digits and - (course rule "
                    "^[a-z0-9-]{1,40}$, the runner's)")
    for k, v in env.items():
        if k in SECRET_VARS and REAL_KEY_RE.match(v):
            errs.append(f"{k} looks like a REAL key — never type one into a line you share, paste or record "
                        "(course rule); keep a <placeholder> here")
    if p["placeholders_unquoted"]:
        warns.append(f"unquoted {', '.join(p['placeholders_unquoted'])}: bash reads < and > as redirections — replace "
                     "the placeholder with your value before you run the line")
    return errs, warns


def express_bypassed(line: str) -> bool:
    """Playbook: setting NEMOCLAW_PROVIDER (or NEMOCLAW_NO_EXPRESS=1) skips the Express Install prompt."""
    env = parse(line)["bash_env"]
    return bool(env.get("NEMOCLAW_PROVIDER")) or env.get("NEMOCLAW_NO_EXPRESS") == "1"

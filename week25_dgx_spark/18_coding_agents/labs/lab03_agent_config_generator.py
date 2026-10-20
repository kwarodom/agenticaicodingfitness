#!/usr/bin/env python3
"""Lab 18-3 · Agent config generator: every coding agent's settings, written to .runs/ only.

The playbooks' main path needs no config at all: `ollama launch <agent> --model …`
on the Spark. You need files when the agent runs somewhere else (your laptop) or
when `ollama launch` is not an option. This lab writes all of them for one
endpoint and one model, validates each one, and prints how to apply it:

  PLAYBOOK  on_spark_launch.sh      pull + `ollama launch claude|opencode|codex`  (cli-coding-agent)
  PLAYBOOK  ollama-override.conf    systemd drop-in: OLLAMA_HOST, OLLAMA_ORIGINS  (vibe-coding Step 2)
  PLAYBOOK  continue-config.yaml    Continue in VS Code → remote Ollama           (vibe-coding Step 6)
  COURSE    claude-code-direct.env  Claude Code → Ollama's Anthropic API, no `ollama launch`
  COURSE    opencode.json           OpenCode project config → Ollama's OpenAI API
  COURSE    codex-home/config.toml  Codex config, used with CODEX_HOME so ~/.codex is never touched

It NEVER writes to ~/.claude, ~/.codex, ~/.config/opencode, ~/.continue or any other
global config. Everything lands in week25/18_coding_agents/.runs/agent-config/
(gitignored), and the lab proves the global files were not modified.

Run: .venv/bin/python week25/18_coding_agents/labs/lab03_agent_config_generator.py [--host spark-a] [--tunnel]
"""
import argparse
import json
import os
import sys
import tomllib
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, api_host, check, note, result, step, table  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / ".runs" / "agent-config"
REL = "week25/18_coding_agents/.runs/agent-config"          # as seen from the repo root
PLAYBOOK_MODEL = "qwen3.6:35b-a3b-mtp-q4_K_M"
# (~/.claude.json is left out on purpose: a running Claude Code session rewrites it all the time.)
GLOBAL_CONFIGS = ["~/.claude/settings.json", "~/.codex/config.toml",
                  "~/.config/opencode/opencode.json", "~/.continue/config.yaml"]


def fingerprint(paths: list[str]) -> dict:
    """(exists, size, mtime) of each global config — os.stat only, the files are never opened."""
    fp = {}
    for p in paths:
        full = Path(os.path.expanduser(p))
        try:
            st = full.stat()
            fp[p] = (True, st.st_size, st.st_mtime_ns)
        except OSError:
            fp[p] = (False, 0, 0)
    return fp


def build(host: str, port: int, model: str, ctx: int) -> dict[str, tuple[str, str, str]]:
    """relative file → (provenance, content, how to apply)."""
    native = f"http://{host}:{port}"
    files = {}
    files["on_spark_launch.sh"] = ("PLAYBOOK", f"""#!/usr/bin/env bash
# Commands from the cli-coding-agent playbook (DGX Spark row). Run ON THE SPARK, one agent at a time.
set -euo pipefail
ollama --version
ollama pull {model}
case "${{1:-claude}}" in
  claude)   ollama launch claude   --model {model} ;;
  opencode) export PATH="$HOME/.opencode/bin:$PATH"; ollama launch opencode --model {model} ;;
  codex)    ollama launch codex    --model {model} ;;
  *) echo "usage: $0 claude|opencode|codex"; exit 2 ;;
esac
""", "scp to the Spark, then: bash on_spark_launch.sh claude   (note: `ollama launch` edits that agent's "
     "own profile on the Spark; `ollama launch <agent> --restore` puts it back)")
    files["ollama-override.conf"] = ("PLAYBOOK", f"""# /etc/systemd/system/ollama.service.d/override.conf on the Spark (vibe-coding playbook, Step 2).
# Paste via: sudo systemctl edit ollama   then: sudo systemctl daemon-reload && sudo systemctl restart ollama
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_ORIGINS=*"
# Course addition: the local-coding-agent playbook raises context to {ctx} with OLLAMA_CONTEXT_LENGTH on
# `ollama serve`. Uncomment to make that permanent for the service (uses more memory per request).
# Environment="OLLAMA_CONTEXT_LENGTH={ctx}"
""", "only if agents on OTHER machines must reach the Spark's Ollama. 0.0.0.0 + ORIGINS=* means no auth: "
     "prefer an ssh tunnel (--tunnel)")
    files["continue-config.yaml"] = ("PLAYBOOK", f"""name: Config
version: 1.0.0
schema: v1

assistants:
  - name: default
    model: OllamaRemote

models:
  - name: OllamaRemote
    provider: ollama
    model: {model}
    apiBase: {native}
    title: {model}
    roles:
      - chat
      - edit
      - autocomplete
""", "VS Code → Continue → gear → Models → gear next to Chat opens Continue's config.yaml; paste this in "
     "yourself (the lab never edits ~/.continue)")
    files["claude-code-direct.env"] = ("COURSE", f"""# Claude Code → Ollama's Anthropic-compatible API, without `ollama launch`.
# Not in the NVIDIA playbook (it recommends `ollama launch claude`, and its Troubleshooting warns that a
# direct Anthropic-compatible setup can produce prose without editing files). Lab 18-1 checks /v1/messages.
ANTHROPIC_BASE_URL={native}
ANTHROPIC_AUTH_TOKEN=ollama
ANTHROPIC_API_KEY=
""", f"one shell only, nothing persists:  ( set -a; . {REL}/claude-code-direct.env; set +a; "
     f"cd ~/cli-agent-demo && claude --model {model} )")
    files["opencode.json"] = ("COURSE", json.dumps({
        "$schema": "https://opencode.ai/config.json",
        "provider": {"ollama": {
            "npm": "@ai-sdk/openai-compatible", "name": "Ollama on DGX Spark",
            "options": {"baseURL": f"{native}/v1"},
            "models": {model: {"name": f"{model} (Spark)"}}}}}, indent=2) + "\n",
        "copy into the PROJECT root (~/cli-agent-demo/opencode.json), not ~/.config/opencode; then run "
        f"`opencode` there and pick ollama/{model}")
    files["codex-home/config.toml"] = ("COURSE", f"""# Codex CLI config for a local Ollama. Used with CODEX_HOME so ~/.codex is never touched.
# Not in the NVIDIA playbook (it uses `ollama launch codex`). Check the keys against `codex --help` for your version.
model = "{model}"
model_provider = "spark-ollama"

[model_providers.spark-ollama]
name = "Ollama on DGX Spark"
base_url = "{native}/v1"
wire_api = "responses"
""", f"CODEX_HOME=$PWD/{REL}/codex-home codex   (Lab 18-1 checks that /v1/responses answers)")
    return files


def validate(rel: str, text: str) -> str:
    if rel.endswith(".json"):
        json.loads(text)
        return "valid JSON"
    if rel.endswith(".yaml"):
        d = yaml.safe_load(text)
        assert d["models"][0]["provider"] == "ollama" and d["models"][0]["apiBase"].startswith("http")
        return "valid YAML · provider ollama"
    if rel.endswith(".toml"):
        d = tomllib.loads(text)
        assert d["model_provider"] in d["model_providers"]
        return "valid TOML · provider defined"
    if rel.endswith(".env"):
        keys = [ln.split("=", 1)[0] for ln in text.splitlines() if ln and not ln.startswith("#")]
        assert "ANTHROPIC_BASE_URL" in keys
        return f"{len(keys)} variables"
    if rel.endswith(".conf"):
        assert "[Service]" in text and "OLLAMA_HOST=" in text
        return "systemd [Service] block"
    if rel.endswith(".sh"):
        assert "ollama launch" in text
        return "bash script"
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="", help="Spark address as your laptop sees it (default: SPARK_API_HOST / "
                                                "SPARK_HOST, else the playbook's YOUR_HARDWARE_IP placeholder)")
    ap.add_argument("--tunnel", action="store_true", help="point agents at localhost:21434 (ssh -L 21434:localhost:11434)")
    ap.add_argument("--model", default=PLAYBOOK_MODEL)
    ap.add_argument("--ctx", type=int, default=64_000, help="context length (local-coding-agent playbook: 64000)")
    args = ap.parse_args()

    banner("Lab 18-3 · agent config generator — Claude Code, OpenCode, Codex, Continue",
           "writes to .runs/agent-config/ only · validates every file · never touches your global config",
           status=False)
    if args.tunnel:
        host, port = "localhost", 21434
    else:
        host, port = (args.host or api_host("a") or "YOUR_HARDWARE_IP"), 11434
    before = fingerprint(GLOBAL_CONFIGS)

    step(1, f"endpoint and model: http://{host}:{port} · {args.model}")
    if host == "YOUR_HARDWARE_IP":
        note("No Spark configured, so the files keep the playbook's YOUR_HARDWARE_IP placeholder. "
             "Rerun with --host spark-a (or --tunnel) to fill it in.")
    if args.tunnel:
        print("$ ssh -N -L 21434:localhost:11434 spark-a     # run this on the laptop first (Module 01, lab 03)")

    step(2, "write and validate each file")
    OUT.mkdir(parents=True, exist_ok=True)
    files = build(host, port, args.model, args.ctx)
    rows, how = [], []
    for rel, (prov, text, apply) in files.items():
        dest = (OUT / rel).resolve()
        assert str(dest).startswith(str(OUT.resolve()) + os.sep), f"refusing to write outside .runs: {dest}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        try:
            status = "✓ " + validate(rel, text)
        except Exception as e:  # noqa: BLE001
            status = f"✕ {type(e).__name__}: {e}"
        rows.append([rel, prov, f"{len(text.encode()):>5} B", status])
        how.append((rel, apply))
    table(rows, ["file", "source", "size", "check"])
    print(f"→ all files in {OUT.relative_to(HERE.parents[1])}/")

    step(3, "how to apply each one (you do this; the lab does not)")
    for rel, apply in how:
        print(f"│ {rel:24s} {apply}")

    step(4, "prove your global agent config was not touched")
    after = fingerprint(GLOBAL_CONFIGS)
    good = True
    for p in GLOBAL_CONFIGS:
        same = before[p] == after[p]
        good &= check(same, f"{p:34s} {'unchanged' if before[p][0] else 'absent (not created)'}",
                      f"{p} CHANGED during this lab — report this as a bug")
    note("PLAYBOOK files follow the NVIDIA playbooks word for word except the host/model you chose. "
         "COURSE files are this course's own wiring for when `ollama launch` is not an option.")
    result("Generated and validated; nothing outside .runs/ was written." if good and
           all(r[3].startswith("✓") for r in rows) else "A check failed — see the ✕ lines above.")
    if not good:
        sys.exit(1)


if __name__ == "__main__":
    main()

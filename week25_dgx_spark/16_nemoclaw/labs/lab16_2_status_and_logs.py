#!/usr/bin/env python3
"""Lab 16-2 · Status, policy and logs: what is your always-on agent doing right now?

Read-only inspection of a NemoClaw sandbox that onboarding created (default name
`my-assistant`; set NEMOCLAW_SANDBOX to use another). Every command is one the
NemoClaw playbooks document: `nemoclaw list`, `status`, `policy-list`,
`openshell policy get --full`, `openshell forward list`, a bounded slice of the logs,
and the two boundary tests the application playbook uses (public internet refused,
`inference.local` allowed).

Secrets: the dashboard URL carries a login token. This lab prints it with the token
masked, and masks token-shaped strings in the log slice too.

Run: .venv/bin/python week25/16_nemoclaw/labs/lab16_2_status_and_logs.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, cfg, note, result, sh, step, table, warn  # noqa: E402

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
SANDBOX = cfg("NEMOCLAW_SANDBOX", "my-assistant")
if not NAME_RE.match(SANDBOX):
    warn(f"NEMOCLAW_SANDBOX={SANDBOX!r} is not a valid sandbox name — using my-assistant")
    SANDBOX = "my-assistant"

# Text quoted from the NemoClaw playbooks (REFERENCE) — only where the playbook actually prints it.
REF_DASHBOARD = "http://127.0.0.1:18790/#token=<token>"
REF_EGRESS_DENIED = "curl: (56) CONNECT tunnel failed, response 403"


def mask(text: str) -> str:
    """Hide the dashboard token and anything token-shaped before printing."""
    text = re.sub(r"(#token=)[^\s&]+", r"\1•••", text)
    text = re.sub(r"(hf_|nvapi-|sk-|xox[bap]-)[A-Za-z0-9_\-]{8,}", r"\1•••", text)
    text = re.sub(r"\b\d{8,10}:[A-Za-z0-9_\-]{30,}\b", "<telegram-bot-token •••>", text)
    return re.sub(r"(?i)((?:token|api[_-]?key|password)\s*[=:]\s*)\S+", r"\1•••", text)


def seen(r, text: str) -> str:
    """What a check shows — only a live (or recorded) run is evidence about your sandbox."""
    return text if r.source in ("live", "recorded") else "— (dry: shape only)"


def run(title: str, cmd: str, **kw):
    r = sh(cmd, timeout=60, quiet=True, **kw)
    body = (mask(r.out or "") if r.source in ("live", "recorded") else (r.out or "")).rstrip()
    if r.source == "reference":
        print("◈ REFERENCE — expected output from the NVIDIA playbook (not your machine):")
    elif r.source == "example":
        print("◈ EXAMPLE — illustrative shape only (not a playbook quote, not a measurement):")
    elif r.source == "recorded":
        print("◈ RECORDED from a Spark — replay, not your machine")
    print(body or "(no output)")
    return r


banner("Lab 16-2 · status, policy and logs", f"read-only inspection of sandbox '{SANDBOX}'")

rows = []
step(1, "which sandboxes exist?")
r = run("list", "nemoclaw list", example=f"{SANDBOX}  (default)  Running")
rows.append(["nemoclaw list", r.source, seen(r, "sandbox listed" if SANDBOX in (r.out or "") else "not listed")])

step(2, "sandbox status and inference route")
r = run("status", f"nemoclaw {SANDBOX} status",
        example=f"Sandbox:   {SANDBOX}  Running\nInference: <provider> → <your-selected-model>")
rows.append(["status", r.source, seen(r, "Running" if "running" in (r.out or "").lower() else "check the output")])

step(3, "network presets applied to this sandbox")
r = run("policy-list", f"nemoclaw {SANDBOX} policy-list", example="balanced\n  local-inference\n  …")
rows.append(["policy-list", r.source, seen(r, f"{len((r.out or '').splitlines())} line(s)")])

step(4, "every host the sandbox may reach (the live policy)")
r = run("policy", f"openshell policy get {SANDBOX} --full | grep -E \"host:|port:\"",
        example="      - host: host.openshell.internal\n        port: 8000")
hosts = re.findall(r"host:\s*(\S+)", r.out or "")
rows.append(["policy get --full", r.source, seen(r, f"{len(hosts)} host(s): {', '.join(hosts[:3])}{' …' if len(hosts) > 3 else ''}")])
if r.live and len(hosts) > 12:
    warn("More than 12 allowed hosts. Least privilege: remove presets you do not use with policy-remove <preset>.")

step(5, "port forwards and the dashboard URL (token masked)")
run("forwards", "openshell forward list", example=f"18789 → {SANDBOX}  (background)")
r = run("dashboard", f"nemoclaw {SANDBOX} dashboard-url --quiet", reference=REF_DASHBOARD)
rows.append(["dashboard-url", r.source, "token masked: never paste it in chat"])

step(6, "the last 40 log lines (bounded; `logs --follow` streams forever)")
r = run("logs", f"timeout 20 nemoclaw {SANDBOX} logs 2>&1 | tail -40",
        example="[gateway] agent turn complete · 1 tool call · inference.local 200")
denials = sum(1 for ln in (r.out or "").splitlines() if re.search(r"(?i)denied|403|blocked", ln))
rows.append(["logs (tail 40)", r.source, seen(r, f"{denials} line(s) mention denied/403/blocked")])

step(7, "boundary test 1 — the public internet is refused")
r = run("egress", f"nemoclaw {SANDBOX} exec -- bash -lc 'curl -sS --max-time 5 https://example.com' 2>&1 | head -3",
        reference=REF_EGRESS_DENIED)
refused = "403" in (r.out or "") or "tunnel failed" in (r.out or "")
rows.append(["curl example.com", r.source, seen(r, "refused (good)" if refused else "NOT refused — check presets")])
if r.live and not refused:
    warn(f"The sandbox reached example.com. Run `nemoclaw {SANDBOX} policy-list` and policy-remove what you do not need.")

step(8, "boundary test 2 — the model route is allowed")
r = run("inference", f"nemoclaw {SANDBOX} exec -- bash -lc 'curl -sf https://inference.local/v1/models' | head -c 300",
        example='{"object":"list","data":[{"id":"<your-selected-model>","object":"model"}]}')
rows.append(["curl inference.local", r.source, seen(r, "model list returned" if '"data"' in (r.out or "") else "no model list")])

print()
table(rows, ["check", "source", "what it shows"])
note("Watch live: `nemoclaw <name> logs --follow` in one terminal, `openshell term` (host TUI) to approve or deny "
     "blocked network requests.")
if any(x[1] == "live" for x in rows):
    result("a healthy agent: status Running, few allowed hosts, example.com refused, inference.local answering.")
else:
    result("DRY run: REFERENCE lines are quoted from the playbooks, EXAMPLE lines are illustrative. "
           "Onboard a sandbox (Section 4) and run this again LIVE.")

#!/usr/bin/env python3
"""Lab 05-3 · Hermes traces to Langfuse without leaking keys: credential handles, and a leak checker.

Research tutorial Lab 4.3 (L4.3), citing the NemoClaw Hermes quickstart. The pattern: the real Langfuse keys
become an OpenShell credential of type `langfuse-hermes-v1`; the sandbox only ever sees placeholders, and
OpenShell substitutes the real values at egress.

  1. the flow, with placeholders only. The two `export` lines hold secrets, so the lab NEVER runs them — you type
     them in the ⌨ terminal on the Spark. `credentials add` itself only names the env vars; it goes through
     change(), and only when both vars are set in that shell (checked without printing them);
  2. rebuild + gateway restart through change(); the inside-the-sandbox step is shown, not run;
  3. on THIS laptop, for real: a checker that scans ~/.hermes/.env and config.yaml text for raw pk-lf- / sk-lf-
     keys and for key variables that do not belong there (Part 4 exercise 4's lesson), on three samples and on
     your own ~/.hermes if it exists. Findings are redacted — the checker never prints a key.

Run: .venv/bin/python week26/05_tracing/labs/lab05_3_langfuse_handles.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from clawkit import SANDBOX_RE, banner, change, note, ok, result, sh, step, table, warn  # noqa: E402

HERMES = "my-hermes"                     # the research tutorial's Hermes sandbox name
CRED = "my-hermes-langfuse"
assert SANDBOX_RE.match(HERMES) and SANDBOX_RE.match(CRED)

banner("Lab 05-3 · Langfuse credential handles", "placeholders only · the lab never sees a key · leak checker for real")

step(1, "the flow — what runs where (research tutorial Lab 4.3, citing the NemoClaw Hermes quickstart)")
FLOW = f"""# on: spark — in YOUR ⌨ terminal (the secrets never pass through a lab)
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
nemohermes credentials add {CRED} \\
  --type langfuse-hermes-v1 \\
  --credential LANGFUSE_PUBLIC_KEY \\
  --credential LANGFUSE_SECRET_KEY
unset LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY
nemohermes {HERMES} rebuild

nemohermes {HERMES} connect
hermes plugins enable observability/langfuse      # inside the sandbox
exit
nemohermes {HERMES} gateway restart"""
print(FLOW)
table([
    ["export … / unset …", "your shell on the Spark", "holds the raw keys for a few seconds", "never by a lab"],
    ["credentials add … --type langfuse-hermes-v1", "OpenShell gateway", "stores the keys as a credential handle",
     "change(), only if both vars are set"],
    ["rebuild", "NemoClaw", "recreates the sandbox with the handle attached", "change()"],
    ["plugins enable observability/langfuse", "inside the sandbox", "turns the Hermes plugin on", "you, after connect"],
    ["gateway restart", "Hermes gateway", "picks up the plugin", "change()"],
], ["line", "where it acts", "what it does", "who runs it"])

step(2, "on the Spark: is nemohermes there, and are the two key variables set? (values never printed)")
sh("command -v nemohermes || echo 'nemohermes: not on PATH'", example="/home/<you>/.local/bin/nemohermes", timeout=30)
chk = sh('for v in LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY; do if [ -n "${!v}" ]; then echo "$v=set"; '
         'else echo "$v=unset"; fi; done',
         example="LANGFUSE_PUBLIC_KEY=unset\nLANGFUSE_SECRET_KEY=unset", timeout=30)
both_set = chk.live and chk.out.count("=set") == 2
if both_set:
    change(f"nemohermes credentials add {CRED} --type langfuse-hermes-v1 "
           "--credential LANGFUSE_PUBLIC_KEY --credential LANGFUSE_SECRET_KEY",
           example=f"(credential {CRED} registered — the sandbox will see placeholders)")
else:
    print(f"$ nemohermes credentials add {CRED} --type langfuse-hermes-v1 --credential LANGFUSE_PUBLIC_KEY "
          "--credential LANGFUSE_SECRET_KEY   [NOT RUN]")
    print("→ the key variables are not set in the lab's shell (they should not be) — run the whole block above in "
          "the ⌨ terminal, with your real keys, then come back")
change(f"nemohermes {HERMES} rebuild", example=f"(sandbox {HERMES} rebuilt with credential {CRED} attached)")
print(f"$ nemohermes {HERMES} connect   → then, inside the sandbox: hermes plugins enable observability/langfuse   "
      "[you run this — interactive]")
change(f"nemohermes {HERMES} gateway restart", example="(Hermes gateway restarted)")
note("only the non-secret HERMES_LANGFUSE_BASE_URL belongs in ~/.hermes/.env — not the keys, and not copies of the "
     "placeholders (Hermes quickstart, cited in the research tutorial)")

step(3, "the leak checker — for real, on this laptop")
RAW = re.compile(r"\b(pk-lf-|sk-lf-)[A-Za-z0-9_\-]{8,}")
KEYVAR = re.compile(r"(?im)^\s*(?:export\s+)?(LANGFUSE_(?:PUBLIC|SECRET)_KEY)\s*[=:]\s*(\S*)")


def scan(name: str, text: str) -> list[tuple[str, str, str]]:
    """[(severity, where, finding)] — never returns a key, only its first 6 characters + •••."""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        for m in RAW.finditer(line):
            out.append(("HIGH", f"{name}:{i}", f"raw Langfuse key {m.group(1)}••• — exfiltrable by prompt injection; "
                                                "move it to an OpenShell credential (langfuse-hermes-v1)"))
    for m in KEYVAR.finditer(text):
        line_no = text[:m.start()].count("\n") + 1
        if not RAW.search(m.group(0)):
            out.append(("MEDIUM", f"{name}:{line_no}", f"{m.group(1)} is set here (value not shown) — keys and copied "
                                                        "placeholders do not belong in the agent's config"))
    if not out:
        out.append(("OK", name, "no Langfuse keys; only non-secret settings"))
    return out


# Fake keys, made up for this lab (they are the right SHAPE, nothing more).
SAMPLES = {
    "sample-bad/.hermes/.env": ("HERMES_LANGFUSE_BASE_URL=https://cloud.langfuse.com\n"
                                "LANGFUSE_PUBLIC_KEY=pk-lf-FAKE0000-1111-2222-3333-444455556666\n"
                                "LANGFUSE_SECRET_KEY=sk-lf-FAKE0000-1111-2222-3333-444455556666\n"),
    "sample-bad/.hermes/config.yaml": ("plugins:\n  observability/langfuse:\n    enabled: true\n"
                                       "    LANGFUSE_SECRET_KEY: sk-lf-FAKE9999-8888-7777-6666-555544443333\n"),
    "sample-copied-placeholder/.hermes/.env": ("HERMES_LANGFUSE_BASE_URL=https://cloud.langfuse.com\n"
                                               "LANGFUSE_PUBLIC_KEY=<placeholder copied from the sandbox env>\n"),
    "sample-good/.hermes/.env": "HERMES_LANGFUSE_BASE_URL=https://cloud.langfuse.com\n",
}
rows = []
for name, text in SAMPLES.items():
    rows += [list(f) for f in scan(name, text)]
mine = [p for p in (Path.home() / ".hermes" / ".env", Path.home() / ".hermes" / "config.yaml") if p.is_file()]
for p in mine:
    rows += [list(f) for f in scan(f"~/.hermes/{p.name} (yours)", p.read_text(encoding="utf-8", errors="replace"))]
for sev, where, finding in rows:
    print(f"{ {'HIGH': '✕', 'MEDIUM': '⚠', 'OK': '✓'}[sev]} {sev:6s} {where} — {finding}")
high = sum(1 for r in rows if r[0] == "HIGH")
med = sum(1 for r in rows if r[0] == "MEDIUM")
print(f"◆ {high} HIGH · {med} MEDIUM · {sum(1 for r in rows if r[0] == 'OK')} OK")
if mine:
    ok(f"also scanned your own {', '.join('~/.hermes/' + p.name for p in mine)} (read-only, redacted)")
else:
    note("no ~/.hermes on this laptop — only the samples were scanned. On the Spark, count (never print) matches "
         "inside the sandbox:")
print(f"$ grep -cE 'pk-lf-|sk-lf-' ~/.hermes/.env ~/.hermes/config.yaml   "
      f"# inside the sandbox (nemohermes {HERMES} connect)")

step(4, "why a key in /sandbox/.hermes is worse than useless (Part 4 exercise 4)")
for problem, why in (
        ("exfiltrable", "the agent can read and rewrite its own config tree; a prompt-injected agent can send it out"),
        ("not a boundary", "the docs treat /sandbox/.hermes as mutable, agent-controlled state, not isolation"),
        ("does not even work", "OpenShell expects the placeholder and substitutes at egress; a raw key skips the handle")):
    print(f"│ {problem:18s} {why}")
note("sources: the research tutorial's Part 4 solution 4, citing NemoClaw security best practices + the Hermes quickstart")
note("NAT has its own `langfuse` exporter in 1.9 (fields endpoint, public_key, secret_key; empty keys fall back to "
     "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY). It builds the Basic auth header itself — whether OpenShell's "
     "placeholder substitution reaches a header NAT base64-encodes is not covered by the course's sources: "
     "verify on your unit before relying on it.")
if high:
    warn(f"{high} raw key(s) found in the samples — exactly what the checker is for (they are fake)")
result("Keys live in an OpenShell credential; the sandbox gets placeholders; the checker flags anything else.")

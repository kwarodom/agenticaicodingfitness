#!/usr/bin/env python3
"""Lab 02-3 · Lifecycle: the read-only commands you run every day, and the ones a lab must not run.

Step 1 asks the Spark, read-only: `nemoclaw list`, `nemoclaw <s> status`, `nemoclaw <s> policy list`,
`openshell sandbox list`, `openshell forward list`. Step 2 checks the same OpenShell commands with the real CLI
on THIS laptop (it parses them; with no gateway behind it, anything that parsed fails with "Connection refused").
Step 3 sorts every L1.4 verb into read-only / change / interactive / streaming / SECRET. Step 4 shows the
change gate: a snapshot and a restart go through clawkit.change(), so they run only in LIVE mode with
🔓 Allow changes on. Step 5 explains why the lab never prints `dashboard-url` or `gateway-token` output.

Run: .venv/bin/python week26/02_first_claw/labs/lab02_3_lifecycle.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import clawkit  # noqa: E402
from clawkit import (banner, change, note, openshell_offline, parsed_ok, result, sandbox, sh, step,  # noqa: E402
                     table, warn)

S = sandbox()                                           # CLAW_SANDBOX from 🖥 Spark setup, else my-assistant
FORWARD_REF = "You should see your sandbox name with port `18789`."

banner("Lab 02-3 · lifecycle", f"sandbox `{S}` · read-only on the Spark · changes only behind 🔓")

step(1, "what exists on the Spark? (read-only)")
sh("nemoclaw list", timeout=60, example=f"""Sandboxes:
  {S}   <agent>   <provider> / <model>   <phase>        ← EXAMPLE shape""")
st = sh(f"nemoclaw {S} status", timeout=60, example=f"""Sandbox:   {S}
Phase:     <phase>
Provider:  <provider>
Model:     <model>
Inference: <health>                                     ← EXAMPLE shape, fields vary by release""")
pl = sh(f"nemoclaw {S} policy list", timeout=60, example="""tier:     balanced
presets:  npm  pypi  huggingface  brew  brave  openclaw-pricing      ← EXAMPLE shape""")
if pl.live and not pl.ok:
    note("the playbook spells it `policy-list` — trying that form (your release decides; --help wins)")
    sh(f"nemoclaw {S} policy-list", timeout=60)
sh("openshell sandbox list", timeout=60, example=f"""NAME           PHASE     AGE
{S}   <phase>   <age>                  ← EXAMPLE shape""")
sh("openshell forward list", timeout=30, reference=FORWARD_REF)

step(2, "the same OpenShell commands, parsed by the real CLI on THIS laptop")
rows = []
for args in (["sandbox", "list"], ["sandbox", "get", S], ["forward", "list"],
             ["logs", S, "--source", "sandbox", "-n", "20"], ["status"]):
    r = openshell_offline(args)
    last = next((ln.strip() for ln in reversed(r.out.splitlines()) if ln.strip()), "")
    verdict = ("✓ parsed · needs the gateway" if parsed_ok(r) else
               "✓ ran locally" if r.code == 0 else f"✕ exit {r.code}")
    rows.append(["openshell " + " ".join(args), verdict, last[:60]])
table(rows, ["command", "laptop CLI 0.0.111", "last line it printed"])
note("`forward list` is local state (forwards are processes on the machine you type on), so it answers even here. "
     "The laptop CLI is 0.0.111; NemoClaw pins " + clawkit.OPENSHELL_PINNED + " on the Spark — `--help` wins there.")

step(3, f"every L1.4 verb, sorted — what may a lab run for you? (<s> = {S})")
VERBS = [
    ("nemoclaw <s> status", "read-only", "sh() — runs in LIVE"),
    ("nemoclaw <s> policy list", "read-only", "sh() — runs in LIVE"),
    ("nemoclaw list · openshell sandbox|forward list", "read-only", "sh() — runs in LIVE"),
    ("nemoclaw <s> policy add <preset> --dry-run", "preview", "sh() — shows the merge, changes nothing"),
    ("nemoclaw <s> snapshot create --name before-change", "change", "change() — needs 🔓"),
    ("nemoclaw <s> restart · stop · start", "change", "change() — needs 🔓"),
    ("nemoclaw inference set --model … --sandbox <s>", "change (hot)", "change() — the sandbox keeps running"),
    ("nemoclaw <s> rebuild · onboard --recreate-sandbox", "change (recreates)", "change() — needs 🔓"),
    ("nemoclaw onboard --fresh --gpu", "DESTROYS + recreates", "never from a lab — ⌨ terminal only"),
    ("nemoclaw launch <s> · nemoclaw <s> connect", "interactive", "⌨ terminal only (needs a TTY)"),
    ("nemoclaw <s> logs --follow · openshell term", "streaming", "⌨ terminal only (never ends)"),
    ("nemoclaw <s> dashboard-url · gateway-token", "SECRET", "⌨ terminal only — never printed here"),
]
table([[c, k, how] for c, k, how in VERBS], ["command", "kind", "how the course runs it"])

step(4, "the change gate — snapshot first, then restart (L1.4: watch the status change)")
change(f"nemoclaw {S} snapshot create --name before-change", preview=f"nemoclaw {S} status",
       example="(the status block from step 1 — read-only preview)")
change(f"nemoclaw {S} restart", example="<restart output>        ← EXAMPLE")
after = sh(f"nemoclaw {S} status", timeout=60, example="Phase:     <phase after restart>        ← EXAMPLE shape")
if st.live and after.live:
    same = st.out.strip() == after.out.strip()
    note("status unchanged — was the restart skipped (🔓 off)?" if same else "status changed — compare the two blocks")
else:
    warn("DRY: nothing was snapshotted or restarted. In LIVE mode with 🔓 on, the two status blocks are your evidence.")

step(5, "why this lab never runs dashboard-url or gateway-token")
print("│ `nemoclaw <s> dashboard-url --quiet` prints http://127.0.0.1:18789/#token=<token>. That token is a bearer")
print("│ credential for the agent's Control UI: whoever holds it can chat with, and steer, your always-on agent.")
print("│ A lab's output is streamed into the runner's console, kept in its run history, and can be RECORDED.")
demo = "http://127.0.0.1:18789/#token=EXAMPLE-not-a-real-token"
print(f"│ clawkit redacts the shape it knows:  {demo}  →  {clawkit._redact(demo)}")
note("…but a redactor is a safety net, not a plan. Run those two commands yourself in the ⌨ terminal, on the Spark, "
     "and open the URL there (or through `ssh -L 18789:127.0.0.1:18789 <you>@<spark>` — use 127.0.0.1, not localhost).")
result("Read-only verbs run for you; changes wait for 🔓; interactive, streaming and secret verbs are yours to type.")

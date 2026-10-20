#!/usr/bin/env python3
"""Lab 02-2 · The install line: build and check the one-command NemoClaw installer, without running it.

Offline, runs anywhere. From a few choices (agent, provider, sandbox name, policy tier, web search) it composes
the non-interactive `curl -fsSL https://www.nvidia.com/nemoclaw.sh | VARS bash` line, then checks it against
the rules the research tutorial's Part 1 documents: variables on the bash side of the pipe, the third-party
notice accepted, a documented provider, a documented tier, a valid sandbox name, keys only as placeholders.
Every line is also syntax-checked by the real bash on this laptop (`bash -n` parses, it never executes).

It never runs the installer. The runner does not run `curl | bash` for you: copy the line you want and paste it
into the ⌨ terminal yourself (# on: spark). The last step asks the Spark the L1.2 pass question: does
`nemoclaw --version` answer?

Run: .venv/bin/python week26/02_first_claw/labs/lab02_2_install_line.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE))
import installkit as ik  # noqa: E402
from clawkit import banner, laptop, note, ok, result, sh, step, table, warn  # noqa: E402

RUNS = HERE / ".runs"
RUNS.mkdir(exist_ok=True)


def show(line: str) -> None:
    for i, ln in enumerate(line.splitlines()):
        print(("$ " if i == 0 else "  ") + ln + ("   [NOT RUN — paste it in the ⌨ terminal, # on: spark]" if i == 0 else ""))


def slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


banner("Lab 02-2 · the install line", "offline · builds and checks installer lines · never runs them")

step(1, "the interactive one-liner (what most people run first)")
show(f"{ik.CURL} | bash")
note("After the third-party notice, a Spark is offered Express Install: managed local vLLM, a maintained "
     "Express model, sandbox name `my-assistant`, Balanced policy (DGX Spark NemoClaw playbook). Take it once.")
note("The sandbox only exists after `nemoclaw onboard` completes — no `connect`, `launch` or `openclaw tui` before.")

step(2, "why the variables go on the bash side of the pipe — a real demo on this laptop")
note("`echo` stands in for curl: it prints a one-line 'installer' that reports what IT can see.")
script = "echo 'echo \"installer sees ACCEPT=${NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE:-<unset>}\"'"
wrong = f"NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 {script} | bash"
right = f"{script} | NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 bash"
r_wrong = laptop(["env", "-u", "NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE", "bash", "-c", wrong], show=wrong)
r_right = laptop(["env", "-u", "NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE", "bash", "-c", right], show=right)
if "<unset>" in r_wrong.out and "=1" in r_right.out:
    ok("an assignment in front of a command reaches only THAT command: curl gets it, the installer (bash) does not")

step(3, "build the lines from choices")
VARIANTS = [
    ("existing vLLM :8000", dict(agent="openclaw", provider="vllm", vllm_port=8000, sandbox="my-assistant",
                                  tier="balanced")),
    ("managed vLLM", dict(agent="openclaw", provider="install-vllm", sandbox="my-assistant", tier="balanced")),
    ("local Ollama", dict(agent="openclaw", provider="ollama", model="nemotron-3-nano:30b", sandbox="ollama-claw",
                          tier="restricted")),
    ("Hermes, locked down", dict(agent="hermes", provider="vllm", sandbox="my-hermes", tier="restricted",
                                 web_search="none")),
    ("NVIDIA Endpoints (cloud)", dict(agent="openclaw", provider="build", sandbox="my-gpt-claw")),
    ("pinned release", dict(agent="", sandbox="", non_interactive=False, accept=False, pin_tag="vX.Y.Z")),
]
lines = {}
for name, choice in VARIANTS:
    lines[name] = ik.build(**choice)
    (RUNS / f"install_{slug(name)}.sh").write_text(lines[name] + "\n", encoding="utf-8")
note(f"wrote {len(lines)} lines to week26/02_first_claw/.runs/install_*.sh — bash -n parses them, it never runs them")
syn = laptop(["bash", "-c", 'for f in install_*.sh; do bash -n "$f" && echo "syntax ok  $f"; done'], cwd=RUNS,
             show='for f in .runs/install_*.sh; do bash -n "$f" && echo "syntax ok  $f"; done')
rows = []
for name, choice in VARIANTS:
    errs, warns = ik.lint(lines[name])
    prov = choice.get("provider", "")
    where = ("◆ on this Spark" if ik.PROVIDERS[prov][2] else "⚠ leaves the Spark") if prov else "wizard decides"
    rows.append([name, prov or "(Express/wizard)", where, "skipped" if ik.express_bypassed(lines[name]) else "offered",
                 "✓" if not errs else f"✕ {len(errs)}", "✓" if f"install_{slug(name)}.sh" in syn.out else "✕"])
table(rows, ["variant", "NEMOCLAW_PROVIDER", "inference runs", "Express", "lint", "bash -n"])
print()
show(lines["existing vLLM :8000"])
print()
show(lines["Hermes, locked down"])
note("Keys are written as '<your-key>' placeholders on purpose. For the NVIDIA Endpoints line, prompts leave the "
     "Spark — the provider trust table lists local Ollama as 'no data leaves the machine', not cloud endpoints.")

step(4, "the mistakes the checker catches")
BAD = [
    ("vars before curl", f"NEMOCLAW_NON_INTERACTIVE=1 NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 {ik.CURL} | bash"),
    ("notice not accepted", f"{ik.CURL} | NEMOCLAW_NON_INTERACTIVE=1 NEMOCLAW_PROVIDER=vllm bash"),
    ("provider typo", f"{ik.CURL} | NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 NEMOCLAW_PROVIDER=local-vllm bash"),
    ("tier typo", f"{ik.CURL} | NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 NEMOCLAW_POLICY_TIER=strict bash"),
    ("bad sandbox name", f"{ik.CURL} | NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 NEMOCLAW_SANDBOX_NAME=My_Claw bash"),
    ("hermes-provider on OpenClaw", f"{ik.CURL} | NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 NEMOCLAW_AGENT=openclaw "
                                    "NEMOCLAW_PROVIDER=hermes-provider bash"),
    ("tavily with no key", f"{ik.CURL} | NEMOCLAW_NON_INTERACTIVE=1 NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 "
                           "NEMOCLAW_WEB_SEARCH_PROVIDER=tavily bash"),
    ("a real-looking key", f"{ik.CURL} | NEMOCLAW_NON_INTERACTIVE=1 NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 "
                           "NEMOCLAW_PROVIDER=build NVIDIA_INFERENCE_API_KEY=nvapi-EXAMPLEONLYnotarealkey bash"),
]
caught = 0
for name, line in BAD:
    errs, warns = ik.lint(line)
    caught += bool(errs)
    print(f"│ {name:28s} → {errs[0] if errs else 'MISSED'}")
ok(f"{caught}/{len(BAD)} broken lines refused before anyone pasted them")

step(5, "the placeholder trap — what bash does with an unedited <key> (real, in .runs/)")
trap = "TAVILY_API_KEY=<key> NEMOCLAW_POLICY_TIER=restricted env"
r = laptop(["bash", "-c", trap], cwd=RUNS, show=f"bash -c '{trap}'")
if r.code != 0:
    warn("bash read `<key>` as 'take input from a file called key' (and `>` would have written a file named after "
         "the next word). Replace every <placeholder> before you paste a line.")

step(6, "L1.2 pass check on the Spark: does the CLI answer? (read-only)")
v = sh("command -v nemoclaw && nemoclaw --version", timeout=30,
       example="/home/<you>/.local/bin/nemoclaw\nnemoclaw vX.Y.Z        ← EXAMPLE shape, not a version")
if v.live and v.ok:
    ok("nemoclaw answers — L1.2 passed")
elif v.live:
    print("✕ nemoclaw not found — paste the install line in the ⌨ terminal, or `source ~/.bashrc` if it just installed")
else:
    warn("DRY: not checked. Run the install line yourself in the ⌨ terminal (# on: spark), then re-run this lab LIVE.")
result("You can now write, check and explain an installer line. Run one yourself; the lab never will.")

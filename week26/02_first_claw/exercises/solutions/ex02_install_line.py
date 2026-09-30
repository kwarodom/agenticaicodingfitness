#!/usr/bin/env python3
"""Exercise 02 · The install line — reference solution.

Fill in the three TODOs, save, then run:
    .venv/bin/python week26/02_first_claw/exercises/ex02_install_line.py

The checker is free and offline: it parses your line (it never runs it) and asks the real bash on this laptop
to syntax-check it with `bash -n`. Stuck? Compare with exercises/solutions/.
"""
import subprocess
import sys
from pathlib import Path

MODULE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MODULE.parent / "common"))
sys.path.insert(0, str(MODULE))
import installkit as ik  # noqa: E402
from clawkit import banner, check  # noqa: E402

# ── TODO 1 ── (Part 1, exercise 2) write the non-interactive install line for a Hermes claw named `alto-hermes`
#   that uses an ALREADY-RUNNING vLLM on port 8000, Tavily web search, and the `restricted` policy tier.
#   Use a <placeholder> for the Tavily key — never a real one. Continuation lines (\) are fine.
INSTALL_LINE = """
curl -fsSL https://www.nvidia.com/nemoclaw.sh | \\
  NEMOCLAW_NON_INTERACTIVE=1 NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 \\
  NEMOCLAW_AGENT=hermes NEMOCLAW_SANDBOX_NAME=alto-hermes \\
  NEMOCLAW_PROVIDER=vllm NEMOCLAW_VLLM_PORT=8000 \\
  NEMOCLAW_WEB_SEARCH_PROVIDER=tavily TAVILY_API_KEY='<your-tavily-key>' \\
  NEMOCLAW_POLICY_TIER=restricted bash
"""

# ── TODO 2 ── (Part 1, exercise 5) NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1 must be in the environment of which
#   process for the installer to see it: "curl" or "bash"?
PROCESS_THAT_READS_IT = "bash"

# ── TODO 3 ── (Part 1, exercise 4) two ways to change the model of sandbox `my-assistant`:
#   KEEP    — changes the inference route; the sandbox keeps running
#   DESTROY — destroys and recreates the sandbox so you can pick a new model in the wizard
KEEP_SANDBOX = "nemoclaw inference set --model <model> --provider <provider> --sandbox my-assistant"
DESTROY_SANDBOX = "nemoclaw onboard --fresh --gpu"


# ─────────────────────────── checker — no need to edit below ────────────────
WANT = {
    "NEMOCLAW_NON_INTERACTIVE": "1", "NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE": "1", "NEMOCLAW_AGENT": "hermes",
    "NEMOCLAW_SANDBOX_NAME": "alto-hermes", "NEMOCLAW_PROVIDER": "vllm", "NEMOCLAW_VLLM_PORT": "8000",
    "NEMOCLAW_WEB_SEARCH_PROVIDER": "tavily", "NEMOCLAW_POLICY_TIER": "restricted",
}


def main() -> None:
    banner("Exercise 02 · the install line", "offline checker · parses your line · never runs it", status=False)
    good = True
    line = INSTALL_LINE.strip()
    errs, warns = ik.lint(line) if line else (["the line is empty"], [])
    env = ik.parse(line)["bash_env"] if line else {}
    wrong = [f"{k}={v}" for k, v in WANT.items() if env.get(k) != v]
    key = env.get("TAVILY_API_KEY", "")
    key_ok = bool(key) and not ik.REAL_KEY_RE.match(key) and ("<" in key or key.startswith("$"))
    syntax = subprocess.run(["bash", "-n", "-c", line], capture_output=True, text=True) if line else None
    good &= check(line and not errs and not wrong and key_ok and syntax.returncode == 0,
                  "install line: hermes · alto-hermes · vllm on 8000 · tavily (key as a placeholder) · restricted · "
                  "all on the bash side · bash -n OK",
                  "TODO 1: " + ("; ".join(errs[:2]) if errs else
                                f"missing or wrong: {', '.join(wrong)}" if wrong else
                                "TAVILY_API_KEY must be there, as a <placeholder>" if not key_ok else
                                f"bash -n: {syntax.stderr.strip()[:120]}"))
    for w in warns:
        print(f"⚠ {w}")
    good &= check(PROCESS_THAT_READS_IT == "bash",
                  "the bash side: the variable must be in the environment of the bash process that RUNS the script",
                  f"TODO 2: {PROCESS_THAT_READS_IT!r} — which process runs the installer, and which one only "
                  "downloads it? (Lab 02-2, step 2)")
    keep, destroy = " ".join(KEEP_SANDBOX.split()), " ".join(DESTROY_SANDBOX.split())
    good &= check("inference set" in keep and "--sandbox my-assistant" in keep and "--model" in keep
                  and "onboard" in destroy and "--fresh" in destroy,
                  "model change: `inference set` is hot (route only) · `onboard --fresh` destroys and recreates",
                  "TODO 3: KEEP should be a `nemoclaw inference set … --sandbox my-assistant` line; DESTROY an "
                  "`onboard` with the destructive flag (Section 4's verb table)")
    if not good:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)
    print("\n⚠ Verify the combination on your unit: each variable is documented on its own (quickstart, Hermes "
          "quickstart, network-policies reference), but the docs do not show this exact combination.")
    print("═ Done. Paste the line into the ⌨ terminal on a Spark (# on: spark) when you are ready — the course never will.")


if __name__ == "__main__":
    main()

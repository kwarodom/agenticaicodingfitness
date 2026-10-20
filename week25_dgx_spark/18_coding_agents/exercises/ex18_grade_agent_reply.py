#!/usr/bin/env python3
"""Exercise 18 · Grade a coding agent's reply, the way an eval harness does.

Fill in the three TODOs, save, then run:
    .venv/bin/python week25/18_coding_agents/exercises/ex18_grade_agent_reply.py

The checker is free and offline. Its fixtures follow the reply shapes lab 02 met on
a Mac's Ollama (a clean tool call, a failed edit, a tool call printed as text, a server
that refuses tools; the gemma3:4b error and the text prefix are verbatim) plus three
hand-made edge cases. Stuck? Compare with
exercises/solutions/.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from sparkkit import banner, check  # noqa: E402


# ── TODO 1 ── return the tool call's arguments as a dict, or None.
#   tool_call looks like {"function": {"name": "write_file", "arguments": ...}}.
#   `arguments` is a JSON *string* on OpenAI-style servers, but some servers send a dict.
#   Anything that does not end up as a dict (bad JSON, a list, None) → None.
def tool_args(tool_call: dict):
    return None


# ── TODO 2 ── may the agent write to this path? True only for a plain relative path inside the
#   workspace: not empty, not absolute ("/etc/passwd", "~/.bashrc"), no ".." segment, no backslashes.
def safe_path(path: str) -> bool:
    return None


# ── TODO 3 ── one verdict per reply. `reply` is {"error": str} when the server refused the request,
#   otherwise {"text": str, "tool_calls": [...]}. `tests_passed` is only meaningful for a real edit.
#   Return, checking in this order:
#     "no-tool-support"  the reply has an "error"
#     "text-tool-call"   no tool_calls, but the text contains "<tool_call>" or "<function="
#     "prose-only"       no tool_calls at all
#     "bad-args"         tool_args(first call) is None, or its "content" is not a str
#     "unsafe-path"      safe_path(args["path"]) is False
#     "edit+pass" / "edit+fail"   depending on tests_passed
def grade(reply: dict, tests_passed: bool) -> str:
    return None


# ─────────────────────────── checker — no need to edit below ────────────────
GOOD_CALL = {"function": {"name": "write_file", "arguments":
             json.dumps({"path": "math_utils.py", "content": "def add(a, b):\n    return a + b"})}}
FIXTURES = [  # (label, reply, tests_passed, expected verdict)
    ("lab 02 shape · clean tool call", {"text": "", "tool_calls": [GOOD_CALL]}, True, "edit+pass"),
    ("lab 02 shape · edit, tests fail", {"text": "", "tool_calls": [{"function": {"name": "write_file", "arguments":
        json.dumps({"path": "text_utils.py", "content": "def slugify(t):\n    return t.lower()"})}}]}, False, "edit+fail"),
    ("lab 02 · tool call printed as text", {"text": "<tool_call>\n<function=write_file>\n<parameter=path>durations.py",
                                            "tool_calls": []}, False, "text-tool-call"),
    ("lab 02 · gemma3:4b refuses tools", {"error": "registry.ollama.ai/library/gemma3:4b does not support tools"}, False,
     "no-tool-support"),
    ("hand-made · explains instead", {"text": "Here is the code: def add(a, b): return a + b", "tool_calls": []},
     False, "prose-only"),
    ("hand-made · truncated JSON", {"text": "", "tool_calls": [{"function": {"name": "write_file",
                                    "arguments": '{"path": "math_utils.py", "content": "def add('}}]}, False, "bad-args"),
    ("hand-made · escapes workspace", {"text": "", "tool_calls": [{"function": {"name": "write_file",
                                       "arguments": {"path": "../../.bashrc", "content": "echo hi"}}}]}, False,
     "unsafe-path"),
]


def main() -> None:
    banner("Exercise 18 · grade a coding agent's reply", "offline checker · free · no Spark needed", status=False)
    ok = True
    ok &= check(tool_args(GOOD_CALL) == {"path": "math_utils.py", "content": "def add(a, b):\n    return a + b"}
                and tool_args({"function": {"arguments": {"path": "a.py", "content": ""}}}) == {"path": "a.py", "content": ""}
                and tool_args({"function": {"arguments": "{not json"}}) is None
                and tool_args({"function": {"arguments": "[1, 2]"}}) is None,
                "tool_args: JSON string → dict · dict passes through · bad JSON and lists → None",
                f"TODO 1: tool_args(GOOD_CALL) should be a dict (got {tool_args(GOOD_CALL)!r})")
    cases = {"math_utils.py": True, "src/app.py": True, "": False, "/etc/passwd": False, "~/.bashrc": False,
             "../secrets.txt": False, "a/../../b.py": False, "..\\evil.py": False}
    got = {p: safe_path(p) for p in cases}
    ok &= check(got == cases, "safe_path: 2 allowed · 6 refused (empty, absolute, ~, .., nested .., backslash)",
                f"TODO 2: wrong for {[p for p in cases if got[p] != cases[p]]}")
    verdicts = [(label, grade(reply, passed), want) for label, reply, passed, want in FIXTURES]
    ok &= check(all(g == w for _, g, w in verdicts), "grade: all 7 fixtures get the right verdict",
                "TODO 3: " + "; ".join(f"{lbl} → {g!r}, want {w!r}" for lbl, g, w in verdicts if g != w)[:300])
    if not ok:
        print("\n⚠ fix the ✕ lines above, save, and run again.")
        sys.exit(1)

    print("\n▣ your grader, applied to the fixtures")
    for label, g, _ in verdicts:
        print(f"│ {label:36s} → {g}")
    edits = sum(g.startswith("edit") for _, g, _ in verdicts)
    print(f"\n═ {edits} of {len(verdicts)} replies were real edits. Only 'edit+pass' counts as success — "
          "run lab 02 with --trials 5 and grade a model before you hand it your repo.")


if __name__ == "__main__":
    main()

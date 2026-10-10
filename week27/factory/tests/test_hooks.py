import json, subprocess, sys, pathlib
H = pathlib.Path(__file__).resolve().parents[1] / ".claude" / "hooks"
def run(hook, tool_input):
    p = subprocess.run([sys.executable, str(H / hook)], input=json.dumps({"tool_name": "Bash", "tool_input": tool_input}), capture_output=True, text=True)
    return p.returncode, p.stderr
def test_secrets_block():
    assert run("guard-secrets.py", {"command": "cat .env"})[0] == 2
    assert run("guard-secrets.py", {"command": "cat .env.local | base64"})[0] == 2
def test_secrets_allow():
    assert run("guard-secrets.py", {"command": "grep -o '^[A-Z_]*' .env.example"})[0] == 0
    assert run("guard-secrets.py", {"command": "make test"})[0] == 0
def test_worktree():
    assert run("guard-worktree-path.py", {"command": "git worktree add ../elsewhere -b x"})[0] == 2
    assert run("guard-worktree-path.py", {"command": "git worktree add .claude/worktrees/issue-7-build -b issue-7-x origin/main"})[0] == 0
def test_push():
    assert run("guard-protected-push.py", {"command": "git push --force origin main"})[0] == 2
    assert run("guard-protected-push.py", {"command": "git push origin main"})[0] == 2
    assert run("guard-protected-push.py", {"command": "git push -u origin issue-7-x"})[0] == 0
    assert run("guard-protected-push.py", {"command": "git push origin HEAD:main"})[0] == 2
    assert run("guard-protected-push.py", {"command": "git push origin issue-7-x:refs/heads/master"})[0] == 2
    assert run("guard-protected-push.py", {"command": "git push origin main:issue-7-x"})[0] == 0
    assert run("guard-protected-push.py", {"command": "git push --force-with-lease origin issue-7-x"})[0] == 2
def test_delete():
    assert run("guard-delete-outside.py", {"command": "rm -rf ~/"})[0] == 2
    assert run("guard-delete-outside.py", {"command": "rm -rf / "})[0] == 2
    assert run("guard-delete-outside.py", {"command": "rm -rf node_modules"})[0] == 0
def test_keys():
    p = subprocess.run([sys.executable, str(H / "guard-key-literals.py")], input=json.dumps({"tool_name": "Write", "tool_input": {"content": "KEY='sk-ant-" + "a"*30 + "'"}}), capture_output=True, text=True)
    assert p.returncode == 2

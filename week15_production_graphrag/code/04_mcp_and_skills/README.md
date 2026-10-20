# 04 · MCP & Skills (Week 7)

> Skill: `agentic-coding-fitness:mcp-and-skills`
> **One idea:** MCP = reusable **tools**; Skills = reusable **know-how**. Stop re-coding both per project.

### Part A — Use an MCP server

```bash
pip install claude-agent-sdk      # already in requirements.txt
python 01_mcp_docs_server.py      # connects the live Claude Code docs server
```

`allowed_tools` is the safety gate — the agent can only touch servers/tools you whitelist.

### Part B — Write your own Skill

`my-first-skill/SKILL.md` is a complete, minimal skill (a commit-message writer). To actually use it:

```bash
# copy it where Claude Code looks for personal skills
cp -r my-first-skill ~/.claude/skills/commit-message
# restart Claude Code (or /reload-plugins), then stage a change and ask:
#   "write me a commit message"
```

The chain you're learning: **Skill** (one `SKILL.md`) → **Plugin** (a bundle, like this bootcamp) → **Marketplace** (a repo others install from). That's exactly how the `agentic-coding-fitness` plugin you installed is built.

**Meta moment:** you just learned the format this whole course was packaged in.

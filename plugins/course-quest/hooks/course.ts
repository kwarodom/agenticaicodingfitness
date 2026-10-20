// Every week of the Agentic Coding Fitness course, surveyed from each week's own docs
// (root README, week READMEs, tutorials). Regenerate rather than hand-edit: see README.md.

import type { Week } from './course-types'

export const WEEKS: Week[] = [
  {
    "id": "02",
    "title": "Claude API Foundations",
    "phase": 1,
    "emoji": "💬",
    "tagline": "Say hi to Claude from code: one call, a stream, then a real chat",
    "summary": "You call Claude from Python three ways: a single request, a token-by-token stream, and a multi-turn chat. A $0 mock drill proves the big idea: the API is stateless, so the messages list you resend is the bot's memory.",
    "concepts": [
      {
        "name": "messages.create",
        "explain": "One request in, one response out. You pick a model, set max_tokens, and pass a list of role/content messages."
      },
      {
        "name": "Streaming",
        "explain": "messages.stream yields text chunks as they are generated, so the answer types itself out live instead of arriving all at once."
      },
      {
        "name": "Stateless API",
        "explain": "The model remembers nothing between calls. The chat only continues because you keep the messages list and resend all of it every turn."
      },
      {
        "name": "Token usage",
        "explain": "Every response reports input and output tokens. Input grows each turn because the whole history is sent again."
      },
      {
        "name": "Mock first, swap later",
        "explain": "mock_llm.py copies the client's shape, so the same loop runs at $0 and later on real Claude with only two lines changed."
      }
    ],
    "flow": {
      "title": "The multi-turn memory loop",
      "nodes": [
        "Your message",
        "messages[]",
        "Claude API",
        "Reply",
        "Append reply"
      ],
      "loop": true,
      "caption": "Each turn the whole messages list goes to Claude, and the reply is appended so the next turn remembers it"
    },
    "needs": [
      "ANTHROPIC_API_KEY (not for the $0 drill)"
    ],
    "start": ".venv/bin/python week02_claude_api_foundations/lab/drill.py",
    "labs": [
      {
        "name": "$0 memory drill (mock LLM)",
        "path": "week02_claude_api_foundations/lab/drill.py",
        "run": ".venv/bin/python week02_claude_api_foundations/lab/drill.py",
        "mode": "$0 · no API key",
        "what": "Tell the mock your name, then ask it back. It answers because the loop resends the history."
      },
      {
        "name": "Same drill on real Claude",
        "path": "week02_claude_api_foundations/lab/real_drill.py",
        "run": ".venv/bin/python week02_claude_api_foundations/lab/real_drill.py",
        "mode": "needs API key · Haiku",
        "what": "The identical loop against claude-haiku-4-5. Watch the input token count grow each turn."
      },
      {
        "name": "Single API call",
        "path": "week02_claude_api_foundations/claudeapicall.py",
        "run": ".venv/bin/python week02_claude_api_foundations/claudeapicall.py",
        "mode": "needs API key",
        "what": "Asks Claude to explain agentic AI in 3 sentences and prints the token counts."
      },
      {
        "name": "Streaming tokens",
        "path": "week02_claude_api_foundations/claudestreamingapi.py",
        "run": ".venv/bin/python week02_claude_api_foundations/claudestreamingapi.py",
        "mode": "needs API key",
        "what": "The same question, but the answer streams onto your screen word by word."
      },
      {
        "name": "Multi-turn coding mentor",
        "path": "week02_claude_api_foundations/claudemulti_turn.py",
        "run": ".venv/bin/python week02_claude_api_foundations/claudemulti_turn.py",
        "mode": "needs API key",
        "what": "A 3-turn chat about Python decorators with a system prompt, each answer building on the last."
      }
    ],
    "quiz": [
      {
        "q": "Why does the chatbot remember your name on the second turn?",
        "options": [
          "The server stores a session",
          "The model has long-term memory",
          "The API key holds state",
          "You resend the whole messages list"
        ],
        "answer": 3,
        "why": "The API is stateless. The messages list your code keeps and resends each turn is the only memory."
      },
      {
        "q": "Which client call makes text appear token by token?",
        "options": [
          "messages.create",
          "messages.count",
          "messages.batch",
          "messages.stream"
        ],
        "answer": 3,
        "why": "claudestreamingapi.py uses client.messages.stream and prints each chunk from stream.text_stream."
      },
      {
        "q": "What changes between drill.py and real_drill.py?",
        "options": [
          "Only how the client is created",
          "The whole chat loop",
          "The message format",
          "The model must be Opus"
        ],
        "answer": 0,
        "why": "Only the import and client lines change. The loop is byte-for-byte identical, so mock and real share one surface."
      }
    ],
    "funFact": "The $0 mock fakes token counts by counting words: every word in the whole history counts as input, every word in the reply as output.",
    "readme": "README.md",
    "minutes": 45
  },
  {
    "id": "03",
    "title": "Tool Use & Smart Assistants",
    "phase": 1,
    "emoji": "🔧",
    "tagline": "Give Claude hands: weather, math, web search, even your lights",
    "summary": "You describe tools as JSON schemas and Claude decides when to call them. Your code runs each tool and sends the result back until Claude can answer in plain text. Then you build a 3-tool assistant and add tools of your own.",
    "concepts": [
      {
        "name": "Tool schemas",
        "explain": "Each tool has a name, a description and a JSON input_schema. The description is what Claude reads to decide when to use it."
      },
      {
        "name": "tool_use / tool_result",
        "explain": "Claude replies with a tool_use block; you run the function and send back a tool_result with the same tool_use_id."
      },
      {
        "name": "Loop until text",
        "explain": "Keep calling Claude while it asks for tools. When a reply has no tool calls, that text is the final answer."
      },
      {
        "name": "Safe tools",
        "explain": "toolsuse.py uses eval() with a warning; the assistant swaps in simpleeval and truncates long results to 3000 chars."
      },
      {
        "name": "Errors as results",
        "explain": "If a tool fails, return a helpful error string instead of crashing, so Claude can explain it or try something else."
      }
    ],
    "flow": {
      "title": "The tool-use round trip",
      "nodes": [
        "Question",
        "Claude",
        "tool_use",
        "Your function",
        "tool_result"
      ],
      "loop": true,
      "caption": "Claude asks for a tool, your code runs it and the result goes back, until Claude answers in text"
    },
    "needs": [
      "ANTHROPIC_API_KEY",
      "Tapo light wrapper on :8003 (optional, for lights)"
    ],
    "start": ".venv/bin/python week03_tool_use/buildsmartassistant3tools.py",
    "labs": [
      {
        "name": "3-tool smart assistant",
        "path": "week03_tool_use/buildsmartassistant3tools.py",
        "run": ".venv/bin/python week03_tool_use/buildsmartassistant3tools.py",
        "mode": "needs API key",
        "what": "Web search, calculator and file reader. Asks Thailand's GDP × 1.05, counts README lines, handles a missing file."
      },
      {
        "name": "Weather, math & smart lights",
        "path": "week03_tool_use/toolsuse.py",
        "run": ".venv/bin/python week03_tool_use/toolsuse.py",
        "mode": "needs API key + :8003 light wrapper",
        "what": "Asks Claude to turn off all lights. Without the wrapper on :8003 you see the tool's error message instead."
      },
      {
        "name": "Homework: add a 4th tool",
        "path": "week03_tool_use/hwadd4thtool.py",
        "run": "",
        "mode": "homework · empty file",
        "what": "An empty starter. Add a new tool schema plus its function to the assistant."
      },
      {
        "name": "Homework: add a 5th tool",
        "path": "week03_tool_use/hwadd5thtool.py",
        "run": "",
        "mode": "homework · empty file",
        "what": "An empty starter. Add one more tool and test a question that needs it."
      }
    ],
    "quiz": [
      {
        "q": "How do you tell Claude that the tool result belongs to its request?",
        "options": [
          "Same tool name",
          "Same model",
          "Same max_tokens",
          "Same tool_use_id"
        ],
        "answer": 3,
        "why": "Each tool_result is sent with the tool_use_id of the tool_use block it answers."
      },
      {
        "q": "When does the ask() loop stop?",
        "options": [
          "After 3 tool calls",
          "When max_tokens is hit",
          "After one call",
          "When a reply has no tool_use blocks"
        ],
        "answer": 3,
        "why": "If no tool calls are in the response, the code joins the text blocks and returns them as the answer."
      },
      {
        "q": "What does the smart assistant use for math instead of eval()?",
        "options": [
          "simpleeval",
          "numpy",
          "sympy",
          "Claude itself"
        ],
        "answer": 0,
        "why": "buildsmartassistant3tools.py calls simpleeval.simple_eval; toolsuse.py's eval() carries a warning comment."
      }
    ],
    "funFact": "The weather tool needs no API key: it geocodes the city with Open-Meteo, then fetches the current temperature and humidity.",
    "readme": "README.md",
    "minutes": 60
  },
  {
    "id": "04",
    "title": "Autonomous Pipelines & Hardware",
    "phase": 1,
    "emoji": "🚁",
    "tagline": "Chain Claude calls into a research bot, then fly a drone",
    "summary": "You chain many Claude calls into a 5-step research pipeline that searches the web, summarizes, writes a report and grades itself. Then you reach into the physical world by flying a DJI Tello drone, and try free models through OpenRouter.",
    "concepts": [
      {
        "name": "Pipeline state",
        "explain": "One state dict holds the topic, queries, sources, summaries, report, score and a timestamped log as each step runs."
      },
      {
        "name": "Chained prompts",
        "explain": "Each step is its own focused Claude call: make queries, summarize each source, synthesize, then score."
      },
      {
        "name": "Parsing JSON replies",
        "explain": "A helper strips code fences and stray text so Claude's JSON for queries and scores can be loaded safely."
      },
      {
        "name": "Self-scoring",
        "explain": "Claude rates its own report 1-10 on accuracy, completeness, clarity and usefulness, with feedback."
      },
      {
        "name": "Code meets hardware",
        "explain": "djitellopy flies the Tello: takeoff, moves, flips, spins and a square patrol, plus a live camera view."
      }
    ],
    "flow": {
      "title": "The research pipeline",
      "nodes": [
        "Topic",
        "3 queries",
        "Web search",
        "Summaries",
        "Report",
        "Self-score"
      ],
      "loop": false,
      "caption": "A topic becomes queries, search results, bullet summaries, a 300-word report and a quality score"
    },
    "needs": [
      "ANTHROPIC_API_KEY",
      "OPENROUTER_API_KEY (free-model demo)",
      "DJI Tello drone + its Wi-Fi (drone labs)"
    ],
    "start": ".venv/bin/python week04_autonomous_pipelines/pipeline.py",
    "labs": [
      {
        "name": "Autonomous research pipeline",
        "path": "week04_autonomous_pipelines/pipeline.py",
        "run": ".venv/bin/python week04_autonomous_pipelines/pipeline.py",
        "mode": "needs API key",
        "what": "Researches AI building energy optimization in Southeast Asia and saves final_report.md where you run it."
      },
      {
        "name": "Read a sample report",
        "path": "week04_autonomous_pipelines/final_report.md",
        "run": "",
        "what": "A saved pipeline output: intro, key findings, implications and conclusion."
      },
      {
        "name": "Free models via OpenRouter",
        "path": "week04_autonomous_pipelines/openrouterfreemodel.py",
        "run": ".venv/bin/python week04_autonomous_pipelines/openrouterfreemodel.py",
        "mode": "needs OPENROUTER_API_KEY",
        "what": "Asks how many r's are in 'strawberry', then 'Are you sure?' with reasoning passed back."
      },
      {
        "name": "Drone camera view",
        "path": "week04_autonomous_pipelines/droneeyes.py",
        "run": ".venv/bin/python week04_autonomous_pipelines/droneeyes.py",
        "mode": "needs Tello drone",
        "what": "Shows battery level and the Tello's live video in a window. Press q to quit."
      },
      {
        "name": "Drone stunt show",
        "path": "week04_autonomous_pipelines/dronecontrol.py",
        "run": ".venv/bin/python week04_autonomous_pipelines/dronecontrol.py",
        "mode": "needs Tello drone · flies!",
        "what": "Takeoff, dashes, four flips, a 360 spin, a square patrol, a bow, then landing."
      }
    ],
    "quiz": [
      {
        "q": "Which step in pipeline.py checks the quality of the report?",
        "options": [
          "step2_search",
          "step5_quality_score",
          "step3_summarize",
          "step4_synthesize"
        ],
        "answer": 1,
        "why": "step5_quality_score asks Claude to rate the report 1-10 on four criteria and return JSON."
      },
      {
        "q": "How many search queries does the pipeline generate?",
        "options": [
          "3",
          "1",
          "5",
          "10"
        ],
        "answer": 0,
        "why": "step1 asks Claude for exactly 3 search queries as a JSON array."
      },
      {
        "q": "What does dronecontrol.py do if the battery is under 30%?",
        "options": [
          "Refuses to fly",
          "Flies slower",
          "Skips the flips",
          "Lands at once"
        ],
        "answer": 0,
        "why": "It prints 'Battery too low for tricks!' and exits before takeoff."
      }
    ],
    "funFact": "The drone script refuses to do tricks below 30% battery, and if anything goes wrong mid-flight it calls land() right away.",
    "readme": "README.md",
    "minutes": 75
  },
  {
    "id": "05",
    "title": "The Agent Loop",
    "phase": 1,
    "emoji": "🔁",
    "tagline": "Reason, act, observe, repeat: your first real agent",
    "summary": "You build a reusable Agent class that loops: ask Claude, run the tools it picks, feed back the results, and stop when it is done or out of turns. Then you give it file and Python tools so it reviews and fixes a code file on its own.",
    "concepts": [
      {
        "name": "ReAct loop",
        "explain": "Each iteration Claude reasons, picks tools (act), and reads the tool results (observe) before deciding the next move."
      },
      {
        "name": "Stop conditions",
        "explain": "The loop ends when stop_reason is end_turn with no tool calls, or after max_iterations (10) as a safety cap."
      },
      {
        "name": "Pluggable tools",
        "explain": "Agent takes a system prompt, a tool list and a tool_executor, so the same class can power many different agents."
      },
      {
        "name": "Code review agent",
        "explain": "Four tools: read_file, write_file, run_python and run_lint (flake8). It edits a file, then runs it to verify."
      },
      {
        "name": "Guard rails",
        "explain": "run_python has a 10-second timeout, and long tool results are only printed in short previews."
      }
    ],
    "flow": {
      "title": "Reason → Act → Observe",
      "nodes": [
        "Goal",
        "Reason",
        "Act (tool)",
        "Observe",
        "Done?"
      ],
      "loop": true,
      "caption": "Tool results feed the next reasoning step until Claude ends its turn or hits 10 iterations"
    },
    "needs": [
      "ANTHROPIC_API_KEY"
    ],
    "start": ".venv/bin/python week05_agent_loop/autoagent.py",
    "labs": [
      {
        "name": "Read the sample to fix",
        "path": "week05_agent_loop/sample.py",
        "run": "",
        "what": "The small target script: circle area, list processing, a user class and an API fetch."
      },
      {
        "name": "Run the code review agent",
        "path": "week05_agent_loop/autoagent.py",
        "run": ".venv/bin/python week05_agent_loop/autoagent.py",
        "mode": "needs API key · rewrites sample.py",
        "what": "Watch each iteration's thoughts, tool calls and results as it lints, fixes and reruns sample.py."
      }
    ],
    "quiz": [
      {
        "q": "When does Agent.run() decide the job is finished?",
        "options": [
          "After any tool call",
          "stop_reason is end_turn with no tool use",
          "When flake8 passes",
          "After exactly 3 loops"
        ],
        "answer": 1,
        "why": "The DONE check returns when the response's stop_reason is end_turn and it contains no tool_use blocks."
      },
      {
        "q": "What stops a runaway agent?",
        "options": [
          "A token budget",
          "The user pressing q",
          "max_iterations",
          "A timer on Claude"
        ],
        "answer": 2,
        "why": "The loop runs at most max_iterations times (10), then returns 'Max iterations reached'."
      },
      {
        "q": "Which tool does the agent use to find style issues?",
        "options": [
          "run_lint",
          "read_file",
          "run_python",
          "write_file"
        ],
        "answer": 0,
        "why": "run_lint runs flake8 on the file and returns its output, or 'No lint issues found!'."
      }
    ],
    "funFact": "The agent really edits files: its write_file tool overwrites sample.py, so the copy in the repo is already a cleaned-up version.",
    "readme": "README.md",
    "minutes": 45
  },
  {
    "id": "06",
    "title": "Full-Stack Agent App",
    "phase": 2,
    "emoji": "🛒",
    "tagline": "A real API built with an agent, steered by CLAUDE.md",
    "summary": "A deload week: you study an e-commerce dashboard API (Express, TypeScript, Postgres, Redis) built with a coding agent. The lesson is context engineering: CLAUDE.md, AGENTS.md and memory notes tell the agent what the project is and how to behave.",
    "concepts": [
      {
        "name": "CLAUDE.md",
        "explain": "Project memory: stack, conventions, key context like JWT auth and Redis on 6379, and the common npm commands."
      },
      {
        "name": "AGENTS.md",
        "explain": "Behaviour rules: ask before new files, a review checklist, and security reminders like never commit .env files."
      },
      {
        "name": "Memory notes",
        "explain": "memory-notes.md records what is built, design decisions and the TODO list so the next session picks up fast."
      },
      {
        "name": "JWT + roles",
        "explain": "Access tokens last 15 min, refresh tokens 7 days. requireAdmin guards admin-only and write endpoints."
      },
      {
        "name": "Safe orders",
        "explain": "Order creation runs in a BEGIN/COMMIT/ROLLBACK transaction that checks stock, so nothing is oversold."
      }
    ],
    "flow": {
      "title": "A request through the API",
      "nodes": [
        "Client",
        "Express route",
        "Middleware",
        "Controller",
        "Postgres/Redis"
      ],
      "loop": false,
      "caption": "A request is routed, checked for auth and valid input, then handled with Postgres data or a Redis cache"
    },
    "needs": [
      "Node.js + npm",
      "PostgreSQL on :5432",
      "Redis on :6379",
      "week06_fullstack_agent_app/.env from week06_fullstack_agent_app/.env.example"
    ],
    "start": "cd week06_fullstack_agent_app && npm install && npm test",
    "labs": [
      {
        "name": "Project memory: CLAUDE.md",
        "path": "week06_fullstack_agent_app/CLAUDE.md",
        "run": "",
        "what": "See what the agent is told about the stack, conventions, ports and commands."
      },
      {
        "name": "Agent rules: AGENTS.md",
        "path": "week06_fullstack_agent_app/AGENTS.md",
        "run": "",
        "what": "Default behaviour, review checklist, communication style and security reminders."
      },
      {
        "name": "Session memory notes",
        "path": "week06_fullstack_agent_app/memory-notes.md",
        "run": "",
        "what": "Every endpoint, architecture decision, the schema and the TODO list in one page."
      },
      {
        "name": "Run the test suite",
        "path": "week06_fullstack_agent_app/src/__tests__",
        "run": "cd week06_fullstack_agent_app && npm test",
        "mode": "Jest + supertest",
        "what": "Auth, product, order and API tests. The integration test mocks the DB pool and Redis."
      },
      {
        "name": "Run the API + Swagger docs",
        "path": "week06_fullstack_agent_app/src/server.ts",
        "run": "cd week06_fullstack_agent_app && npm run dev",
        "port": 3000,
        "mode": "needs Postgres + Redis",
        "what": "Starts the API on :3000. Open /api/docs for Swagger UI or /health for a quick check."
      }
    ],
    "quiz": [
      {
        "q": "How long does a JWT access token last in this API?",
        "options": [
          "15 minutes",
          "5 minutes",
          "1 hour",
          "7 days"
        ],
        "answer": 0,
        "why": "memory-notes.md: access token 15 min, refresh token 7 days."
      },
      {
        "q": "Which file holds the agent's behaviour rules and review checklist?",
        "options": [
          "CLAUDE.md",
          "package.json",
          "AGENTS.md",
          "openapi.ts"
        ],
        "answer": 2,
        "why": "AGENTS.md covers default behaviour, a code review checklist, communication style and security reminders."
      },
      {
        "q": "How does order creation avoid overselling?",
        "options": [
          "Redis locks",
          "Client-side checks",
          "A cron job",
          "A database transaction"
        ],
        "answer": 3,
        "why": "It uses a pg client with BEGIN/COMMIT/ROLLBACK to check stock and decrement it atomically."
      }
    ],
    "funFact": "Both dashboard endpoints are cached in Redis for 5 minutes, with keys like dashboard:<endpoint>:<days>.",
    "readme": "week06_fullstack_agent_app/memory-notes.md",
    "minutes": 60
  },
  {
    "id": "07",
    "title": "MCP & Skills",
    "phase": 2,
    "emoji": "🔌",
    "tagline": "Plug in tools once (MCP), package know-how once (Skills)",
    "summary": "You connect Claude to ready-made MCP servers (a remote docs server and a local filesystem server) through the Claude Agent SDK, and compare that with a hand-rolled tool loop. Then you read a worked SKILL.md that packages a PR-description routine. Together they stop you re-coding the same tools and instructions in every project.",
    "concepts": [
      {
        "name": "MCP = USB-C for AI tools",
        "explain": "Write a tool server once and any agent can use it: files, docs, databases. One protocol, many servers, no custom glue per tool."
      },
      {
        "name": "Namespaced tools",
        "explain": "Agent SDK tools from a server are named mcp__<server>__<tool>. allowed_tools is your whitelist: the agent only touches what you permit."
      },
      {
        "name": "stdio vs HTTP transport",
        "explain": "A local server runs as a child process over stdin/stdout (the npx filesystem server); a remote one talks Streamable HTTP (the docs server)."
      },
      {
        "name": "Skill = SKILL.md folder",
        "explain": "Frontmatter description is the trigger; the body is the playbook. Claude loads it only when the task matches."
      },
      {
        "name": "Progressive disclosure",
        "explain": "Metadata is always loaded, the body on trigger, references/ files only on demand, so one session can hold 100+ skills."
      }
    ],
    "flow": {
      "title": "An MCP tool call",
      "nodes": [
        "Prompt",
        "Agent SDK",
        "allowed_tools",
        "MCP server",
        "Result"
      ],
      "loop": false,
      "caption": "Your prompt goes through the SDK, passes the whitelist, hits the MCP server's tool, and the answer comes back."
    },
    "needs": [
      "ANTHROPIC_API_KEY",
      "Claude Code CLI (used by the Agent SDK)",
      "Node.js + npx (filesystem server)",
      "POE_API_KEY (Poe scripts only)"
    ],
    "start": ".venv/bin/python week07_mcp_and_skills/agenttooldt.py",
    "labs": [
      {
        "name": "Plain chat loop",
        "path": "week07_mcp_and_skills/agent.py",
        "run": ".venv/bin/python week07_mcp_and_skills/agent.py",
        "mode": "needs API key · interactive",
        "what": "Chat with Claude in the terminal; the message list is its short-term memory. Type quit to exit."
      },
      {
        "name": "One tool schema, one call",
        "path": "week07_mcp_and_skills/claudeapi.py",
        "run": ".venv/bin/python week07_mcp_and_skills/claudeapi.py",
        "mode": "needs API key",
        "what": "Sends a get_current_datetime tool schema and prints the message list, including Claude's tool_use block."
      },
      {
        "name": "Hand-rolled tool loop",
        "path": "week07_mcp_and_skills/agenttooldt.py",
        "run": ".venv/bin/python week07_mcp_and_skills/agenttooldt.py",
        "mode": "needs API key",
        "what": "A while-loop runs the tool whenever stop_reason is tool_use, then prints the exact time as HH:MM:SS."
      },
      {
        "name": "Remote MCP: Claude Code docs",
        "path": "week07_mcp_and_skills/mcpserver.py",
        "run": ".venv/bin/python week07_mcp_and_skills/mcpserver.py",
        "mode": "needs API key",
        "what": "The Agent SDK connects to the HTTP docs MCP server and explains what Claude Code hooks are."
      },
      {
        "name": "Local MCP: filesystem server",
        "path": "week07_mcp_and_skills/mcpfilesystem.py",
        "run": ".venv/bin/python week07_mcp_and_skills/mcpfilesystem.py",
        "mode": "edit the folder path first",
        "what": "Starts the filesystem MCP server via npx and lists files. The folder path is hard-coded, so point it at yours."
      },
      {
        "name": "Web search + time tools via Poe",
        "path": "week07_mcp_and_skills/claudeapi-tool-search.py",
        "run": ".venv/bin/python week07_mcp_and_skills/claudeapi-tool-search.py",
        "mode": "needs POE_API_KEY",
        "what": "Claude picks between search_web (DuckDuckGo) and get_current_time to answer two test questions."
      },
      {
        "name": "Read a worked Skill",
        "path": "week07_mcp_and_skills/skill.md",
        "run": "",
        "what": "A pr-description skill: allowed-tools and model in the frontmatter, a What/Why/Changes format, extras on demand."
      }
    ],
    "quiz": [
      {
        "q": "With the Agent SDK, what is the name of a tool exposed by an MCP server called claude-code-docs?",
        "options": [
          "claude-code-docs.tool",
          "mcp__claude-code-docs__<tool>",
          "tools/claude-code-docs",
          "docs::<tool>"
        ],
        "answer": 1,
        "why": "Server tools are namespaced mcp__<server-name>__<tool>, which is why allowed_tools uses mcp__claude-code-docs__*."
      },
      {
        "q": "week07_mcp_and_skills/mcpfilesystem.py launches its server with npx as a child process. Which MCP transport is that?",
        "options": [
          "Streamable HTTP",
          "stdio",
          "WebSocket",
          "gRPC"
        ],
        "answer": 1,
        "why": "Local servers run as a child process and exchange JSON-RPC over stdin/stdout: the stdio transport."
      },
      {
        "q": "Which part of a SKILL.md decides when Claude uses the skill?",
        "options": [
          "The frontmatter description",
          "The file name",
          "The last heading",
          "The allowed-tools list"
        ],
        "answer": 0,
        "why": "The description is the trigger: write it as what it does plus when to use it, because that text is what gets matched."
      }
    ],
    "funFact": "Loading every tool from seven MCP servers can eat about 67K tokens of context before you even ask a question.",
    "readme": "plugins/agentic-coding-fitness/skills/mcp-and-skills/SKILL.md",
    "minutes": 60
  },
  {
    "id": "08",
    "title": "RAG — Knowledge Agents",
    "phase": 2,
    "emoji": "📚",
    "tagline": "Give your agent a library card so it stops making things up",
    "summary": "Guided by a step-by-step lab PDF, you use Claude Code to build retrieval-augmented generation over five seed company documents: first from scratch, then with LlamaIndex, then as an agent that decides when to search. You also test chunk sizes, teach the agent to say 'I don't know', and add metadata filters and ChromaDB persistence.",
    "concepts": [
      {
        "name": "Two-phase RAG pipeline",
        "explain": "Ingestion runs once: load, chunk, embed, store. Inference runs per query: embed the question, search top-K chunks, add them to the prompt, generate."
      },
      {
        "name": "Chunking is decision #1",
        "explain": "Small chunks are precise but lose context; big ones keep context but add noise. Start at 512 tokens, 50 overlap, top_k=5, then test."
      },
      {
        "name": "'I don't know' is a feature",
        "explain": "A strict prompt (answer only from context, cite the source, admit gaps) stops the agent hallucinating on out-of-scope questions."
      },
      {
        "name": "Vanilla vs Agentic RAG",
        "explain": "Vanilla RAG retrieves once, the same way every time. Agentic RAG makes the vector DB a tool the agent can call again, rephrase, and combine."
      },
      {
        "name": "Metadata filtering",
        "explain": "Tag documents (doc_type, dept) and filter before vector search: narrow first, then search, for far more relevant answers."
      }
    ],
    "flow": {
      "title": "Agentic RAG",
      "nodes": [
        "Question",
        "Agent",
        "Search KB",
        "Chunks",
        "Answer"
      ],
      "loop": true,
      "caption": "The agent searches, reads the chunks, and searches again if something is missing before it answers with sources."
    },
    "needs": [
      "ANTHROPIC_API_KEY",
      "Claude Code installed",
      "pip: llama-index, chromadb, sentence-transformers"
    ],
    "start": "open week08_rag_knowledge_agents/Week8_RAG_Knowledge_Agents_Lab.pdf",
    "labs": [
      {
        "name": "Set up the lab project (Steps 1–4)",
        "path": "week08_rag_knowledge_agents/Week8_RAG_Knowledge_Agents_Lab.pdf",
        "run": "",
        "mode": "build it with Claude Code",
        "what": "Install Claude Code, write a CLAUDE.md, have Claude generate 5 AltoTech seed docs, and install the RAG packages."
      },
      {
        "name": "RAG from scratch (Step 8)",
        "path": "week08_rag_knowledge_agents/Week8_RAG_Knowledge_Agents_Lab.pdf",
        "run": "",
        "mode": "build it with Claude Code",
        "what": "Chunk, embed with all-MiniLM-L6-v2, cosine-search the top 3 and answer. Relevant chunks should score above 0.4."
      },
      {
        "name": "LlamaIndex RAG + chat (Step 9)",
        "path": "week08_rag_knowledge_agents/Week8_RAG_Knowledge_Agents_Lab.pdf",
        "run": "",
        "mode": "build it with Claude Code",
        "what": "Same result in about 30 lines instead of 80, plus a 3-turn chat engine that handles follow-up questions."
      },
      {
        "name": "Chunk size + 'I don't know' (10–11)",
        "path": "week08_rag_knowledge_agents/Week8_RAG_Knowledge_Agents_Lab.pdf",
        "run": "",
        "mode": "build it with Claude Code",
        "what": "Compare 256/512/1024-token chunks, then check that out-of-scope questions get a 'no information' answer."
      },
      {
        "name": "Agentic RAG agent (Steps 12–13)",
        "path": "week08_rag_knowledge_agents/Week8_RAG_Knowledge_Agents_Lab.pdf",
        "run": "",
        "mode": "build it with Claude Code",
        "what": "A tool-use agent with search, calculate and create_report. Watch it search several times on multi-hop questions."
      },
      {
        "name": "Metadata filters + ChromaDB (14–15)",
        "path": "week08_rag_knowledge_agents/Week8_RAG_Knowledge_Agents_Lab.pdf",
        "run": "",
        "mode": "build it with Claude Code",
        "what": "Compare filtered and unfiltered answers, then persist the index so later runs load from disk instead of re-embedding."
      }
    ],
    "quiz": [
      {
        "q": "Is a smaller chunk size always better for RAG?",
        "options": [
          "Yes, smaller is always more precise",
          "No: small loses context, large adds noise",
          "Yes, if you use ChromaDB",
          "Only for PDFs"
        ],
        "answer": 1,
        "why": "256 tokens is precise but loses context; 1024 keeps context but is noisy. 512 with overlap is usually the sweet spot, but test on your data."
      },
      {
        "q": "What does similarity_top_k=5 do in index.as_query_engine()?",
        "options": [
          "Retrieves the 5 most similar chunks",
          "Retries the query 5 times",
          "Uses 5 embedding models",
          "Keeps 5 chat turns"
        ],
        "answer": 0,
        "why": "It fetches the 5 chunks closest to the query by cosine similarity. More K means more context but also more noise and cost."
      },
      {
        "q": "What is the key difference between Vanilla RAG and Agentic RAG?",
        "options": [
          "The agent decides what and when to search",
          "Agentic is always faster",
          "Vanilla uses better embeddings",
          "There is no difference"
        ],
        "answer": 0,
        "why": "In Agentic RAG the vector DB is a tool: the agent can search several times, rephrase queries, and combine sources."
      }
    ],
    "funFact": "The raw RAG demo is about 80 lines; the LlamaIndex version gives the same quality in about 30.",
    "readme": "week08_rag_knowledge_agents/Week8_RAG_Knowledge_Agents_Lab.pdf",
    "minutes": 180
  },
  {
    "id": "09",
    "title": "Multi-Agent Systems",
    "phase": 3,
    "emoji": "🐝",
    "tagline": "Three ways to make agents work as a team: chain, route, swarm",
    "summary": "You link 30 years of multi-agent research (agents, BDI, FIPA-ACL, Contract Net) to modern code by building the three core MAS patterns: a sequential Researcher → Writer → Editor pipeline, a LangGraph router for support tickets, and an asyncio swarm of five building auditors. Each pattern exists in plain Anthropic SDK code, and CrewAI and AG2 versions are there to compare.",
    "concepts": [
      {
        "name": "What makes an agent",
        "explain": "Wooldridge & Jennings (1995): a system situated in an environment, able to act autonomously to meet its design objectives."
      },
      {
        "name": "BDI architecture",
        "explain": "Beliefs are what it knows, desires what it wants, intentions the desires it has committed to and will keep pursuing."
      },
      {
        "name": "Pattern A: sequential",
        "explain": "Agents run in order, each building on the last one's output: Researcher → Writer → Editor. Good for content and document pipelines."
      },
      {
        "name": "Pattern B: graph routing",
        "explain": "A router node reads shared state and sends the ticket to billing, technical or general. In LangGraph that is add_conditional_edges."
      },
      {
        "name": "Pattern C: parallel swarm",
        "explain": "Specialists run at the same time with asyncio.gather and write to shared state (stigmergy). Total time is about the slowest single call."
      }
    ],
    "flow": {
      "title": "Pattern B: graph routing",
      "nodes": [
        "Ticket",
        "Classifier",
        "Specialist",
        "Response"
      ],
      "loop": false,
      "caption": "A ticket enters, the classifier picks a category, the conditional edge routes it to one specialist who writes the reply."
    },
    "needs": [
      "ANTHROPIC_API_KEY",
      "pip install -r week09_multi_agent_systems/requirements.txt (AG2)",
      "crewai + SERPER/ZAI/OPENROUTER keys (CrewAI version only)"
    ],
    "start": ".venv/bin/python week09_multi_agent_systems/ex1_sequential_pipeline.py",
    "labs": [
      {
        "name": "Read the theory + workshop",
        "path": "week09_multi_agent_systems/week9_2_multi_agent_theory_workshop.pdf",
        "run": "",
        "what": "Agents, BDI, FIPA-ACL, Contract Net and the three MAS patterns, plus the exercise briefs and a 10-question quiz."
      },
      {
        "name": "Check your environment",
        "path": "week09_multi_agent_systems/verify_setup.py",
        "run": ".venv/bin/python week09_multi_agent_systems/verify_setup.py",
        "mode": "makes one tiny API call",
        "what": "Checks Python, framework imports, ANTHROPIC_API_KEY and a live Claude call, then prints N/N checks passed."
      },
      {
        "name": "Ex 1: sequential pipeline",
        "path": "week09_multi_agent_systems/ex1_sequential_pipeline.py",
        "run": ".venv/bin/python week09_multi_agent_systems/ex1_sequential_pipeline.py",
        "mode": "needs API key",
        "what": "Researcher → Writer → Editor on the raw SDK, with timing per stage. Saves the post to ex1_sequential_output.md."
      },
      {
        "name": "Ex 2: LangGraph router",
        "path": "week09_multi_agent_systems/ex2_graph_routing.py",
        "run": ".venv/bin/python week09_multi_agent_systems/ex2_graph_routing.py",
        "mode": "needs API key",
        "what": "A classifier node routes support tickets to billing, technical or general specialists. Writes ex2_routing_output.md."
      },
      {
        "name": "Ex 3: parallel swarm audit",
        "path": "week09_multi_agent_systems/ex3_parallel_swarm.py",
        "run": ".venv/bin/python week09_multi_agent_systems/ex3_parallel_swarm.py",
        "mode": "needs API key",
        "what": "Five specialists audit one Bangkok office tower at once; compare parallel time with sequential and see the speedup."
      },
      {
        "name": "Ex 1 the CrewAI way",
        "path": "week09_multi_agent_systems/ex1_crewai_sequential.py",
        "run": ".venv/bin/python week09_multi_agent_systems/ex1_crewai_sequential.py",
        "mode": "needs crewai + extra keys",
        "what": "The same crew with CrewAI's Process.sequential, LLMs via z.ai and OpenRouter, and a Serper web-search tool."
      }
    ],
    "quiz": [
      {
        "q": "In LangGraph, which construct routes to different nodes based on state values?",
        "options": [
          "dynamic_edge",
          "switch_node",
          "add_conditional_edges",
          "router()"
        ],
        "answer": 2,
        "why": "add_conditional_edges() picks the next node from state, which is how the ticket classifier hands off to a specialist."
      },
      {
        "q": "Main advantage of running many LLM calls with asyncio.gather()?",
        "options": [
          "Lower token cost",
          "Better accuracy",
          "Simpler code",
          "Wall-clock speedup from concurrency"
        ],
        "answer": 3,
        "why": "LLM calls are I/O-bound, so they overlap: total time is close to the slowest single call, not the sum."
      },
      {
        "q": "A chiller agent writes its load to shared state and an optimizer reads it. Which pattern is this?",
        "options": [
          "Contract Net",
          "Stigmergy",
          "Negotiation",
          "Master/slave"
        ],
        "answer": 1,
        "why": "Stigmergy is indirect coordination through a shared environment, like the swarm's shared results list."
      }
    ],
    "funFact": "FIPA-ACL, the classic agent language, defines 22 communicative acts, such as inform, request and cfp (call for proposals).",
    "readme": "week09_multi_agent_systems/week9_2_multi_agent_theory_workshop.pdf",
    "minutes": 120
  },
  {
    "id": "10",
    "title": "Production & Observability",
    "phase": 3,
    "emoji": "🔭",
    "tagline": "Make your agents survivable, pausable, traceable and affordable",
    "summary": "Over five progressive LangGraph notebooks you build a support-ticket routing system: a first StateGraph, then a tool-calling supervisor, SQLite checkpointing with a human-approval pause, LangSmith tracing, and finally a Claude Agent SDK subagent inside a graph node. The lesson is state discipline, the thing that turns a demo into something you can run in production.",
    "concepts": [
      {
        "name": "State, nodes, edges",
        "explain": "State is a TypedDict, nodes are functions that return a state update, and edges decide what runs next. Every pattern is built from these three."
      },
      {
        "name": "Supervisor = LLM + tools",
        "explain": "The supervisor is just an LLM that calls tools, and each tool is a specialist (TECHNICAL, BILLING, GENERAL). Simple beats the library."
      },
      {
        "name": "Checkpoint + interrupt()",
        "explain": "SqliteSaver saves state after every node. interrupt() pauses for a human, and Command(resume=...) continues from that exact spot, even after a restart."
      },
      {
        "name": "Tracing with LangSmith",
        "explain": "Each trace shows inputs, outputs, tokens, latency and the state diff for every node. If you can't see it, you can't fix it or control its cost."
      },
      {
        "name": "LangGraph + Agent SDK",
        "explain": "LangGraph owns the workflow; the Claude Agent SDK runs the agent inside a node, with subagents to isolate context."
      }
    ],
    "flow": {
      "title": "Human-in-the-loop with checkpoints",
      "nodes": [
        "Ticket",
        "Draft",
        "Interrupt",
        "Human OK",
        "Send"
      ],
      "loop": false,
      "caption": "State is saved after each node; the graph pauses at the interrupt, a human approves, and it resumes to send the reply."
    },
    "needs": [
      "OPENROUTER_API_KEY (free tier)",
      "week10_production_observability venv: uv pip install -r requirements.txt",
      "ANTHROPIC_API_KEY (Ex 5 only)",
      "LANGSMITH_API_KEY (Ex 4, optional)"
    ],
    "start": "cd week10_production_observability && uv venv && uv pip install -r requirements.txt && .venv/bin/python verify_setup.py",
    "labs": [
      {
        "name": "Ex 1: Hello StateGraph",
        "path": "week10_production_observability/notebooks/01_hello_graph.py",
        "run": "cd week10_production_observability && .venv/bin/python notebooks/01_hello_graph.py",
        "mode": "free model via OpenRouter",
        "what": "A typed SupportState, a classify node and a router: your first compiled and invoked graph."
      },
      {
        "name": "Ex 2: Tool-calling supervisor",
        "path": "week10_production_observability/notebooks/02_supervisor.py",
        "run": "cd week10_production_observability && .venv/bin/python notebooks/02_supervisor.py",
        "mode": "free model via OpenRouter",
        "what": "A supervisor routes tickets to TECHNICAL, BILLING or GENERAL specialist tools. Hit a 429? Swap the model= line."
      },
      {
        "name": "Ex 3: Checkpointing + HITL",
        "path": "week10_production_observability/notebooks/03_checkpointing.py",
        "run": "cd week10_production_observability && .venv/bin/python notebooks/03_checkpointing.py",
        "mode": "free model via OpenRouter",
        "what": "draft → human_approval → send. The graph pauses at interrupt(), saves to checkpoints.db, then resumes on approve."
      },
      {
        "name": "Ex 4: LangSmith tracing",
        "path": "week10_production_observability/notebooks/04_langsmith.py",
        "run": "cd week10_production_observability && .venv/bin/python notebooks/04_langsmith.py",
        "mode": "tracing off without LangSmith key",
        "what": "Run a weak prompt and a strong prompt, then compare their traces side by side at smith.langchain.com."
      },
      {
        "name": "Ex 5: Claude Agent SDK hybrid",
        "path": "week10_production_observability/notebooks/05_hybrid_sdk.py",
        "run": "cd week10_production_observability && .venv/bin/python notebooks/05_hybrid_sdk.py",
        "mode": "skips SDK call without API key",
        "what": "A LangGraph node spawns a researcher subagent through the Claude Agent SDK to diagnose a ticket."
      },
      {
        "name": "Stuck? Reference solutions",
        "path": "week10_production_observability/solutions",
        "run": "",
        "what": "One finished solution per exercise. Copy it over your notebook file to catch up mid-class."
      }
    ],
    "quiz": [
      {
        "q": "You set state['foo'] = 'bar' inside a node, but the change never shows up. Why?",
        "options": [
          "SQLite is read-only",
          "TypedDicts are immutable",
          "You need LangSmith on",
          "Nodes must return a dict with the changed field"
        ],
        "answer": 3,
        "why": "LangGraph merges the dict a node returns into the state. Changing state in place does nothing, so return the update instead."
      },
      {
        "q": "interrupt() doesn't pause your graph. What did you most likely forget?",
        "options": [
          "Compiling with a checkpointer",
          "An END edge",
          "Setting LANGSMITH_TRACING",
          "A supervisor node"
        ],
        "answer": 0,
        "why": "Pausing needs saved state: compile the graph with a checkpointer, such as SqliteSaver, for interrupt() to work."
      },
      {
        "q": "Why might LangSmith silently show no traces?",
        "options": [
          "LANGSMITH_TRACING was set after importing LangChain",
          "Free tier has no traces",
          "OpenRouter blocks it",
          "Traces need Postgres"
        ],
        "answer": 0,
        "why": "The tracing hook installs at import time, so set the env vars first or restart your kernel."
      }
    ],
    "funFact": "Anthropic's own measurements show multi-agent systems use about 15× more tokens than chats, so cost caps matter.",
    "readme": "week10_production_observability/README.md",
    "minutes": 150
  },
  {
    "id": "11",
    "title": "Mastery — Models & Patterns",
    "phase": 3,
    "emoji": "🧭",
    "tagline": "Pick the right model, framework and pattern, then drill till it's easy",
    "summary": "Agent Forge is a single offline web page where you review the first ten weeks, compare 15 models, run a model-picking wizard, animate topologies in a playground and take a 25-question quiz. Then you work through 14 graded multi-agent drills, from a two-agent dialogue up to a hotel operations command center.",
    "concepts": [
      {
        "name": "Choose the model on purpose",
        "explain": "The wizard weighs your task, budget, latency and hosting to suggest the top 3 models. At huge scale, cost matters most."
      },
      {
        "name": "Topologies trade off",
        "explain": "Sequential, parallel, hierarchical, swarm, router or reflection: same task, different latency, cost and quality. The playground shows the difference."
      },
      {
        "name": "17 canonical patterns",
        "explain": "ReAct, Reflection, Plan-and-Execute, Router, Tool Search, RAG, Human-in-the-Loop, Guardrails and more, each with when to use it."
      },
      {
        "name": "Bounded autonomy",
        "explain": "Loops need guards: retry counters in the self-healing pipeline, iteration caps in the planner-critic loop, budget caps on cost."
      },
      {
        "name": "Framework-free MAS",
        "explain": "A small dataclass Agent plus the raw anthropic SDK reproduces CrewAI-style pipelines, so you see the coordination itself."
      }
    ],
    "flow": {
      "title": "Planner-executor-critic (Ex 13)",
      "nodes": [
        "Planner",
        "Researcher",
        "Synthesiser",
        "Critic",
        "Finalise"
      ],
      "loop": true,
      "caption": "Sub-questions go to tool calls, a draft report goes to the critic, and a low score sends it back for another pass."
    },
    "needs": [
      "A web browser (Agent Forge)",
      "ANTHROPIC_API_KEY (drills, Live API mode)",
      "autogen-agentchat + autogen-ext[anthropic] (Ex 07, 14)"
    ],
    "start": "open week11_models_and_patterns/index.html",
    "labs": [
      {
        "name": "Agent Forge mastery lab",
        "path": "week11_models_and_patterns/index.html",
        "run": "open week11_models_and_patterns/index.html",
        "mode": "$0 Sim mode · Live API optional",
        "what": "Weekly review, model atlas, wizard, capability radar, topology playground, 17 patterns and a 25-question quiz."
      },
      {
        "name": "Ex 01: Two-agent dialogue",
        "path": "week11_models_and_patterns/exercises/ex01_two_agent_dialogue/ex01_two_agent_dialogue.py",
        "run": ".venv/bin/python week11_models_and_patterns/exercises/ex01_two_agent_dialogue/ex01_two_agent_dialogue.py",
        "mode": "Beginner · needs API key",
        "what": "A Teacher and a Student Claude with separate message histories take turns explaining a Python concept."
      },
      {
        "name": "Ex 03: LangGraph router",
        "path": "week11_models_and_patterns/exercises/ex03_langgraph_router/ex03_langgraph_router.py",
        "run": ".venv/bin/python week11_models_and_patterns/exercises/ex03_langgraph_router/ex03_langgraph_router.py",
        "mode": "Beginner · needs API key",
        "what": "A StateGraph with conditional edges and START/END sends each query to one of three specialist nodes."
      },
      {
        "name": "Ex 06: Parallel fact-checker",
        "path": "week11_models_and_patterns/exercises/ex06_parallel_fact_checker/ex06_parallel_fact_checker.py",
        "run": ".venv/bin/python week11_models_and_patterns/exercises/ex06_parallel_fact_checker/ex06_parallel_fact_checker.py",
        "mode": "Easy · needs API key",
        "what": "Three checkers judge a claim at once with asyncio.gather; confidence-weighted voting decides the verdict."
      },
      {
        "name": "Ex 09: Market research swarm",
        "path": "week11_models_and_patterns/exercises/ex09_market_research_swarm/ex09_market_research_swarm.py",
        "run": ".venv/bin/python week11_models_and_patterns/exercises/ex09_market_research_swarm/ex09_market_research_swarm.py",
        "mode": "Intermediate · needs API key",
        "what": "One prerequisite step, then parallel researchers, then a synthesis. Saves a report next to the script."
      },
      {
        "name": "Ex 10: Self-healing pipeline",
        "path": "week11_models_and_patterns/exercises/ex10_langgraph_self_healing/ex10_langgraph_self_healing.py",
        "run": ".venv/bin/python week11_models_and_patterns/exercises/ex10_langgraph_self_healing/ex10_langgraph_self_healing.py",
        "mode": "Advanced · needs API key",
        "what": "Bad sensor readings go to a Claude healer and back to validation, up to 3 retries, before an alert fires."
      },
      {
        "name": "Ex 13: Autonomous research",
        "path": "week11_models_and_patterns/exercises/ex13_autonomous_research/ex13_autonomous_research.py",
        "run": ".venv/bin/python week11_models_and_patterns/exercises/ex13_autonomous_research/ex13_autonomous_research.py",
        "mode": "Expert · needs API key",
        "what": "Planner, tool-calling researcher, synthesiser and critic loop. DIAGRAMS.md shows the graph."
      },
      {
        "name": "Ex 14: Hotel ops command center",
        "path": "week11_models_and_patterns/exercises/ex14_hotel_ops_command_center/ex14_hotel_ops_command_center.py",
        "run": ".venv/bin/python week11_models_and_patterns/exercises/ex14_hotel_ops_command_center/ex14_hotel_ops_command_center.py",
        "mode": "Expert · needs AutoGen install",
        "what": "An AutoGen SelectorGroupChat of Energy, Maintenance, Guest and F&B agents with an OpsDirector."
      }
    ],
    "quiz": [
      {
        "q": "You have 12 tools. What's the best practice?",
        "options": [
          "Put all 12 schemas in the system prompt",
          "Use tool search to load the top-k",
          "Pick one and ignore the rest",
          "Switch models"
        ],
        "answer": 1,
        "why": "Retrieving only the most relevant tools per query saves tokens and improves tool selection."
      },
      {
        "q": "When is the Reflection pattern most useful?",
        "options": [
          "When you need very low latency",
          "When you have no tools",
          "When inputs are short",
          "When quality matters more than cost"
        ],
        "answer": 3,
        "why": "Generator → Critic → Revise loops spend extra tokens to get better quality. Skip them on simple tasks."
      },
      {
        "q": "What's the main risk of an autonomous agent loop?",
        "options": [
          "Too few tokens",
          "Runaway cost without a budget cap",
          "Hallucination",
          "Slow internet"
        ],
        "answer": 1,
        "why": "Always set step and spend limits: loops can run away, so observability and budgets are mandatory."
      }
    ],
    "funFact": "The 14 drills skip CrewAI on purpose: five exercises that called for it use a small dataclass Agent on the raw SDK instead.",
    "readme": "week11_models_and_patterns/README.md",
    "minutes": 150
  },
  {
    "id": "14",
    "title": "Agent Memory with Knowledge Graphs",
    "phase": 4,
    "emoji": "🕸️",
    "tagline": "Give your agents a memory that survives the restart",
    "summary": "You spin up Neo4j, build a smart-hotel knowledge graph, and learn Cypher by querying it. Then agents write every decision back into the graph as Event nodes, so the MAS has durable, queryable memory.",
    "concepts": [
      {
        "name": "Cypher patterns",
        "explain": "Neo4j's query language draws graphs in ASCII art: (a:Room)-[:HAS_FLOOR]->(b). A SQL JOIN becomes pattern matching."
      },
      {
        "name": "Knowledge graph",
        "explain": "Hotel, floors, rooms, HVAC units, sensors and agents are nodes; relationships connect them, so questions can hop across links."
      },
      {
        "name": "AgentMemory",
        "explain": "A reusable class: store_event() writes an :Event node linked to the agent and to the rooms or devices it concerns."
      },
      {
        "name": "GraphRAG",
        "explain": "GraphCypherQAChain turns a plain-English question into Cypher, runs it on the graph and has Claude answer from the rows."
      },
      {
        "name": "LLM graph builder",
        "explain": "LLMGraphTransformer reads free text like an incident note and extracts nodes and relationships straight into Neo4j."
      }
    ],
    "flow": {
      "title": "Decide, remember, recall",
      "nodes": [
        "Agent",
        "store_event",
        "Neo4j graph",
        "Cypher query",
        "Answer"
      ],
      "loop": true,
      "caption": "An agent's decision becomes an Event node; later a question becomes Cypher that recalls it."
    },
    "needs": [
      "Docker (Neo4j 5.18 + APOC + GDS on :7687)",
      "ANTHROPIC_API_KEY"
    ],
    "start": "docker compose -f week14_agent_memory_knowledge_graphs/1.docker-compose.yml up -d",
    "labs": [
      {
        "name": "Start Neo4j",
        "path": "week14_agent_memory_knowledge_graphs/1.docker-compose.yml",
        "run": "docker compose -f week14_agent_memory_knowledge_graphs/1.docker-compose.yml up -d",
        "mode": "$0 · local Docker",
        "what": "Boots Neo4j with APOC and Graph Data Science; Browser UI on :7474, Bolt on :7687."
      },
      {
        "name": "Connect and create a node",
        "path": "week14_agent_memory_knowledge_graphs/2.neo4j_connect.py",
        "run": ".venv/bin/python week14_agent_memory_knowledge_graphs/2.neo4j_connect.py",
        "mode": "$0",
        "what": "Verifies the driver connects, creates a SensorAgent node and reads it back."
      },
      {
        "name": "Build the smart-hotel graph",
        "path": "week14_agent_memory_knowledge_graphs/6.hotel_kg_builder.py",
        "run": ".venv/bin/python week14_agent_memory_knowledge_graphs/6.hotel_kg_builder.py",
        "mode": "$0",
        "what": "Builds a 300-room hotel with function halls, HVAC, sensors, budgets and 5 MAS agents."
      },
      {
        "name": "Cypher tour in the Browser",
        "path": "week14_agent_memory_knowledge_graphs/NEO4J_TUTORIAL.md",
        "run": "",
        "what": "Run MATCH, WHERE, paths, aggregation and MERGE queries on the hotel graph, then try the exercises."
      },
      {
        "name": "Ask the graph in English",
        "path": "week14_agent_memory_knowledge_graphs/3.langchain_neo4j_demo.py",
        "run": ".venv/bin/python week14_agent_memory_knowledge_graphs/3.langchain_neo4j_demo.py",
        "mode": "needs API key",
        "what": "Prints the graph schema, then Claude writes Cypher to answer 'Which agents exist?'."
      },
      {
        "name": "Text to graph with an LLM",
        "path": "week14_agent_memory_knowledge_graphs/4.llm_graph_builder.py",
        "run": ".venv/bin/python week14_agent_memory_knowledge_graphs/4.llm_graph_builder.py",
        "mode": "needs API key",
        "what": "An HVAC failure note becomes nodes and relationships; see what Claude extracted."
      },
      {
        "name": "Store and recall memories",
        "path": "week14_agent_memory_knowledge_graphs/lab1_hotel_memory.py",
        "run": ".venv/bin/python week14_agent_memory_knowledge_graphs/lab1_hotel_memory.py",
        "mode": "$0 · needs APOC",
        "what": "Logs sensor readings and HVAC actions as Events, then asks which rooms get the most actions."
      },
      {
        "name": "Hotel MAS with graph memory",
        "path": "week14_agent_memory_knowledge_graphs/lab1_hotel_mas.py",
        "run": ".venv/bin/python week14_agent_memory_knowledge_graphs/lab1_hotel_mas.py",
        "mode": "needs API key",
        "what": "A CrewAI Sensor Monitor flags hot zones; the Energy Optimizer acts and logs its decisions to the graph."
      }
    ],
    "quiz": [
      {
        "q": "How does Cypher describe the shape of data you want to match?",
        "options": [
          "SQL SELECT with JOINs",
          "JSON path strings",
          "Regular expressions",
          "ASCII-art patterns like (a)-[:REL]->(b)"
        ],
        "answer": 3,
        "why": "Cypher uses ASCII art: parentheses for nodes, arrows with brackets for relationships."
      },
      {
        "q": "In the SQL-to-Cypher mapping, what replaces a SQL JOIN?",
        "options": [
          "A node label",
          "A property",
          "Pattern matching",
          "An index"
        ],
        "answer": 2,
        "why": "Tables become labels, rows become nodes, foreign keys become relationships, and JOINs become pattern matching."
      },
      {
        "q": "What does AgentMemory.store_event() write into Neo4j?",
        "options": [
          "A CSV row",
          "A vector embedding",
          "An :Event node linked to the agent by :PERFORMED",
          "A new database"
        ],
        "answer": 2,
        "why": "It creates an Event with type, JSON data and timestamp, linked (agent)-[:PERFORMED]->(event) and to related entities."
      }
    ],
    "funFact": "The hotel graph is seeded with random.seed(42), so every learner builds the exact same 300-room hotel.",
    "readme": "week14_agent_memory_knowledge_graphs/NEO4J_TUTORIAL.md",
    "minutes": 120
  },
  {
    "id": "15",
    "title": "Production GraphRAG",
    "phase": 4,
    "emoji": "📊",
    "tagline": "GraphRAG in 7 frameworks, then prove it works with RAGAS",
    "summary": "You load a hotel IoT knowledge graph, master Cypher and GDS, and query it with GraphRAG across LangChain, LangGraph, CrewAI, ADK, FastMCP and LlamaIndex. Then you score answers with RAGAS and gate CI on quality. A second track builds a 5-agent hotel MAS over four memory layers.",
    "concepts": [
      {
        "name": "Cypher + GDS",
        "explain": "Beyond basic MATCH: APOC, full-text search and Graph Data Science algorithms enrich the hotel graph."
      },
      {
        "name": "GraphRAG frameworks",
        "explain": "The same hotel graph queried from 7 stacks, from GraphCypherQAChain to an MCP server exposing graph tools."
      },
      {
        "name": "RAGAS metrics",
        "explain": "Faithfulness, answer relevancy, context precision and context recall score answers using an LLM as judge."
      },
      {
        "name": "CI quality gate",
        "explain": "A script fails the build when any RAGAS score drops below its floor, so regressions never ship."
      },
      {
        "name": "4-layer memory",
        "explain": "L1 dict, L2 SQLite, L3 ChromaDB vectors, L4 Neo4j graph: each layer trades speed for depth."
      }
    ],
    "flow": {
      "title": "The GraphRAG improvement loop",
      "nodes": [
        "Build",
        "Evaluate",
        "Diagnose",
        "Improve"
      ],
      "loop": true,
      "caption": "RAGAS scores flow to the worst metric, one targeted fix goes in, then you re-measure."
    },
    "needs": [
      "Docker (Neo4j + APOC + GDS on :7687)",
      "ANTHROPIC_API_KEY",
      "OPENAI_API_KEY (Part 2 embeddings only)"
    ],
    "start": "docker compose -f week15_production_graphrag/kg_mastery/docker-compose.yml up -d",
    "labs": [
      {
        "name": "Start Neo4j with GDS",
        "path": "week15_production_graphrag/kg_mastery/docker-compose.yml",
        "run": "docker compose -f week15_production_graphrag/kg_mastery/docker-compose.yml up -d",
        "mode": "$0 · local Docker",
        "what": "Neo4j 5.x with APOC and Graph Data Science; Browser on :7474."
      },
      {
        "name": "Load the hotel IoT dataset",
        "path": "week15_production_graphrag/kg_mastery/part1_fundamentals/00_load_hotel_dataset.py",
        "run": ".venv/bin/python week15_production_graphrag/kg_mastery/part1_fundamentals/00_load_hotel_dataset.py",
        "mode": "$0",
        "what": "MERGEs rooms, devices, readings, alerts, staff, guests and suppliers. Safe to re-run."
      },
      {
        "name": "Cypher basics",
        "path": "week15_production_graphrag/kg_mastery/part1_fundamentals/01_cypher_basics.py",
        "run": ".venv/bin/python week15_production_graphrag/kg_mastery/part1_fundamentals/01_cypher_basics.py",
        "mode": "$0",
        "what": "Twelve core Cypher patterns run against the hotel graph, results printed."
      },
      {
        "name": "GraphRAG with LangChain",
        "path": "week15_production_graphrag/kg_mastery/part3_graphrag/01_langchain_graphcypher.py",
        "run": ".venv/bin/python week15_production_graphrag/kg_mastery/part3_graphrag/01_langchain_graphcypher.py",
        "mode": "needs API key",
        "what": "A plain-English question becomes generated Cypher, then a grounded answer."
      },
      {
        "name": "Score answers with RAGAS",
        "path": "week15_production_graphrag/kg_mastery/part4_evaluation/01_ragas_eval.py",
        "run": ".venv/bin/python week15_production_graphrag/kg_mastery/part4_evaluation/01_ragas_eval.py",
        "mode": "needs API key + ragas",
        "what": "Two hotel Q&A examples scored on the four RAGAS metrics."
      },
      {
        "name": "Improvement-loop tracker",
        "path": "week15_production_graphrag/kg_mastery/part4_evaluation/05_improvement_loop.py",
        "run": ".venv/bin/python week15_production_graphrag/kg_mastery/part4_evaluation/05_improvement_loop.py",
        "mode": "$0 · stdlib",
        "what": "Logs each iteration's four scores and the change you made, then prints the trend."
      },
      {
        "name": "Working + episodic memory",
        "path": "week15_production_graphrag/smart_hotel_mas/checkpoints/checkpoint2_memory.py",
        "run": ".venv/bin/python week15_production_graphrag/smart_hotel_mas/checkpoints/checkpoint2_memory.py",
        "mode": "$0 · no services",
        "what": "L1 dict scratchpad plus an L2 SQLite log that survives restarts."
      },
      {
        "name": "Full 5-agent hotel MAS",
        "path": "week15_production_graphrag/smart_hotel_mas/checkpoints/checkpoint6_full_mas.py",
        "run": ".venv/bin/python week15_production_graphrag/smart_hotel_mas/checkpoints/checkpoint6_full_mas.py",
        "mode": "needs API key + its own Neo4j",
        "what": "A CrewAI crew of 5 agents tied to all four memory layers over a 200-room hotel."
      }
    ],
    "quiz": [
      {
        "q": "Which RAGAS metric catches hallucinated claims not found in the retrieved context?",
        "options": [
          "faithfulness",
          "context_recall",
          "answer_relevancy",
          "context_precision"
        ],
        "answer": 0,
        "why": "Faithfulness checks every claim in the answer against the retrieved context; the target is above 0.90."
      },
      {
        "q": "What does the CI/CD gate in Part 4 do?",
        "options": [
          "Deploys Neo4j",
          "Fails the build if a metric regresses below its floor",
          "Generates test questions",
          "Rewrites bad Cypher"
        ],
        "answer": 1,
        "why": "03_cicd_gate.py uses floors (0.85/0.80/0.75/0.70) below which a change must not ship."
      },
      {
        "q": "In smart_hotel_mas, which store backs L3 semantic memory?",
        "options": [
          "ChromaDB vectors",
          "Python dict",
          "SQLite",
          "Neo4j"
        ],
        "answer": 0,
        "why": "L1 is a dict, L2 SQLite, L3 ChromaDB embeddings and L4 the Neo4j knowledge graph."
      }
    ],
    "funFact": "Quality is a trend, not a one-shot score: the improvement loop says make ONE change, re-measure, keep it only if scores rise.",
    "readme": "week15_production_graphrag/kg_mastery/README.md",
    "minutes": 180
  },
  {
    "id": "16",
    "title": "NVIDIA NeMo Agent Toolkit",
    "phase": 4,
    "emoji": "🧩",
    "tagline": "Agents from YAML: register tools, compose teams, add a human",
    "summary": "Using NVIDIA's NeMo Agent Toolkit (NAT), you register Python tools, wire them into a ReAct agent with a config.yml, and add a LlamaIndex RAG tool. Then a supervisor routes work to specialist agents, and a custom agent asks a human before plotting.",
    "concepts": [
      {
        "name": "Tool registration",
        "explain": "Each tool is a Python function registered with NAT, then switched on by name in the config's functions section."
      },
      {
        "name": "Config-driven workflow",
        "explain": "LLMs, embedders, tools and the agent type live in config.yml; run it with nat run --config_file."
      },
      {
        "name": "RAG as a tool",
        "explain": "A LlamaIndex tool chunks, embeds and retrieves a product catalog so the agent can answer about unseen products."
      },
      {
        "name": "Supervisor + specialists",
        "explain": "A supervisor routes each request to the best-fit agent: sales data, product specs or plotting."
      },
      {
        "name": "Human-in-the-loop",
        "explain": "A custom agent calls an approval tool and waits for yes or no before generating a chart."
      }
    ],
    "flow": {
      "title": "Supervisor routes the request",
      "nodes": [
        "User query",
        "Supervisor",
        "Specialist",
        "Tools",
        "Answer"
      ],
      "loop": false,
      "caption": "A question goes to the supervisor, which hands it to a specialist agent that calls its tools."
    },
    "needs": [
      "NVIDIA_API_KEY (build.nvidia.com)",
      "nvidia-nat[langchain,llama-index]",
      "jupyterlab"
    ],
    "start": "jupyter lab week16_nemo_agent_toolkit/adding_tools_to_agents.ipynb",
    "labs": [
      {
        "name": "Adding tools to agents",
        "path": "week16_nemo_agent_toolkit/adding_tools_to_agents.ipynb",
        "run": "jupyter lab week16_nemo_agent_toolkit/adding_tools_to_agents.ipynb",
        "mode": "needs NVIDIA API key",
        "what": "Create a retail sales workflow, register 3 tools, run queries, then add a LlamaIndex RAG tool."
      },
      {
        "name": "Multi-agent orchestration",
        "path": "week16_nemo_agent_toolkit/multi_agent_orchestration.ipynb",
        "run": "jupyter lab week16_nemo_agent_toolkit/multi_agent_orchestration.ipynb",
        "mode": "needs NVIDIA API key",
        "what": "A supervisor delegates to specialist agents, then a custom agent asks for HITL approval before plotting."
      }
    ],
    "quiz": [
      {
        "q": "Where do you switch tools on for a NAT agent?",
        "options": [
          "In a .env file",
          "Inside the LLM prompt",
          "In a Dockerfile",
          "In the functions section of config.yml"
        ],
        "answer": 3,
        "why": "Tools are registered in Python, then listed under functions in config.yml and given to the agent."
      },
      {
        "q": "Which command runs a NAT workflow with an input?",
        "options": [
          "python -m nat serve",
          "jupyter run",
          "nat run --config_file ... --input ...",
          "docker compose up"
        ],
        "answer": 2,
        "why": "The notebooks run every workflow with nat run --config_file=<config.yml> --input \"...\"."
      },
      {
        "q": "What does the custom data-visualization agent add over the ReAct one?",
        "options": [
          "Faster tokens",
          "A vector database",
          "Human-in-the-loop approval before plotting",
          "Automatic retries"
        ],
        "answer": 2,
        "why": "It swaps in a custom agent with an approval tool that asks the user yes or no before making a plot."
      }
    ],
    "funFact": "In a notebook the HITL prompt can't take keyboard input, so the demo pipes echo \"yes\" into nat run to play the human.",
    "readme": "week16_nemo_agent_toolkit/README.md",
    "minutes": 90
  },
  {
    "id": "17",
    "title": "Long-Running & Distributed Agents",
    "phase": 4,
    "emoji": "⏸️",
    "tagline": "Agents that sleep for days and wake up knowing exactly where they were",
    "summary": "You build a hotel work-order agent with Google ADK that pauses for a vendor part and resumes without losing context, because state lives in a durable session, not chat history. Then agents delegate across services with A2A, and auth.md keeps credentials fresh on every wake.",
    "concepts": [
      {
        "name": "Durable state machine",
        "explain": "Behaviour is driven by current_step in persisted session state, never by replaying the chat history."
      },
      {
        "name": "Persistent sessions",
        "explain": "DatabaseSessionService keeps state in SQLite locally or Cloud SQL in prod, so a 3-day pause is just a DB row."
      },
      {
        "name": "Webhook resume",
        "explain": "An external event calls Runner.run_async(state_delta=...) to wake the agent: push, not poll."
      },
      {
        "name": "A2A agent cards",
        "explain": "Agents you don't own are discovered at /.well-known/agent.json and given tasks with six lifecycle states."
      },
      {
        "name": "Store the grant",
        "explain": "auth.md tokens are short-lived, so the agent keeps the durable grant and re-mints a scoped token each time it wakes."
      }
    ],
    "flow": {
      "title": "Pause, wait, resume",
      "nodes": [
        "Work order",
        "Pause",
        "Webhook",
        "State delta",
        "Resume"
      ],
      "loop": true,
      "caption": "The agent parks at a step, an outside event pushes a state change, and it picks up exactly there."
    },
    "needs": [
      "fastapi (week17_long_running_distributed_agents/requirements.txt)",
      "google-adk (optional; live agents)",
      "Model key for live agents: ALTO_LLM_API_KEY, ANTHROPIC_API_KEY or GOOGLE_API_KEY"
    ],
    "start": ".venv/bin/python week17_long_running_distributed_agents/checkpoints/checkpoint1_state_machine.py",
    "labs": [
      {
        "name": "Durable state machine",
        "path": "week17_long_running_distributed_agents/checkpoints/checkpoint1_state_machine.py",
        "run": ".venv/bin/python week17_long_running_distributed_agents/checkpoints/checkpoint1_state_machine.py",
        "mode": "$0 · offline",
        "what": "Build the work-order state machine and ADK agent wiring; behaviour comes from current_step."
      },
      {
        "name": "Survive a restart",
        "path": "week17_long_running_distributed_agents/checkpoints/checkpoint2_durable_sessions.py",
        "run": ".venv/bin/python week17_long_running_distributed_agents/checkpoints/checkpoint2_durable_sessions.py",
        "mode": "$0 · offline",
        "what": "Pause, kill the process, resume: session state comes back from SQLite."
      },
      {
        "name": "Webhook resume",
        "path": "week17_long_running_distributed_agents/checkpoints/checkpoint3_webhook_resume.py",
        "run": ".venv/bin/python week17_long_running_distributed_agents/checkpoints/checkpoint3_webhook_resume.py",
        "mode": "$0 · needs FastAPI",
        "what": "A delivery webhook applies a state_delta and wakes the agent."
      },
      {
        "name": "In-process sub-agents",
        "path": "week17_long_running_distributed_agents/checkpoints/checkpoint4_sub_agents.py",
        "run": ".venv/bin/python week17_long_running_distributed_agents/checkpoints/checkpoint4_sub_agents.py",
        "mode": "$0 · offline",
        "what": "Delegate to focused sub_agents, and see the wall: they must be your own code."
      },
      {
        "name": "A2A cards + task lifecycle",
        "path": "week17_long_running_distributed_agents/checkpoints/checkpoint5_a2a_cards.py",
        "run": ".venv/bin/python week17_long_running_distributed_agents/checkpoints/checkpoint5_a2a_cards.py",
        "mode": "$0 · offline",
        "what": "Validate a vendor Agent Card and walk a task through all six states, incl. input-required."
      },
      {
        "name": "Fleet capstone",
        "path": "week17_long_running_distributed_agents/checkpoints/checkpoint6_fleet.py",
        "run": ".venv/bin/python week17_long_running_distributed_agents/checkpoints/checkpoint6_fleet.py",
        "mode": "$0 · offline",
        "what": "Long-running ADK plus A2A delegation, end to end."
      },
      {
        "name": "Live HR onboarding agent",
        "path": "week17_long_running_distributed_agents/hr_onboarding/tutorial_server.py",
        "run": ".venv/bin/python week17_long_running_distributed_agents/hr_onboarding/tutorial_server.py",
        "port": 8070,
        "mode": "needs a model key",
        "what": "Click-through guide; a real ADK agent pauses for a signature and a laptop, resumed by webhooks."
      },
      {
        "name": "auth.md × ADK guide",
        "path": "week17_long_running_distributed_agents/authmd_adk/tutorial_server.py",
        "run": ".venv/bin/python week17_long_running_distributed_agents/authmd_adk/tutorial_server.py",
        "port": 8080,
        "mode": "Part 1 offline · live part needs model",
        "what": "Discover, re-mint at wake, OTP approval, 403 on out-of-scope writes, then revocation."
      }
    ],
    "quiz": [
      {
        "q": "What drives the work-order agent's behaviour after a long pause?",
        "options": [
          "Replaying the chat history",
          "A cron job",
          "current_step in persisted session state",
          "The LLM's memory"
        ],
        "answer": 2,
        "why": "The durable state machine reads current_step from state, which is what lets it wake up correct days later."
      },
      {
        "q": "MCP is vertical. What is A2A?",
        "options": [
          "A database driver",
          "A vector index",
          "Horizontal: agent to another agent across services",
          "Another name for MCP"
        ],
        "answer": 2,
        "why": "MCP wires an agent down to its tools; A2A wires it across to peer agents you don't own."
      },
      {
        "q": "Why does the auth.md agent store a grant instead of a token?",
        "options": [
          "Tokens are too large",
          "Grants are encrypted",
          "ADK forbids tokens",
          "Tokens are short-lived but the agent may sleep for days"
        ],
        "answer": 3,
        "why": "The agent stores the durable grant and re-mints a fresh, scoped token every time it wakes."
      }
    ],
    "funFact": "Every week17_long_running_distributed_agents checkpoint runs offline: they build real ADK agents and walk the real A2A lifecycle, mocking only the LLM call.",
    "readme": "week17_long_running_distributed_agents/README.md",
    "minutes": 150
  },
  {
    "id": "18",
    "title": "Production Loops, Self-Evolving & Edge AI",
    "phase": 5,
    "emoji": "🧠",
    "tagline": "Ship the loop, let it learn, then unplug it from the cloud",
    "summary": "Three web apps: watch Claude Agent SDK loops run with hooks, sessions and budget caps; train an agent that writes its own skills and gets cheaper each run; and run the whole stack on a local model for $0 per token.",
    "concepts": [
      {
        "name": "Visible agent loop",
        "explain": "Every message is tagged REASON, ACT, OBSERVE or RESULT, with turn counts and real USD cost."
      },
      {
        "name": "Hooks and caps",
        "explain": "A PreToolUse hook blocks rm -rf before it runs; max_turns and max_budget_usd stop a runaway loop."
      },
      {
        "name": "Tripartite memory",
        "explain": "Episodic SessionDB, semantic MEMORY.md/USER.md and procedural SKILL.md files: what happened, what's known, how to do it."
      },
      {
        "name": "Consolidation",
        "explain": "A background loop turns session logs into facts and skills, so the same task needs fewer turns next time."
      },
      {
        "name": "Sovereign inference",
        "explain": "Point the OpenAI SDK at Ollama on localhost: same API, nothing leaves the machine, cloud cost $0."
      }
    ],
    "flow": {
      "title": "The self-evolving loop",
      "nodes": [
        "Run task",
        "SessionDB",
        "Consolidate",
        "SKILL.md",
        "Run again"
      ],
      "loop": true,
      "caption": "Each run is logged, distilled into memory and skills, and loaded by the next run."
    },
    "needs": [
      "claude CLI signed in (no API key)",
      "Ollama + gemma4:12b (sovereign_ai_edge)",
      "uv pip install -r each folder's requirements.txt"
    ],
    "start": ".venv/bin/python week18_self_evolving_sovereign_edge/agent_loop/tutorial_server.py",
    "labs": [
      {
        "name": "Agent loop web app",
        "path": "week18_self_evolving_sovereign_edge/agent_loop/tutorial_server.py",
        "run": ".venv/bin/python week18_self_evolving_sovereign_edge/agent_loop/tutorial_server.py",
        "port": 8090,
        "mode": "claude CLI · a few cents",
        "what": "Click through 10 parts and watch real loops stream, colour-coded by REASON, ACT, OBSERVE."
      },
      {
        "name": "Safety hooks",
        "path": "week18_self_evolving_sovereign_edge/agent_loop/demos/step05_hooks_safety.py",
        "run": ".venv/bin/python week18_self_evolving_sovereign_edge/agent_loop/demos/step05_hooks_safety.py",
        "mode": "claude CLI",
        "what": "A PreToolUse hook blocks rm -rf; a PostToolUse hook audits every tool call."
      },
      {
        "name": "Bounded execution",
        "path": "week18_self_evolving_sovereign_edge/agent_loop/demos/step09_production.py",
        "run": ".venv/bin/python week18_self_evolving_sovereign_edge/agent_loop/demos/step09_production.py",
        "mode": "claude CLI",
        "what": "max_budget_usd stops a runaway loop and you handle the stop subtype."
      },
      {
        "name": "Self-evolving guide",
        "path": "week18_self_evolving_sovereign_edge/self_evolving_agent/tutorial_server.py",
        "run": ".venv/bin/python week18_self_evolving_sovereign_edge/self_evolving_agent/tutorial_server.py",
        "port": 8090,
        "mode": "SIM $0 / LIVE via claude CLI",
        "what": "Walk the architecture and run checkpoints 1 to 7 with live output."
      },
      {
        "name": "Context fencing",
        "path": "week18_self_evolving_sovereign_edge/self_evolving_agent/checkpoints/checkpoint2_semantic_fencing.py",
        "run": ".venv/bin/python week18_self_evolving_sovereign_edge/self_evolving_agent/checkpoints/checkpoint2_semantic_fencing.py",
        "mode": "$0 · offline",
        "what": "Recalled memory is wrapped as reference data, not commands, to defuse injection."
      },
      {
        "name": "Watch it self-evolve",
        "path": "week18_self_evolving_sovereign_edge/self_evolving_agent/server.py",
        "run": ".venv/bin/python week18_self_evolving_sovereign_edge/self_evolving_agent/server.py",
        "port": 8088,
        "mode": "LIVE ⇄ SIM toggle",
        "what": "Run a task, consolidate, rerun: the turns and cost bar drop and a new SKILL.md appears."
      },
      {
        "name": "Sovereign edge web app",
        "path": "week18_self_evolving_sovereign_edge/sovereign_ai_edge/tutorial_server.py",
        "run": ".venv/bin/python week18_self_evolving_sovereign_edge/sovereign_ai_edge/tutorial_server.py",
        "port": 8091,
        "mode": "$0 · local Ollama",
        "what": "10 chapters of on-device runs: hardware sizing, quantization, local tool calling, hotel HVAC."
      },
      {
        "name": "Air-gap sovereignty audit",
        "path": "week18_self_evolving_sovereign_edge/sovereign_ai_edge/demos/step09_airgap_audit.py",
        "run": ".venv/bin/python week18_self_evolving_sovereign_edge/sovereign_ai_edge/demos/step09_airgap_audit.py",
        "mode": "$0 · local",
        "what": "Checks the endpoint is local and no cloud creds are set; it even flags a *-cloud model."
      }
    ],
    "quiz": [
      {
        "q": "In the agent_loop demos, what stops a runaway loop from spending too much?",
        "options": [
          "A PostToolUse hook",
          "max_budget_usd",
          "disallowed_tools",
          "Session resume"
        ],
        "answer": 1,
        "why": "Each demo is capped on max_turns and max_budget_usd; step09 shows the budget cap stopping a loop."
      },
      {
        "q": "Which store holds procedural memory in the self-evolving agent?",
        "options": [
          "SessionDB (SQLite)",
          "USER.md",
          "ChromaDB",
          "SKILL.md library"
        ],
        "answer": 3,
        "why": "Episodic = SessionDB, semantic = MEMORY.md/USER.md, procedural = the SKILL.md library."
      },
      {
        "q": "What changes to make an OpenAI SDK call run on your own hardware?",
        "options": [
          "The whole SDK",
          "Only the base_url, pointed at the local server",
          "The prompt format",
          "Nothing, it is already local"
        ],
        "answer": 1,
        "why": "Chapter 2's lesson: a drop-in OpenAI call where only base_url changes to Ollama on localhost."
      }
    ],
    "funFact": "By run 5 of an identical task, the self-evolving tutorial reports 64% fewer turns and 66% lower cost, with no human tuning.",
    "readme": "week18_self_evolving_sovereign_edge/agent_loop/README.md",
    "minutes": 180
  },
  {
    "id": "19",
    "title": "Sovereign & Self-Evolving AI on a DGX",
    "phase": 5,
    "emoji": "🖥️",
    "tagline": "Your whole agent stack on one DGX: nothing leaves the box",
    "summary": "Five web apps take the agent stack onto an NVIDIA DGX: serve models, fine-tune them, trace and judge them, give an agent memory that evolves, and front it all with one gateway. Each app runs REAL against a live endpoint or SIM with no GPU, always at $0 cloud cost.",
    "concepts": [
      {
        "name": "REAL vs SIM",
        "explain": "Each app auto-detects a live Ollama/vLLM/DGX endpoint and runs REAL; otherwise a faithful simulator teaches the same thing."
      },
      {
        "name": "Bandwidth sets tok/s",
        "explain": "LLM decode is memory-bandwidth bound, so the Spark's ~273 GB/s, not TOPS, decides how fast a model generates."
      },
      {
        "name": "LoRA / QLoRA",
        "explain": "Adapt a model to your domain on the DGX; QLoRA fits a 70B fine-tune on one Spark and your data never leaves."
      },
      {
        "name": "Laptop vs DGX commands",
        "explain": "You launch tutorials on the laptop; anything needing a GPU (docker, trtllm-serve, mpirun) is shown for the DGX."
      },
      {
        "name": "One gateway (LiteLLM)",
        "explain": "One OpenAI-compatible URL over every backend, with routing, fallbacks, hot-swap, virtual keys and budgets."
      }
    ],
    "flow": {
      "title": "The sovereign DGX journey",
      "nodes": [
        "Serve",
        "Fine-tune",
        "Observe",
        "Self-evolve",
        "Gateway"
      ],
      "loop": false,
      "caption": "A model is served, made a domain expert, traced, given evolving memory, then put behind one gateway."
    },
    "needs": [
      "each app's requirements.txt (uv pip install)",
      "optional: Ollama or a DGX for REAL mode"
    ],
    "start": ".venv/bin/python week19_sovereign_ai_dgx/sovereign_dgx/tutorial_server.py",
    "labs": [
      {
        "name": "Stop 1 · Sovereign DGX serving",
        "path": "week19_sovereign_ai_dgx/sovereign_dgx",
        "run": ".venv/bin/python week19_sovereign_ai_dgx/sovereign_dgx/tutorial_server.py",
        "port": 8092,
        "mode": "$0 · SIM/REAL auto",
        "what": "Run and serve models with Ollama, vLLM, llama.cpp; see what fits in 128 GB and NVFP4 math."
      },
      {
        "name": "Stop 2 · Fine-tune on the DGX",
        "path": "week19_sovereign_ai_dgx/dgx_finetune",
        "run": ".venv/bin/python week19_sovereign_ai_dgx/dgx_finetune/tutorial_server.py",
        "port": 8093,
        "mode": "$0 · SIM/REAL auto",
        "what": "Build a domain SFT dataset, write a QLoRA recipe, watch a simulated training run, eval before/after."
      },
      {
        "name": "Stop 3 · Observability (Phoenix + NAT)",
        "path": "week19_sovereign_ai_dgx/dgx_observability",
        "run": ".venv/bin/python week19_sovereign_ai_dgx/dgx_observability/tutorial_server.py",
        "port": 8094,
        "mode": "$0 · SIM/REAL auto",
        "what": "Trace an agent as AGENT → LLM → TOOL spans, judge traces with an LLM, rebuild it with NAT."
      },
      {
        "name": "Stop 4 · Self-evolving agent v2",
        "path": "week19_sovereign_ai_dgx/self_evolving_agent_v2",
        "run": ".venv/bin/python week19_sovereign_ai_dgx/self_evolving_agent_v2/tutorial_server.py",
        "port": 8095,
        "mode": "BRAIN=local|claude|sim",
        "what": "Swap the brain (DGX / Claude / sim), build episodic memory, consolidate it, prove warm beats cold."
      },
      {
        "name": "Stop 5 · LiteLLM gateway",
        "path": "week19_sovereign_ai_dgx/dgx_litellm",
        "run": ".venv/bin/python week19_sovereign_ai_dgx/dgx_litellm/tutorial_server.py",
        "port": 8096,
        "mode": "$0 · gateway simulated on laptop",
        "what": "Write a LiteLLM config, then explore routing, fallbacks, hot-swap, virtual keys and logging."
      }
    ],
    "quiz": [
      {
        "q": "On a DGX Spark, what mainly decides how fast a model generates tokens?",
        "options": [
          "Peak TOPS",
          "Number of CPU cores",
          "Disk speed",
          "Memory bandwidth"
        ],
        "answer": 3,
        "why": "LLM decode is memory-bandwidth bound, so memory (~273 GB/s on a Spark), not TOPS, sets tok/s."
      },
      {
        "q": "A chapter prints `docker run … nvcr.io/nvidia/…`. Where do you run it?",
        "options": [
          "On your laptop",
          "On the DGX Spark",
          "In the browser",
          "Nowhere, it is fake"
        ],
        "answer": 1,
        "why": "Rule of thumb from the README: if it needs a GPU, it's a DGX command; the app only shows it."
      },
      {
        "q": "In App 4, what does the BRAIN variable choose?",
        "options": [
          "Where memory is stored",
          "Who thinks: local model, Claude or sim",
          "Which port to use",
          "The tunnel type"
        ],
        "answer": 1,
        "why": "BRAIN picks who thinks; DGX_CONN picks how you reach the local model (LAN, tunnel or cloud)."
      }
    ],
    "funFact": "Two DGX Sparks linked over 200GbE give 256 GB, enough to run a 235B model with tensor parallelism.",
    "readme": "week19_sovereign_ai_dgx/README.md",
    "minutes": 180
  },
  {
    "id": "21",
    "title": "Physical AI & Digital Twins for Buildings",
    "phase": 5,
    "emoji": "🏢",
    "tagline": "Give your agents a body: a living twin of the building they run",
    "summary": "Thirteen SIM/REAL apps build a digital twin of a real building on OpenUSD and the NVIDIA Physical AI stack. You go from BIM to a USD scene, bind live telemetry, add energy physics for what-ifs, then let agents learn controls safely in two capstones.",
    "concepts": [
      {
        "name": "Twin = 4 layers",
        "explain": "A digital twin is scene (OpenUSD) + state (live data) + simulation (physics) + agents. A 3D model alone is not a twin."
      },
      {
        "name": "Three-computer model",
        "explain": "DGX trains, OVX/Omniverse simulates, Jetson/AGX deploys: NVIDIA's map of Physical AI."
      },
      {
        "name": "OpenUSD composition",
        "explain": "Stage, prim, attribute; layers, references, payloads and variants stack non-destructive opinions."
      },
      {
        "name": "GlobalId join key",
        "explain": "USD geometry, a Brick/REC graph and a time-series store are joined on the BIM GUID."
      },
      {
        "name": "Calibrate, then trust",
        "explain": "ASHRAE Guideline 14 gate (NMBE ≤ ±10%, CV(RMSE) ≤ 30% hourly) before a twin's what-ifs count."
      }
    ],
    "flow": {
      "title": "Building a living twin",
      "nodes": [
        "BIM / IFC",
        "USD scene",
        "Live state",
        "Simulation",
        "Agents"
      ],
      "loop": true,
      "caption": "Geometry becomes a USD stage, telemetry binds to it, physics answers what-if, agents act and feed back."
    },
    "needs": [
      "each app's requirements.txt (uv pip install)",
      "optional: usd-core, eppy, Ollama for REAL"
    ],
    "start": ".venv/bin/python week21_physical_ai_digital_twins/01_physical_ai_landscape/tutorial_server.py",
    "labs": [
      {
        "name": "01 · Physical AI landscape",
        "path": "week21_physical_ai_digital_twins/01_physical_ai_landscape",
        "run": ".venv/bin/python week21_physical_ai_digital_twins/01_physical_ai_landscape/tutorial_server.py",
        "port": 8200,
        "mode": "$0 · SIM/REAL auto",
        "what": "The three-computer model, Omniverse as libraries, Cosmos, and the Mega and DSX blueprints."
      },
      {
        "name": "02 · OpenUSD foundations",
        "path": "week21_physical_ai_digital_twins/02_openusd_foundations",
        "run": ".venv/bin/python week21_physical_ai_digital_twins/02_openusd_foundations/tutorial_server.py",
        "port": 8201,
        "mode": "$0 · REAL with usd-core",
        "what": "Stages, prims, layers and composition from zero; runs REAL with usd-core, no GPU."
      },
      {
        "name": "03 · BIM → USD",
        "path": "week21_physical_ai_digital_twins/03_bim_to_usd",
        "run": ".venv/bin/python week21_physical_ai_digital_twins/03_bim_to_usd/tutorial_server.py",
        "port": 8202,
        "mode": "$0 · SIM/REAL auto",
        "what": "LOD 100–500, the IFC spatial tree as a USD hierarchy, and GUID preservation as the join key."
      },
      {
        "name": "05 · Live twin binding",
        "path": "week21_physical_ai_digital_twins/05_live_twin_binding",
        "run": ".venv/bin/python week21_physical_ai_digital_twins/05_live_twin_binding/tutorial_server.py",
        "port": 8204,
        "mode": "$0 · SIM/REAL auto",
        "what": "BACnet/Modbus → MQTT → USD attributes; USD + graph + TSDB joined for spatial-temporal queries."
      },
      {
        "name": "07 · What-if scenarios",
        "path": "week21_physical_ai_digital_twins/07_whatif_scenarios",
        "run": ".venv/bin/python week21_physical_ai_digital_twins/07_whatif_scenarios/tutorial_server.py",
        "port": 8206,
        "mode": "$0 · REAL with eppy",
        "what": "Parametric sweeps, the Guideline 14 calibration gate, FMU co-simulation and Alfalfa."
      },
      {
        "name": "09 · RL building controls",
        "path": "week21_physical_ai_digital_twins/09_rl_building_controls",
        "run": ".venv/bin/python week21_physical_ai_digital_twins/09_rl_building_controls/tutorial_server.py",
        "port": 8208,
        "mode": "$0 · SIM/REAL auto",
        "what": "Sinergym, CityLearn and BOPTEST; PPO/SAC vs rules vs Guideline 36, and why tuned MPC often wins."
      },
      {
        "name": "12 · Capstone: twin hotel",
        "path": "week21_physical_ai_digital_twins/12_capstone_twin_hotel",
        "run": ".venv/bin/python week21_physical_ai_digital_twins/12_capstone_twin_hotel/tutorial_server.py",
        "port": 8211,
        "mode": "$0 · SIM/REAL auto",
        "what": "AltoTech Grand Bangkok: scene, live telemetry, calibrated model, what-if console and agents in one."
      },
      {
        "name": "13 · Capstone: smart city",
        "path": "week21_physical_ai_digital_twins/13_capstone_smart_city",
        "run": ".venv/bin/python week21_physical_ai_digital_twins/13_capstone_smart_city/tutorial_server.py",
        "port": 8212,
        "mode": "$0 · SIM/REAL auto",
        "what": "A twin of twins: the hotel referenced into a city stage, camera VLM alerts, district what-ifs."
      }
    ],
    "quiz": [
      {
        "q": "Which four layers make a digital twin in Week 21?",
        "options": [
          "Model, harness, runtime, flywheel",
          "BIM, CAD, GIS, BMS",
          "Scene, state, simulation, agents",
          "Data, cloud, API, UI"
        ],
        "answer": 2,
        "why": "The week's through-line: Digital Twin = SCENE (OpenUSD) + STATE (live data) + SIMULATION (physics) + AGENTS."
      },
      {
        "q": "What joins USD geometry, the semantic graph and the telemetry store?",
        "options": [
          "The file name",
          "The floor number",
          "The BIM GlobalId (GUID)",
          "The MQTT topic"
        ],
        "answer": 2,
        "why": "App 05's three-store architecture joins USD, Brick/REC graph and TSDB on GlobalId; App 03 teaches preserving it."
      },
      {
        "q": "In the three-computer model, what does Jetson/AGX do?",
        "options": [
          "Trains the models",
          "Runs the simulation",
          "Deploys (acts at the edge)",
          "Stores the BIM"
        ],
        "answer": 2,
        "why": "DGX trains, OVX/Omniverse simulates, Jetson/AGX deploys."
      }
    ],
    "funFact": "LOD 500 is a field-verification claim about the as-built building, not 'more polygons' or the prettiest mesh.",
    "readme": "week21_physical_ai_digital_twins/README.md",
    "minutes": 240
  },
  {
    "id": "23",
    "title": "The Open Superintelligence Stack (NVIDIA)",
    "phase": 6,
    "emoji": "🧱",
    "tagline": "Agent = Model + Harness, built on NVIDIA's open stack",
    "summary": "Twelve SIM/REAL apps walk NVIDIA's open stack for long-running, self-improving, sovereign agents: Nemotron models, NIM and Dynamo serving, Skills, AI-Q, NemoClaw, OpenShell, Relay and the data flywheel. A hub maps it all, a Lab Runner runs the hands-on labs, and a capstone runs a smart hotel.",
    "concepts": [
      {
        "name": "Agent = Model + Harness",
        "explain": "The model reasons; the harness (context, tools, memory, security, observability, self-improvement) makes it an agent."
      },
      {
        "name": "Nemotron 3 family",
        "explain": "Open hybrid Mamba-Transformer MoE models (Nano, Super, Ultra) with 1M context, reasoning and tool-calling."
      },
      {
        "name": "NIM and Dynamo",
        "explain": "NIM is one signed container: model + engine + OpenAI API. Dynamo scales it with split prefill/decode and KV-aware routing."
      },
      {
        "name": "OpenShell guardrails",
        "explain": "Signed policies, sandboxes, an egress allowlist and a privacy router keep long-running agents safe."
      },
      {
        "name": "Data flywheel",
        "explain": "Production logs → curate → customize → evaluate → distill to a cheaper model → promote → repeat."
      }
    ],
    "flow": {
      "title": "The self-improving agent stack",
      "nodes": [
        "Runtime",
        "Model",
        "Harness",
        "Observe",
        "Flywheel"
      ],
      "loop": true,
      "caption": "Served models power the harness; traces and evals feed the flywheel, which ships a better model back."
    },
    "needs": [
      "each app's requirements.txt (uv pip install)",
      "optional: Ollama, a DGX or an nvapi- key for REAL"
    ],
    "start": ".venv/bin/python week23_open_superintelligence_stack/00_stack_navigator/tutorial_server.py",
    "labs": [
      {
        "name": "00 · Stack Navigator (hub)",
        "path": "week23_open_superintelligence_stack/00_stack_navigator",
        "run": ".venv/bin/python week23_open_superintelligence_stack/00_stack_navigator/tutorial_server.py",
        "port": 8112,
        "mode": "$0 · SIM/REAL auto",
        "what": "An animated 16-node map of the five-layer stack, a guided tour and launch dots for all 12 apps."
      },
      {
        "name": "00 · Lab Runner",
        "path": "week23_open_superintelligence_stack/00_lab_runner",
        "run": ".venv/bin/python week23_open_superintelligence_stack/00_lab_runner/tutorial_server.py",
        "port": 8113,
        "mode": "$0 · SIM/REAL auto",
        "what": "All 12 TUTORIAL.md files as one course with checkpoints; runs every labs/*.py and streams output."
      },
      {
        "name": "01 · Nemotron models",
        "path": "week23_open_superintelligence_stack/01_nemotron_models",
        "run": ".venv/bin/python week23_open_superintelligence_stack/01_nemotron_models/tutorial_server.py",
        "port": 8100,
        "mode": "$0 · SIM/REAL auto",
        "what": "The Nemotron 3 family, the reasoning tax, raw tool calls on the wire, and fit math per tier."
      },
      {
        "name": "02 · NIM microservices",
        "path": "week23_open_superintelligence_stack/02_nim_microservices",
        "run": ".venv/bin/python week23_open_superintelligence_stack/02_nim_microservices/tutorial_server.py",
        "port": 8101,
        "mode": "$0 · SIM/REAL auto",
        "what": "One signed container = model + engine + OpenAI API; NIM vs raw vLLM/Ollama."
      },
      {
        "name": "04 · Agent Skills",
        "path": "week23_open_superintelligence_stack/04_agent_skills",
        "run": ".venv/bin/python week23_open_superintelligence_stack/04_agent_skills/tutorial_server.py",
        "port": 8103,
        "mode": "$0 · SIM/REAL auto",
        "what": "NVIDIA's portable Agent Skills: author one, load it live, bridge it with MCP and A2A."
      },
      {
        "name": "05 · AI-Q research lab",
        "path": "week23_open_superintelligence_stack/05_aiq_research_lab",
        "run": ".venv/bin/python week23_open_superintelligence_stack/05_aiq_research_lab/tutorial_server.py",
        "port": 8104,
        "mode": "$0 · SIM/REAL auto",
        "what": "Intent Router (Nano) → Deep Agent (Super) → planner and researcher sub-agents via NAT."
      },
      {
        "name": "07 · Guardrails + OpenShell",
        "path": "week23_open_superintelligence_stack/07_guardrails_openshell",
        "run": ".venv/bin/python week23_open_superintelligence_stack/07_guardrails_openshell/tutorial_server.py",
        "port": 8106,
        "mode": "$0 · SIM/REAL auto",
        "what": "Naive vs guarded agents, sign a policy and tamper with it, and a red-team scorecard."
      },
      {
        "name": "12 · Capstone: smart hotel",
        "path": "week23_open_superintelligence_stack/12_capstone_smart_hotel",
        "run": ".venv/bin/python week23_open_superintelligence_stack/12_capstone_smart_hotel/tutorial_server.py",
        "port": 8111,
        "mode": "$0 · SIM/REAL auto",
        "what": "A self-improving agent fleet runs AltoTech Grand Bangkok; the room-1203 alarm runs end to end."
      }
    ],
    "quiz": [
      {
        "q": "In Week 23's formula, an agent equals what?",
        "options": [
          "Model + Harness",
          "Prompt + API key",
          "RAG + Memory",
          "GPU + Container"
        ],
        "answer": 0,
        "why": "The through-line: the model reasons; the harness (tools, memory, security, observability…) makes it an agent."
      },
      {
        "q": "What is a NIM?",
        "options": [
          "A Nemotron training dataset",
          "A guardrail policy file",
          "A GPU type",
          "A signed container: model + engine + OpenAI API"
        ],
        "answer": 3,
        "why": "App 02: one signed container bundles the model, an optimized engine and an OpenAI-compatible API."
      },
      {
        "q": "Which app should you open first to see where everything fits?",
        "options": [
          "00_stack_navigator on :8112",
          "12_capstone_smart_hotel",
          "03_dynamo_serving",
          "11_data_flywheel"
        ],
        "answer": 0,
        "why": "The README calls the Stack Navigator hub the recommended entry point: one map of the whole five-layer stack."
      }
    ],
    "funFact": "Each of the 36 hands-on lab scripts finishes in under a minute and falls back to a clearly labelled SIM with no endpoint.",
    "readme": "week23_open_superintelligence_stack/README.md",
    "minutes": 240
  },
  {
    "id": "24",
    "title": "Typed AI Decisions with Jev (TypeSafe System One)",
    "phase": 6,
    "emoji": "⚖️",
    "tagline": "Jev judges, your code decides: typed answers, not prose",
    "summary": "A 17-module EN/Thai course on Jev, a model that returns typed judgments (choice, noul, score) instead of text. You build routers, triage queues and a hotel copilot where code applies policy and humans approve. Later modules add live voice, Jev 1.14 images and Box, and local System One models you fine-tune on a Mac or DGX Spark. It runs LIVE with a TypeSafe key or DRY from recordings at $0.",
    "concepts": [
      {
        "name": "Three primitives",
        "explain": "choice picks one option with probabilities, noul says if a statement is true (0–1), score places it on your ordered scale."
      },
      {
        "name": "Jev judges, code decides",
        "explain": "Jev only returns judgments; deterministic policy and human approval decide what actually happens."
      },
      {
        "name": "Typed ≠ true",
        "explain": "A perfectly shaped answer can still be wrong, so your code checks, gates and decides."
      },
      {
        "name": "Code does the math",
        "explain": "Jev is not a calculator: compute staleness, deviations and regex matches in Python, then ask Jev to judge."
      },
      {
        "name": "Uncertainty gate",
        "explain": "If Jev is less than 70% sure (the capstone's threshold), send the case to a person."
      }
    ],
    "flow": {
      "title": "A typed decision",
      "nodes": [
        "Message",
        "State + Qs",
        "Jev judges",
        "Code policy",
        "Human OK"
      ],
      "loop": false,
      "caption": "Facts and questions go to Jev; typed answers flow into deterministic policy, then a human approves."
    },
    "needs": [
      "TYPESAFE_API_KEY (LIVE only; DRY is $0)",
      "any one LLM key or Ollama (Module 11)",
      "week24_typed_decisions_jev/.venv-local + ~3 GB models (Modules 16–17; they replay without it)"
    ],
    "start": ".venv/bin/python week24_typed_decisions_jev/00_jev_lab_runner/tutorial_server.py",
    "labs": [
      {
        "name": "Jev Lab Runner",
        "path": "week24_typed_decisions_jev/00_jev_lab_runner",
        "run": ".venv/bin/python week24_typed_decisions_jev/00_jev_lab_runner/tutorial_server.py",
        "port": 8124,
        "mode": "LIVE / DRY switch",
        "what": "All 14 modules in one course: checkpoints, ▶ Run, Ask Jev probability bars, EN/ไทย switch."
      },
      {
        "name": "01 · Hello Jev",
        "path": "week24_typed_decisions_jev/01_hello_jev",
        "run": "JEV_MODE=dry .venv/bin/python week24_typed_decisions_jev/01_hello_jev/labs/lab01_first_call_raw.py",
        "mode": "$0 DRY · key for LIVE",
        "what": "Your first raw Jev call: state + questions in, a typed answer out."
      },
      {
        "name": "02 · Three primitives",
        "path": "week24_typed_decisions_jev/02_three_primitives",
        "run": "JEV_MODE=dry .venv/bin/python week24_typed_decisions_jev/02_three_primitives/labs/lab01_choice.py",
        "mode": "$0 DRY · key for LIVE",
        "what": "Watch a choice split its probability when a message has two needs, then try noul and score."
      },
      {
        "name": "04 · Intent routing",
        "path": "week24_typed_decisions_jev/04_intent_routing",
        "run": "JEV_MODE=dry .venv/bin/python week24_typed_decisions_jev/04_intent_routing/labs/lab01_route_requests.py",
        "mode": "$0 DRY · key for LIVE",
        "what": "Route requests to teams and send uncertain ones to clarify-or-review."
      },
      {
        "name": "07 · AFDD fault triage",
        "path": "week24_typed_decisions_jev/07_afdd_triage",
        "run": "JEV_MODE=dry .venv/bin/python week24_typed_decisions_jev/07_afdd_triage/labs/lab01_triage_incidents.py",
        "mode": "$0 DRY · key for LIVE",
        "what": "Code computes HVAC facts, Jev picks a queue; policy overrides Jev when data can't be trusted."
      },
      {
        "name": "11 · Jev + an LLM",
        "path": "week24_typed_decisions_jev/11_jev_plus_llm",
        "run": "JEV_MODE=dry .venv/bin/python week24_typed_decisions_jev/11_jev_plus_llm/labs/lab02_route_then_write.py",
        "mode": "$0 DRY · LLM key or Ollama",
        "what": "Jev routes, an LLM of your choice writes the reply, then Jev checks the draft."
      },
      {
        "name": "12 · Capstone hotel copilot",
        "path": "week24_typed_decisions_jev/12_capstone_hotel_copilot",
        "run": "JEV_MODE=dry .venv/bin/python week24_typed_decisions_jev/12_capstone_hotel_copilot/labs/lab01_capstone_demo.py",
        "mode": "$0 DRY · key for LIVE",
        "what": "A guest message becomes a routed, safety-checked ticket proposal with execute: False."
      },
      {
        "name": "13 · Multimodal Jev",
        "path": "week24_typed_decisions_jev/13_multimodal_jev",
        "run": "JEV_MODE=dry .venv/bin/python week24_typed_decisions_jev/13_multimodal_jev/labs/lab02_image_to_decision.py",
        "mode": "$0 DRY · keys for LIVE",
        "what": "A vision model turns an image into text; Jev turns that text into a typed decision."
      },
      {
        "name": "14 · Live voice referee",
        "path": "week24_typed_decisions_jev/14_live_realtime_jev",
        "run": "JEV_MODE=dry .venv/bin/python week24_typed_decisions_jev/14_live_realtime_jev/labs/lab03_jev_referee.py",
        "mode": "$0 DRY · keys for LIVE",
        "what": "Jev is the sub-second referee inside a live voice call; every action it flags is a proposal, never an act."
      },
      {
        "name": "15 · Jev sees images",
        "path": "week24_typed_decisions_jev/15_jev_sees_images",
        "run": "JEV_MODE=dry .venv/bin/python week24_typed_decisions_jev/15_jev_sees_images/labs/lab03_box_and_crop.py",
        "mode": "$0 DRY · key for LIVE",
        "what": "Jev 1.14 reads a photo: Choice and Noul on the image, then Box finds where to crop."
      },
      {
        "name": "16 · Local System One bake-off",
        "path": "week24_typed_decisions_jev/16_local_system_one_models",
        "run": "JEV_MODE=dry .venv/bin/python week24_typed_decisions_jev/16_local_system_one_models/labs/lab05_bakeoff.py",
        "mode": "local models · ~2 min",
        "what": "Laya, GLiNER and GLiNER2.5-Decide run offline against Jev on 18 EN/TH tickets: accuracy, speed, cost, calibration."
      },
      {
        "name": "17 · Did the fine-tune work?",
        "path": "week24_typed_decisions_jev/17_finetune_system_one",
        "run": "JEV_MODE=dry .venv/bin/python week24_typed_decisions_jev/17_finetune_system_one/labs/lab06_evaluate.py",
        "mode": "local models · ~2 min",
        "what": "Grade the fine-tuned models on 66 held-out tickets with CIs, fixed/broke and calibration, ready for SHIP or HOLD."
      }
    ],
    "quiz": [
      {
        "q": "Which Jev primitive has no `confidence` field?",
        "options": [
          "choice",
          "score",
          "All have one",
          "noul"
        ],
        "answer": 3,
        "why": "Module 02's checkpoint: noul returns a 0→1 truth value only; choice and score both report confidence."
      },
      {
        "q": "In the AFDD module, who checks whether a reading is older than 10 minutes?",
        "options": [
          "Jev",
          "The LLM writer",
          "Your Python code",
          "The operator"
        ],
        "answer": 2,
        "why": "Jev is not a calculator: exact comparisons of trusted timestamps are done in code before asking Jev."
      },
      {
        "q": "What does DRY mode do when a request was never recorded?",
        "options": [
          "Returns a neutral PLACEHOLDER",
          "Calls the live API anyway",
          "Crashes the lab",
          "Guesses from similar requests"
        ],
        "answer": 0,
        "why": "Unrecorded requests get flat probabilities and noul = 0.5, labelled PLACEHOLDER, which is not a prediction."
      }
    ],
    "funFact": "Jev LIVE costs $0.042 per million input tokens and output is free; a typical lab call is about $0.00002.",
    "readme": "week24_typed_decisions_jev/README.md",
    "minutes": 625
  },
  {
    "id": "25",
    "title": "DGX Spark — Fine-Tune, Serve & Build Sandboxed Agents",
    "phase": 6,
    "emoji": "⚡",
    "tagline": "One Spark (or two) = a private AI stack, no cloud",
    "summary": "A 21-module EN/Thai course built on NVIDIA's DGX Spark playbooks. You serve models many ways, put one LiteLLM gateway in front, fine-tune a hotel router, and run a NAT agent inside an OpenShell sandbox. Labs drive your Spark over SSH (LIVE) or run DRY with every output clearly labelled. Instructors start from START_HERE.md: one ready check and one command get two cabled Sparks ready for class, and Module 05 can run Nemotron 3 Super 120B in FP8 across both.",
    "concepts": [
      {
        "name": "LIVE vs DRY labels",
        "explain": "DRY outputs are RECORDED (real Spark), REFERENCE (quoted from a playbook) or EXAMPLE (illustrative shape)."
      },
      {
        "name": "273 GB/s ceiling",
        "explain": "Single-stream speed ≤ memory bandwidth ÷ bytes of active weights read per token."
      },
      {
        "name": "Engine bake-off",
        "explain": "Ollama, llama.cpp, vLLM, SGLang, TensorRT-LLM and NIM all serve the same OpenAI-style API."
      },
      {
        "name": "One gateway",
        "explain": "LiteLLM gives every engine and both Sparks one URL and one key, with fallbacks and balancing."
      },
      {
        "name": "Sandboxed agent",
        "explain": "A NAT agent runs inside OpenShell and can reach exactly two destinations."
      }
    ],
    "flow": {
      "title": "Capstone: sovereign hotel agent",
      "nodes": [
        "Fine-tune",
        "Evaluate",
        "Serve",
        "Gateway",
        "Agent",
        "Sandbox"
      ],
      "loop": false,
      "caption": "A LoRA hotel router is scored, served on vLLM, put behind LiteLLM, and called as a tool by a sandboxed agent."
    },
    "needs": [
      "DGX Spark via SSH (LIVE only; DRY runs anywhere)",
      "Ollama on the laptop (HTTP labs stand-in)",
      "week25_dgx_spark/.venv-nat and .venv-litellm (Modules 08, 13, 14, 20)",
      "a second Spark over QSFP (optional: Modules 02, 05 §6, 11)"
    ],
    "start": ".venv/bin/python week25_dgx_spark/00_spark_lab_runner/tutorial_server.py",
    "labs": [
      {
        "name": "Spark Lab Runner",
        "path": "week25_dgx_spark/00_spark_lab_runner",
        "run": ".venv/bin/python week25_dgx_spark/00_spark_lab_runner/tutorial_server.py",
        "port": 8125,
        "mode": "LIVE (ssh) / DRY switch",
        "what": "All 21 modules in one course: Spark status chips, Ask the Spark, a terminal for laptop or Spark."
      },
      {
        "name": "Instructor ready check",
        "path": "week25_dgx_spark/START_HERE.md",
        "run": "week25_dgx_spark/spark_host/ready_check.sh",
        "mode": "on Spark A · instructors",
        "what": "One check that both Sparks, the sparklab account, the keys and the downloads are ready before class."
      },
      {
        "name": "01 · Meet your Spark",
        "path": "week25_dgx_spark/01_meet_your_spark",
        "run": ".venv/bin/python week25_dgx_spark/01_meet_your_spark/labs/lab02_memory_budget.py",
        "mode": "$0 · arithmetic, no Spark",
        "what": "Memory-budget arithmetic: what fits in 128 GB and the decode speed ceiling. Runs anywhere."
      },
      {
        "name": "03 · Ollama + Open WebUI",
        "path": "week25_dgx_spark/03_ollama_open_webui",
        "run": "SPARK_MODE=dry .venv/bin/python week25_dgx_spark/03_ollama_open_webui/labs/lab03_speed_benchmark.py",
        "mode": "DRY $0 · LIVE on Spark",
        "what": "Measure time-to-first-token and tok/s and compare them with the bandwidth ceiling."
      },
      {
        "name": "05 · vLLM",
        "path": "week25_dgx_spark/05_vllm",
        "run": "SPARK_MODE=dry .venv/bin/python week25_dgx_spark/05_vllm/labs/lab01_launch_builder.py",
        "mode": "DRY $0 · LIVE on Spark",
        "what": "Build a vLLM launch command; later labs cover batching, tool calls and two-Spark tensor parallel."
      },
      {
        "name": "08 · LiteLLM gateway",
        "path": "week25_dgx_spark/08_litellm_gateway",
        "run": ".venv/bin/python week25_dgx_spark/08_litellm_gateway/labs/lab01_build_config.py",
        "mode": "$0 · laptop, offline",
        "what": "Write a gateway config so every engine and both Sparks sit behind one URL and one key."
      },
      {
        "name": "09 · LLaMA Factory fine-tune",
        "path": "week25_dgx_spark/09_llama_factory",
        "run": "SPARK_MODE=dry .venv/bin/python week25_dgx_spark/09_llama_factory/labs/lab02_hotel_dataset.py",
        "mode": "DRY $0 · LIVE on Spark",
        "what": "Build the hotel-router dataset, then configure and launch a LoRA/QLoRA run."
      },
      {
        "name": "13 · Fine-tune to serve",
        "path": "week25_dgx_spark/13_finetune_to_serve",
        "run": ".venv/bin/python week25_dgx_spark/13_finetune_to_serve/labs/lab01_baseline.py",
        "mode": "$0 · laptop (Ollama or DRY)",
        "what": "Score a baseline with the same rules as the fine-tune: valid JSON, right dept, no missed emergency."
      },
      {
        "name": "20 · Capstone sovereign agent",
        "path": "week25_dgx_spark/20_capstone_sovereign_agent",
        "run": "SPARK_MODE=dry .venv/bin/python week25_dgx_spark/20_capstone_sovereign_agent/labs/lab01_plan_and_preflight.py",
        "mode": "DRY $0 · laptop stand-in",
        "what": "Plan and preflight fine-tune → serve → gateway → NAT agent in a sandbox, across two Sparks."
      }
    ],
    "quiz": [
      {
        "q": "In DRY mode, what does a REFERENCE output mean?",
        "options": [
          "Captured live from your Spark",
          "An illustrative guess",
          "Generated by an LLM",
          "Quoted word-for-word from an NVIDIA playbook"
        ],
        "answer": 3,
        "why": "RECORDED is a real Spark capture, REFERENCE is verbatim playbook text, EXAMPLE is an illustrative shape."
      },
      {
        "q": "What caps how fast one conversation generates tokens on a Spark?",
        "options": [
          "Memory bandwidth (273 GB/s)",
          "The number of Sparks",
          "Disk size",
          "The gateway"
        ],
        "answer": 0,
        "why": "Module 01: single-stream speed ≤ memory bandwidth ÷ bytes of active weights read per token."
      },
      {
        "q": "In the capstone, how does the agent reach the fine-tuned hotel router?",
        "options": [
          "As a tool through the LiteLLM gateway",
          "By loading the LoRA itself",
          "Over email",
          "Through Open WebUI"
        ],
        "answer": 0,
        "why": "The Module 14 NAT agent calls the router as a tool via the Module 08 gateway, from a Module 15 sandbox."
      }
    ],
    "funFact": "audit_references.py checks every REFERENCE against the playbooks: 139 lines, 0 mismatches.",
    "readme": "week25_dgx_spark/README.md",
    "minutes": 600
  },
  {
    "id": "26",
    "title": "NemoClaw on DGX Spark",
    "phase": 6,
    "emoji": "🦞",
    "tagline": "Cage your agent in a sandbox, then prove the bars hold",
    "summary": "You build Alto Ops Claw, a hotel chiller-plant assistant, as a claw: an always-on agent inside an NVIDIA OpenShell sandbox, installed with NemoClaw on a DGX Spark. You write its logic in NeMo Agent Toolkit, then trace, benchmark and harden it. No Spark? Spark labs run DRY and laptop labs still run for real. Module 06 ends with a four-model bake-off on one Spark, Nemotron 3.5 Super VL included.",
    "concepts": [
      {
        "name": "What a claw is",
        "explain": "An always-on, tool-using agent + a local model + an OpenShell sandbox and policy, assembled by the NemoClaw installer."
      },
      {
        "name": "Five layers, two speeds",
        "explain": "Filesystem and process lock at sandbox creation; network and inference are hot-reloadable; gateway auth sits outside."
      },
      {
        "name": "Policy as code",
        "explain": "OpenShell YAML policies you read, write and iterate on: deny, observe, allow, verify."
      },
      {
        "name": "Your own claw in NAT",
        "explain": "NeMo Agent Toolkit workflows with a custom tool, REST, MCP both ways, running inside the same kind of sandbox."
      },
      {
        "name": "Evidence, not claims",
        "explain": "Output is labelled RECORDED, REFERENCE or EXAMPLE; the capstone grader never awards points it cannot prove."
      }
    ],
    "flow": {
      "title": "One claw request, end to end",
      "nodes": [
        "Agent harness",
        "Policy proxy",
        "Gateway :8080",
        "Local model"
      ],
      "loop": false,
      "caption": "The sandboxed agent calls inference.local; the gateway adds the real key and routes it to a local model."
    },
    "needs": [
      "week26_nemoclaw_claws/.venv-nat (NAT 1.9.0)",
      "week26_nemoclaw_claws/.venv-openshell (OpenShell 0.0.111)",
      "Ollama + nemotron-3-nano",
      "DGX Spark via SPARK_HOST (optional, DRY without)"
    ],
    "start": ".venv/bin/python week26_nemoclaw_claws/00_reef_lab_runner/tutorial_server.py",
    "labs": [
      {
        "name": "The claw map",
        "path": "week26_nemoclaw_claws/01_what_is_a_claw/labs/lab01_1_claw_map.py",
        "run": ".venv/bin/python week26_nemoclaw_claws/01_what_is_a_claw/labs/lab01_1_claw_map.py",
        "mode": "read-only · laptop + Spark or DRY",
        "what": "Find which parts of the stack (NAT, OpenShell, Docker, Ollama) exist on your laptop and on your Spark."
      },
      {
        "name": "Prove inference is local",
        "path": "week26_nemoclaw_claws/02_first_claw/labs/lab02_4_inference_is_local.py",
        "run": ".venv/bin/python week26_nemoclaw_claws/02_first_claw/labs/lab02_4_inference_is_local.py",
        "mode": "Spark LIVE / DRY",
        "what": "Check the route names a local provider, list models from inside the sandbox, and watch a cloud call fail."
      },
      {
        "name": "The policy iterate loop",
        "path": "week26_nemoclaw_claws/03_policy_as_code/labs/lab03_3_iterate_loop.py",
        "run": ".venv/bin/python week26_nemoclaw_claws/03_policy_as_code/labs/lab03_3_iterate_loop.py",
        "mode": "changes need 🔓 Allow changes",
        "what": "Deny, observe, allow, verify: feed endpoint specs to the real OpenShell CLI offline, then loop on your Spark."
      },
      {
        "name": "A custom tool: chiller_kpi",
        "path": "week26_nemoclaw_claws/04_nat_claws/labs/lab04_2_custom_tool.py",
        "run": ".venv/bin/python week26_nemoclaw_claws/04_nat_claws/labs/lab04_2_custom_tool.py",
        "mode": "laptop NAT + Ollama",
        "what": "Compute ground truth with no LLM, then ask the agent twice and see why a tool description is a prompt."
      },
      {
        "name": "Agent traces",
        "path": "week26_nemoclaw_claws/05_tracing/labs/lab05_1_file_and_phoenix.py",
        "run": ".venv/bin/python week26_nemoclaw_claws/05_tracing/labs/lab05_1_file_and_phoenix.py",
        "mode": "laptop NAT + Ollama",
        "what": "Run Alto Ops Claw with NAT's file exporter and read its spans, tokens and durations."
      },
      {
        "name": "The sandbox tax",
        "path": "week26_nemoclaw_claws/06_benchmarking/labs/lab06_4_sandbox_tax.py",
        "run": ".venv/bin/python week26_nemoclaw_claws/06_benchmarking/labs/lab06_4_sandbox_tax.py",
        "mode": "laptop stand-in + Spark or DRY",
        "what": "Run the same nat eval on two paths and compare p95 runtime: what the sandbox costs you."
      },
      {
        "name": "Model bake-off",
        "path": "week26_nemoclaw_claws/06_benchmarking/tools/model_bakeoff.py",
        "run": "python3 week26_nemoclaw_claws/06_benchmarking/tools/model_bakeoff.py --name \"Nemotron 3.5 Super VL\" --url http://127.0.0.1:8000/v1 --model nemotron-3.5-super-vl --vision",
        "mode": "on Spark · any OpenAI-compatible server",
        "what": "The same speed, HVAC exam, tool-call and image checks for every model, one request at a time at temperature 0."
      },
      {
        "name": "Threat model: 12 attack steps",
        "path": "week26_nemoclaw_claws/07_hardening/labs/lab07_1_threat_model.py",
        "run": ".venv/bin/python week26_nemoclaw_claws/07_hardening/labs/lab07_1_threat_model.py",
        "mode": "offline",
        "what": "A prompt-injected claw tries 12 steps; a draft and a production policy each decide every one."
      },
      {
        "name": "Capstone: evidence and grade",
        "path": "week26_nemoclaw_claws/08_capstone_alto_ops_claw/labs/lab08_3_evidence_and_grade.py",
        "run": ".venv/bin/python week26_nemoclaw_claws/08_capstone_alto_ops_claw/labs/lab08_3_evidence_and_grade.py",
        "mode": "no LLM calls",
        "what": "Grade Alto Ops Claw v1 on the capstone scheme; Spark points need real Spark evidence."
      }
    ],
    "quiz": [
      {
        "q": "Which two layers can you change on a running OpenShell sandbox?",
        "options": [
          "Filesystem and process",
          "Process and network",
          "Filesystem and inference",
          "Network and inference"
        ],
        "answer": 3,
        "why": "Network and inference are hot-reloadable; filesystem and process are locked when the sandbox is created."
      },
      {
        "q": "In a claw, which part holds the real model-provider key?",
        "options": [
          "The agent harness",
          "The NAT workflow YAML",
          "The sandbox's /tmp folder",
          "The OpenShell gateway"
        ],
        "answer": 3,
        "why": "Agents call inference.local; the gateway injects the real credential, so the agent never holds a provider key."
      },
      {
        "q": "Without a Spark, what does a Spark lab show as output?",
        "options": [
          "Made-up numbers that look real",
          "Labelled RECORDED, REFERENCE or EXAMPLE output",
          "Nothing at all",
          "Laptop numbers labelled as Spark numbers"
        ],
        "answer": 1,
        "why": "Spark output is never invented: RECORDED is a real transcript, REFERENCE is quoted from docs, EXAMPLE is a shape."
      },
      {
        "q": "In Module 06's bake-off, why was Nemotron 3.5 Lightning about 3× faster than the other models?",
        "options": [
          "It skipped the tool calls",
          "It has only 3B active parameters per token",
          "It ran on both Sparks at once",
          "Thinking was turned off only for it"
        ],
        "answer": 1,
        "why": "Speed follows active parameters, then the drafter: Lightning (3B active) hit 93.3 tok/s against 22–31 tok/s for the 12B-active models."
      }
    ],
    "funFact": "NemoClaw ships three harnesses: OpenClaw, Hermes and LangChain Deep Agents, all behind the same OpenShell boundary.",
    "readme": "week26_nemoclaw_claws/README.md",
    "minutes": 750
  },
  {
    "id": "27",
    "title": "Software Factory — an org chart made of loops",
    "phase": 6,
    "emoji": "🏭",
    "tagline": "Build the thing that builds the software: tickets in, merged PRs out",
    "summary": "You build a software factory: a kanban of linted tickets drained by unattended Build → QA → Review lanes, one Claude Code skill each, with state kept only in GitHub. Deterministic guard hooks, a merge gate and behavioural evals keep it honest, and production telemetry files its own verified tickets. The target is Alto Mini, a small hotel-energy app with seeded, ticketable bugs; the reference is Eric Tech's open-source super-board.",
    "concepts": [
      {
        "name": "The ticket is the contract",
        "explain": "Every ticket has Problem, Context, Fix, Acceptance Criteria, Risk and Blocked by, 2–5 checkbox ACs and no vague words, so a lane can act on it unattended."
      },
      {
        "name": "Lanes, not one agent",
        "explain": "Build, QA and Review are separate skills, each in its own worktree per issue, handing work on through GitHub labels and PRs."
      },
      {
        "name": "The test-gap ledger",
        "explain": "QA maps each acceptance criterion to the tests that prove it; an AC with no test is a High gap, and a fix must go red first."
      },
      {
        "name": "The merge gate",
        "explain": "Merge only at the reviewed SHA, with no human-only label, under the size cap, and with tests passing on base + branch."
      },
      {
        "name": "Guards and evals",
        "explain": "PreToolUse hooks exit 2 to block secrets, stray worktrees and protected pushes; plugin evals measure the lanes' behaviour."
      }
    ],
    "flow": {
      "title": "One ticket through the factory",
      "nodes": [
        "Linted ticket",
        "Build lane",
        "QA lane",
        "Review lane",
        "Merge gate",
        "Telemetry"
      ],
      "loop": true,
      "caption": "A ticket becomes a PR that is tested, reviewed and merged at the reviewed SHA; telemetry turns new errors into new tickets."
    },
    "needs": [
      "make + the shared .venv (Alto Mini)",
      "pytest (e.g. uvx pytest) for the exercise solutions",
      "Claude Code + gh for the live factory labs",
      "a GitHub repo you can let the factory push to"
    ],
    "start": "cd week27_software_factory/00_alto_mini && make seed && make test && make dev",
    "labs": [
      {
        "name": "Alto Mini, the target",
        "path": "week27_software_factory/00_alto_mini/Makefile",
        "run": "make -C week27_software_factory/00_alto_mini seed test",
        "port": 8127,
        "mode": "offline",
        "what": "Seed 14 days of hotel-energy data with its known gaps and run the 6 unit and API tests."
      },
      {
        "name": "Lint a ticket",
        "path": "week27_software_factory/02_ticket_and_builder/exercises/ex02_ticket_lint.py",
        "run": "uvx pytest -q week27_software_factory/02_ticket_and_builder/exercises/solutions/test_ex02.py",
        "mode": "offline",
        "what": "Check the five ticket rules: sections, 2–5 checkbox ACs, no vague words, a risk emoji, a Where: line."
      },
      {
        "name": "The test-gap ledger",
        "path": "week27_software_factory/03_qa_lane/exercises/ex03_ledger.py",
        "run": "uvx pytest -q week27_software_factory/03_qa_lane/exercises/solutions/test_ex03.py",
        "mode": "offline",
        "what": "Match each acceptance criterion to test ids and flag every uncovered one as a High gap."
      },
      {
        "name": "Adversarial review",
        "path": "week27_software_factory/04_review_and_merge_gate/exercises/ex04_min_confidence.py",
        "run": "uvx pytest -q week27_software_factory/04_review_and_merge_gate/exercises/solutions/test_ex04.py",
        "mode": "offline",
        "what": "Take the minimum confidence across reviewers, and never let over-engineering alone block a PR."
      },
      {
        "name": "Red-team the guards",
        "path": "week27_software_factory/05_guards_and_evals/exercises/ex05_hook_fuzz.py",
        "run": "cd week27_software_factory/05_guards_and_evals/exercises && python3 ex05_hook_fuzz.py",
        "mode": "offline",
        "what": "Fire commands at every guard hook, read the confusion table, then add five that slip through and fix the regex."
      },
      {
        "name": "The telemetry verifier",
        "path": "week27_software_factory/06_telemetry_to_tickets/exercises/ex06_verifier.py",
        "run": "uvx pytest -q week27_software_factory/06_telemetry_to_tickets/exercises/solutions/test_ex06.py",
        "mode": "offline",
        "what": "Decide whether an error becomes a ticket: real, still happening, not already fixed, not a duplicate."
      },
      {
        "name": "Guard hook tests",
        "path": "week27_software_factory/factory/tests",
        "run": "uvx pytest -q week27_software_factory/factory/tests",
        "mode": "offline",
        "what": "The starter kit's own tests for the PreToolUse guards that block with exit 2."
      }
    ],
    "quiz": [
      {
        "q": "When does the merge gate refuse to merge a reviewed PR?",
        "options": [
          "Never, once review approves",
          "Only when tests fail",
          "When the head moved since review, a human-only label is set, the PR is too big, or tests fail on base + branch",
          "Only when a human-only label is set"
        ],
        "answer": 2,
        "why": "merge-gate.sh checks all four: head == reviewed SHA, no human-only label, size under the cap, and the verify command on base + branch."
      },
      {
        "q": "How does a PreToolUse guard hook block a command?",
        "options": [
          "It exits with code 2",
          "It prints a warning",
          "It deletes the worktree",
          "It asks the model nicely"
        ],
        "answer": 0,
        "why": "In the starter kit, exit 2 blocks the tool call and the hook's message tells the agent why."
      },
      {
        "q": "In the adversarial review exercise, how is the overall confidence computed?",
        "options": [
          "The average across reviewers",
          "The highest reviewer's score",
          "The minimum across reviewers",
          "The last reviewer's score"
        ],
        "answer": 2,
        "why": "ex04 takes the minimum, so a single doubtful reviewer is enough to hold the PR."
      }
    ],
    "funFact": "The course authors ran every offline part of this week, but none of it against a live Claude Code CLI; your first live run in Lab 05 is the verification.",
    "readme": "week27_software_factory/README.md",
    "minutes": 1230
  },
  {
    "id": "28",
    "title": "Securing Multi-Agent Systems with MAESTRO",
    "phase": 6,
    "emoji": "🛡️",
    "tagline": "Whose words is this agent acting on? Attack it, then defend it",
    "summary": "You threat-model, attack and defend the Hotel Ops Ensemble, a simulated hotel multi-agent system, with CSA's seven-layer MAESTRO framework and the OWASP Top 10 for Agentic Applications. Each toy attack motivates a control, and each control is scored on what it stops and what utility it costs. Runs offline on a deterministic SIM brain, or LIVE on local Ollama.",
    "concepts": [
      {
        "name": "MAESTRO's seven layers",
        "explain": "Split an agentic system into seven layers, from L1 Foundation Models up, and list threats per layer and across layers."
      },
      {
        "name": "Threat model as code",
        "explain": "A YAML threat model with risk scores and a heatmap, a threat diff per change, and a CI gate that fails until mitigated."
      },
      {
        "name": "Spotlighting + nonce",
        "explain": "Wrap untrusted text in data markers. One line of page text breaks a static marker; a per-run nonce marker holds."
      },
      {
        "name": "Authorise by provenance",
        "explain": "Decide on who wrote the text behind an action, not on what the agent may do. That stops confused deputies."
      },
      {
        "name": "Security vs utility",
        "explain": "A defense counts only if it stops attacks A01–A12 without breaking benign tasks B01–B08."
      }
    ],
    "flow": {
      "title": "One injected request, hop by hop",
      "nodes": [
        "Guest text",
        "Orchestrator",
        "Specialist",
        "Tool / chiller",
        "Audit log"
      ],
      "loop": false,
      "caption": "An instruction travels agent to agent toward a dangerous tool; every hop is a place a defense can stop it."
    },
    "needs": [
      "Ollama + gemma4:12b (optional, LIVE brain)"
    ],
    "start": ".venv/bin/python week28_maestro_mas_security/00_maestro_lab_runner/tutorial_server.py",
    "labs": [
      {
        "name": "The first compromise",
        "path": "week28_maestro_mas_security/01_why_mas_security/labs/lab01_2_first_compromise.py",
        "run": ".venv/bin/python week28_maestro_mas_security/01_why_mas_security/labs/lab01_2_first_compromise.py",
        "mode": "$0 · SIM/LIVE switch",
        "what": "One extra sentence travels from a guest to the chiller; then rerun with every defense on."
      },
      {
        "name": "Threat diff and CI gate",
        "path": "week28_maestro_mas_security/03_threat_model_as_code/labs/lab03_3_diff_and_gate.py",
        "run": ".venv/bin/python week28_maestro_mas_security/03_threat_model_as_code/labs/lab03_3_diff_and_gate.py",
        "mode": "$0 · SIM only",
        "what": "Add a LINE guest channel, diff the threat model, and watch the CI gate fail, then pass after a fix."
      },
      {
        "name": "Spotlight: static vs nonce",
        "path": "week28_maestro_mas_security/04_goal_hijack_injection/labs/lab04_2_spotlight_static_vs_nonce.py",
        "run": ".venv/bin/python week28_maestro_mas_security/04_goal_hijack_injection/labs/lab04_2_spotlight_static_vs_nonce.py",
        "mode": "$0 · SIM/LIVE switch",
        "what": "Break a static data marker with one line of page text, then see a random per-run nonce hold."
      },
      {
        "name": "Confused deputy + provenance",
        "path": "week28_maestro_mas_security/05_tool_misuse_privilege/labs/lab05_2_confused_deputy_and_provenance.py",
        "run": ".venv/bin/python week28_maestro_mas_security/05_tool_misuse_privilege/labs/lab05_2_confused_deputy_and_provenance.py",
        "mode": "$0 · SIM only",
        "what": "Least agency isn't enough; authorise by who wrote the request and see attacks blocked with reasons."
      },
      {
        "name": "Look-alike vendors, pinning",
        "path": "week28_maestro_mas_security/06_inter_agent_trust_a2a/labs/lab06_2_lookalike_and_pinning.py",
        "run": ".venv/bin/python week28_maestro_mas_security/06_inter_agent_trust_a2a/labs/lab06_2_lookalike_and_pinning.py",
        "mode": "$0 · SIM/LIVE switch",
        "what": "A second \"Acme HVAC Parts\" Agent Card appears; signed, pinned cards stop it. Discovery by name isn't identity."
      },
      {
        "name": "Memory poisoning persists",
        "path": "week28_maestro_mas_security/07_memory_rag_poisoning/labs/lab07_1_memory_poisoning_persists.py",
        "run": ".venv/bin/python week28_maestro_mas_security/07_memory_rag_poisoning/labs/lab07_1_memory_poisoning_persists.py",
        "mode": "$0 · SIM/LIVE switch",
        "what": "One guest saves a fake policy note; another guest's session obeys it, until a memory gate blocks the write."
      },
      {
        "name": "The full red-team suite",
        "path": "week28_maestro_mas_security/09_red_team_scorecard/labs/lab09_1_full_suite.py",
        "run": ".venv/bin/python week28_maestro_mas_security/09_red_team_scorecard/labs/lab09_1_full_suite.py",
        "mode": "$0 · SIM only · ~1 s",
        "what": "12 attacks and 8 benign tasks, naive vs all defenses: attack success rate, utility, and coverage."
      },
      {
        "name": "Capstone: apply and measure",
        "path": "week28_maestro_mas_security/10_capstone_secure_ensemble/labs/lab10_2_apply_and_measure.py",
        "run": ".venv/bin/python week28_maestro_mas_security/10_capstone_secure_ensemble/labs/lab10_2_apply_and_measure.py",
        "mode": "$0 · SIM",
        "what": "Score your defense set against the suite and the GM's budget, with evidence for every threat."
      }
    ],
    "quiz": [
      {
        "q": "How does the SIM brain behave?",
        "options": [
          "It refuses every tool call",
          "It calls a cloud LLM",
          "It obeys every instruction it reads",
          "It replays recorded GPT answers"
        ],
        "answer": 2,
        "why": "SIM is a deterministic stand-in for a gullible LLM: it obeys every instruction, whoever wrote it. Not a real model's rate."
      },
      {
        "q": "What is in the week's red-team suite?",
        "options": [
          "10 attacks and 10 benign tasks",
          "7 attacks, one per layer",
          "12 attacks and 8 benign tasks",
          "20 attacks and no benign tasks"
        ],
        "answer": 2,
        "why": "Attacks A01–A12 and benign tasks B01–B08: a defense is only good if it stops attacks without breaking the hotel."
      },
      {
        "q": "Why does a per-run nonce marker beat a static spotlight marker?",
        "options": [
          "It encrypts the prompt",
          "It makes the model faster",
          "It blocks all tool calls",
          "Injected text can't know it to fake the marker"
        ],
        "answer": 3,
        "why": "One line of page text can fake a static marker. The nonce is random every run, so attacker text can't match it."
      }
    ],
    "funFact": "On the SIM brain the naive ensemble is harmed by all 12 attacks; with every defense on, 0 of 12 succeed and 8 of 8 tasks still work.",
    "readme": "week28_maestro_mas_security/README.md",
    "minutes": 560
  },
  {
    "id": "29",
    "title": "Claude Code Mods, Learned Inside Claude Code",
    "phase": 6,
    "emoji": "🏋️",
    "tagline": "Mod the coding agent you use every day, from inside it",
    "summary": "Mods are TypeScript hooks, shipped in a plugin, that run inside Claude Code and can watch, rewrite or take over a tool call, a prompt or part of the screen. You learn them in Mod Gym, a panel in your terminal that is itself a mod, while growing one lab mod, hello-mod, into a tested, shippable plugin.",
    "concepts": [
      {
        "name": "What makes a mod",
        "explain": "A plugin whose hooks/hooks.json has a modules key: functions that run inside Claude Code and can draw UI."
      },
      {
        "name": "The hook pipeline",
        "explain": "Each hook gets ($, e, next) and can observe, rewrite, wrap or answer an event such as a tool call."
      },
      {
        "name": "Fail-closed guards",
        "explain": "A guard returns { deny }, and its .catch refuses the call if the guard itself breaks. Keep one guard per tool."
      },
      {
        "name": "State that survives",
        "explain": "Pick a module variable, a $.state atom or $.store by what each keeps across a reload, a /clear and a new session."
      },
      {
        "name": "Mods aren't sandboxed",
        "explain": "A mod runs with your permissions. Read the hooks, calls and env reads lines of claude plugin validate first."
      }
    ],
    "flow": {
      "title": "A tool call through your mod",
      "nodes": [
        "Tool call",
        "Your hook",
        "next(e)",
        "Real tool",
        "Band / pane"
      ],
      "loop": false,
      "caption": "A tool call enters your hook, which can observe, rewrite, deny or answer it before next(e) runs the real tool."
    },
    "needs": [
      "Claude Code 2.1.287+",
      "Ollama (optional, lesson 07)"
    ],
    "start": "claude plugin marketplace add ./week29_claude_code_mods && claude plugin install mod-gym@fitness-week29  # then type /mod-gym in Claude Code",
    "labs": [
      {
        "name": "01 What a mod is",
        "path": "week29_claude_code_mods/01_what_is_a_mod/TUTORIAL.md",
        "run": "",
        "mode": "/mod-gym 1",
        "what": "Mods vs skills, settings hooks and MCP; the trust model; audit with claude plugin validate; tour Mod Gym."
      },
      {
        "name": "02 Your first mod: three files",
        "path": "week29_claude_code_mods/02_first_mod/TUTORIAL.md",
        "run": "",
        "mode": "/mod-gym 2 · press c to check",
        "what": "plugin.json, hooks.json modules and register; validate, --plugin-dir and hot reload. Add a spinner suffix."
      },
      {
        "name": "03 The hook pipeline",
        "path": "week29_claude_code_mods/03_hook_pipeline/TUTORIAL.md",
        "run": "",
        "mode": "/mod-gym 3 · press c to check",
        "what": "Observe, rewrite, wrap and answer, reading the Events tape. Add a tool-call counter and prompt context."
      },
      {
        "name": "04 Guards: deny, hold, catch",
        "path": "week29_claude_code_mods/04_guards/TUTORIAL.md",
        "run": "",
        "mode": "/mod-gym 4 · press c to check",
        "what": "{ deny }, $.ui.ask, .catch and fail-closed guards, plus a measured hook-order bug. Add a force-push guard."
      },
      {
        "name": "05 Draw: panes, bands, toasts",
        "path": "week29_claude_code_mods/05_drawing/TUTORIAL.md",
        "run": "",
        "mode": "/mod-gym 5 · press c to check",
        "what": "Surfaces, panes, bands, status and toasts, shared with other plugins. Add a band with a button."
      },
      {
        "name": "06 State that survives",
        "path": "week29_claude_code_mods/06_state_and_store/TUTORIAL.md",
        "run": "",
        "mode": "/mod-gym 6 · press c to check",
        "what": "Module variables vs $.state atoms vs $.store, and the PluginState contract. Make your reps persist."
      },
      {
        "name": "07 Commands, tools, models",
        "path": "week29_claude_code_mods/07_commands_tools_models/TUTORIAL.md",
        "run": "",
        "mode": "/mod-gym 7 · press c to check",
        "what": "Slash commands with no Claude turn, tools Claude calls, $.model, $.process, $.http. Add /reps and log_reps."
      },
      {
        "name": "08 Test, validate, ship",
        "path": "week29_claude_code_mods/08_test_and_ship/TUTORIAL.md",
        "run": "",
        "mode": "/mod-gym 8 · capstone",
        "what": "claude plugin test, tsc, marketplaces and versions; capstone: audit and fix the untested research starter."
      }
    ],
    "quiz": [
      {
        "q": "What turns a Claude Code plugin into a mod?",
        "options": [
          "A skills/<name>/SKILL.md file",
          "A .mcp.json file",
          "A statusLine setting",
          "A modules key in hooks/hooks.json"
        ],
        "answer": 3,
        "why": "A mod is a plugin whose hooks/hooks.json has a modules key. That key, and nothing else, makes it a mod."
      },
      {
        "q": "Why does a module variable like `calls` restart from zero?",
        "options": [
          "Saving the file hot-reloads the module",
          "Claude clears it every turn",
          "It is stored in the cloud",
          "TypeScript forbids globals"
        ],
        "answer": 0,
        "why": "Each save reloads the module, so its variables start fresh. Lesson 06 covers $.state and $.store for lasting values."
      },
      {
        "q": "Are mods sandboxed?",
        "options": [
          "Yes, in a container",
          "Yes, in a WebAssembly VM",
          "Only on macOS",
          "No, they run with your permissions"
        ],
        "answer": 3,
        "why": "A mod can read your files, approve tool calls and spend your usage, so read its validate report before loading it."
      }
    ],
    "funFact": "Anthropic added mods to Claude Code on 1 October 2026, and Mod Gym, the panel that teaches them, is itself a mod.",
    "readme": "week29_claude_code_mods/README.md",
    "minutes": 380
  },
  {
    "id": "30",
    "title": "Multimodal Embeddings on the Edge",
    "phase": 6,
    "emoji": "🪆",
    "tagline": "Photos, voicemails and CCTV as vectors that never leave the building",
    "summary": "You turn text, photos, voicemails and CCTV into vectors on your own machine with Google DeepMind's EmbeddingGemma 2 (740M, Apache 2.0). Then you build search, private RAG and Alto Edge Memory, a hotel agent memory that never sends a byte off the premises. All data is synthetic.",
    "concepts": [
      {
        "name": "The device ladder",
        "explain": "Capacity decides what fits, bandwidth decides how fast it runs. An embedding model fits rungs below any chat model."
      },
      {
        "name": "Matryoshka vectors",
        "explain": "The first 512, 256 or 128 of the 768 numbers are a usable smaller vector. Cut, re-normalise, keep going."
      },
      {
        "name": "One multimodal space",
        "explain": "Text, photos, voice and video share one 768-d space, so a plain text question can find a photo or voicemail."
      },
      {
        "name": "The modality gap",
        "explain": "Items of the same modality cluster together whatever they mean, so in one ranked list notes crowd out media."
      },
      {
        "name": "Egress proof",
        "explain": "\"It runs locally\" becomes evidence: a guard logs every connection attempt and blocks deliberate leaks."
      }
    ],
    "flow": {
      "title": "Sovereign RAG, all on one machine",
      "nodes": [
        "Media + text",
        "EmbeddingGemma",
        "768-d vector",
        "SQLite store",
        "Local Gemma 4"
      ],
      "loop": false,
      "caption": "Notes, photos and voicemails become vectors in one space, are stored locally, and feed a private answer."
    },
    "needs": [
      "week30_multimodal_embeddings_edge/.venv-embed (transformers ≥ 5.19)",
      "1.5 GB model download on first run",
      "Ollama + gemma4:12b (Modules 05, 07)",
      "ffmpeg (video)",
      "Apple-silicon Mac + week30_multimodal_embeddings_edge/.venv-mlx (Module 08)"
    ],
    "start": ".venv/bin/python week30_multimodal_embeddings_edge/00_embed_lab_runner/tutorial_server.py",
    "labs": [
      {
        "name": "The device ladder",
        "path": "week30_multimodal_embeddings_edge/01_local_sovereign_edge/labs/lab01_1_device_ladder.py",
        "run": "week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/01_local_sovereign_edge/labs/lab01_1_device_ladder.py",
        "mode": "offline · no model needed",
        "what": "Climb from a 32 KB Arduino to 8× H100 with a fit rule of thumb, and see where an embedding model lands."
      },
      {
        "name": "First vectors",
        "path": "week30_multimodal_embeddings_edge/02_first_embeddings/labs/lab02_1_first_vectors.py",
        "run": "week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/02_first_embeddings/labs/lab02_1_first_vectors.py",
        "mode": "1.5 GB download on first run",
        "what": "Turn sentences into 768 numbers, check their length is 1, and run your first semantic search."
      },
      {
        "name": "Matryoshka: cut the vector",
        "path": "week30_multimodal_embeddings_edge/03_matryoshka_and_quantization/labs/lab03_1_matryoshka.py",
        "run": "week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/03_matryoshka_and_quantization/labs/lab03_1_matryoshka.py",
        "mode": "offline after download",
        "what": "Keep the first 512, 256 or 128 numbers and measure recall, storage and what you lose."
      },
      {
        "name": "Cross-modal search",
        "path": "week30_multimodal_embeddings_edge/04_multimodal_search/labs/lab04_1_cross_modal.py",
        "run": "week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/04_multimodal_search/labs/lab04_1_cross_modal.py",
        "mode": "all encoders (740M)",
        "what": "A plain-English question finds the right photo or voicemail, with no captions or transcripts."
      },
      {
        "name": "Egress proof",
        "path": "week30_multimodal_embeddings_edge/05_sovereign_rag/labs/lab05_1_egress_proof.py",
        "run": "week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/05_sovereign_rag/labs/lab05_1_egress_proof.py",
        "mode": "network fenced off",
        "what": "Embed and search with only loopback allowed, then watch two deliberate leaks get blocked and logged."
      },
      {
        "name": "Private RAG",
        "path": "week30_multimodal_embeddings_edge/05_sovereign_rag/labs/lab05_3_private_rag.py",
        "run": "week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/05_sovereign_rag/labs/lab05_3_private_rag.py",
        "mode": "Ollama + gemma4:12b",
        "what": "Local embeddings find the notes and a local Gemma 4 answers from them, citing each one."
      },
      {
        "name": "Build Alto Edge Memory",
        "path": "week30_multimodal_embeddings_edge/07_capstone_alto_edge_memory/labs/lab07_1_build_memory.py",
        "run": "week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/07_capstone_alto_edge_memory/labs/lab07_1_build_memory.py",
        "mode": "offline · egress guard on",
        "what": "Index notes, photos, voicemails, CCTV and incidents into one SQLite file at 512 dimensions."
      },
      {
        "name": "The memory agent",
        "path": "week30_multimodal_embeddings_edge/07_capstone_alto_edge_memory/labs/lab07_2_memory_agent.py",
        "run": "week30_multimodal_embeddings_edge/.venv-embed/bin/python week30_multimodal_embeddings_edge/07_capstone_alto_edge_memory/labs/lab07_2_memory_agent.py",
        "mode": "Ollama · ≤ 6 LLM calls",
        "what": "A local Gemma 4 searches each kind of memory before it answers two night-shift questions (EN + Thai)."
      }
    ],
    "quiz": [
      {
        "q": "How many numbers are in a full EmbeddingGemma 2 vector?",
        "options": [
          "128",
          "512",
          "768",
          "3072"
        ],
        "answer": 2,
        "why": "Lab 02-1 shows shape (768,), with length about 1 because the model's last module normalises every vector."
      },
      {
        "q": "What is the modality gap?",
        "options": [
          "Thai text cannot be embedded",
          "Video takes longer to embed than text",
          "GGUF builds lose audio support",
          "Same-modality items cluster together, whatever they mean"
        ],
        "answer": 3,
        "why": "A Thai voicemail sat closer to an unrelated towel voicemail (0.702) than to the English note meaning the same (0.644)."
      },
      {
        "q": "On the device ladder, what decides how fast a model runs?",
        "options": [
          "Capacity",
          "Disk size",
          "Screen resolution",
          "Bandwidth"
        ],
        "answer": 3,
        "why": "Module 01's one idea: capacity decides what fits, bandwidth decides how fast it runs."
      }
    ],
    "funFact": "Cutting vectors from 768 to 256 numbers costs under 1 point on the model card's MTEB multilingual v2: 61.36 vs 60.41.",
    "readme": "week30_multimodal_embeddings_edge/README.md",
    "minutes": 390
  },
  {
    "id": "R1",
    "title": "Agentic Robotics (SO-ARM101)",
    "phase": 7,
    "emoji": "🦾",
    "tagline": "From a simulated arm to a robot agent that asks before it moves",
    "summary": "You grow a simulated SO-ARM101 arm in MuJoCo into a tool-using robot agent: manual control, record and replay, kinematic pick-and-place, wrist-camera scans and a reconstructed, labelled scene. An LLM then plans with six narrow, validated robot tools, and a portal drives the simulation or a safety-gated path to the real arm.",
    "concepts": [
      {
        "name": "Oracle baseline",
        "explain": "Pick-and-place first reads object positions straight from MuJoCo: if this layer fails, perception is not the problem."
      },
      {
        "name": "Damped least-squares IK",
        "explain": "ik.py turns targets into joint angles; pick_place.py runs tuck, pan, hover, descend, grasp, carry, place, release."
      },
      {
        "name": "Active perception",
        "explain": "The arm visits left, center and right survey poses and saves settled wrist-camera RGB with poses in the base frame."
      },
      {
        "name": "Geometry, then names",
        "explain": "Depth Anything V2 estimates depth from RGB, aligned to the z = 0 table plane; then objects get labels."
      },
      {
        "name": "LLM plans, Python moves",
        "explain": "The LLM picks from six narrow tools; deterministic Python validates every motion and waits for APPROVE."
      }
    ],
    "flow": {
      "title": "From camera to approved motion",
      "nodes": [
        "Wrist scan",
        "3D scene",
        "LLM picks tool",
        "Simulate plan",
        "APPROVE",
        "Execute+verify"
      ],
      "loop": false,
      "caption": "Camera views become a labelled 3D scene; the LLM picks safe tools, and nothing moves until a human types APPROVE."
    },
    "needs": [
      "week_agentic_robotic/.venv (Python 3.11 + requirements.txt)",
      "OPENROUTER_API_KEY (hosted labels and LLM agent)",
      "mjpython on macOS (MuJoCo passive viewer)",
      "SO-ARM101 + LeRobot (optional, physical arm)"
    ],
    "start": "cd week_agentic_robotic && .venv/bin/python 00_getting_started/check_setup.py",
    "labs": [
      {
        "name": "Preflight check",
        "path": "week_agentic_robotic/00_getting_started/check_setup.py",
        "run": "cd week_agentic_robotic && .venv/bin/python 00_getting_started/check_setup.py",
        "mode": "offline",
        "what": "Checks MuJoCo, PyTorch and Transformers, picks CUDA/Metal/CPU, loads the SO-101 and renders one wrist frame."
      },
      {
        "name": "View and move every joint",
        "path": "week_agentic_robotic/01_sim_manual_control/view.py",
        "run": "cd week_agentic_robotic && .venv/bin/python 01_sim_manual_control/view.py",
        "mode": "MuJoCo window",
        "what": "Drive all six actuators from the Control panel; teleop.py adds keyboard control with a live wrist-cam inset."
      },
      {
        "name": "Kinematic pick and place",
        "path": "week_agentic_robotic/03_kinematic_pick_place/pick_place.py",
        "run": "cd week_agentic_robotic && .venv/bin/mjpython 03_kinematic_pick_place/pick_place.py all",
        "mode": "macOS: mjpython · or --headless",
        "what": "Watch the IK oracle baseline tuck, hover, grasp, carry and place every object."
      },
      {
        "name": "Scan with the wrist camera",
        "path": "week_agentic_robotic/04_active_perception/scan_workspace.py",
        "run": "cd week_agentic_robotic && .venv/bin/python 04_active_perception/scan_workspace.py",
        "mode": "MuJoCo window",
        "what": "Three survey poses, a green flash per saved frame; open output/latest_scan/contact_sheet.png after."
      },
      {
        "name": "Reconstruct the scene",
        "path": "week_agentic_robotic/05_semantic_scene/reconstruct_scene.py",
        "run": "cd week_agentic_robotic && .venv/bin/python 04_active_perception/scan_workspace.py --scene empty && .venv/bin/python 05_semantic_scene/reconstruct_scene.py",
        "mode": "downloads Depth Anything V2 once",
        "what": "Compare the scan with an empty table and fuse learned depth into 3D objects; then label them."
      },
      {
        "name": "Agentic manipulation",
        "path": "week_agentic_robotic/06_agentic_manipulation/guided_demo.py",
        "run": "cd week_agentic_robotic && .venv/bin/python 06_agentic_manipulation/guided_demo.py --target \"writing tool\" --destination \"right side\"",
        "mode": "offline · needs Milestone 05 output",
        "what": "Learn the six-tool loop with no API: find, plan and simulate a pick-and-place; add --execute to run it."
      },
      {
        "name": "Guided full pipeline",
        "path": "week_agentic_robotic/07_guided_full_pipeline/run_guided_workshop.py",
        "run": "cd week_agentic_robotic && .venv/bin/python 07_guided_full_pipeline/run_guided_workshop.py --skip-prepare",
        "mode": "OPENROUTER_API_KEY or --manual-labels",
        "what": "Replay the whole loop: scan, reconstruct, label, find, plan, simulate, approve, execute, verify."
      },
      {
        "name": "Workshop portal",
        "path": "week_agentic_robotic/08_workshop_portal/app.py",
        "run": "cd week_agentic_robotic && .venv/bin/python 08_workshop_portal/app.py",
        "port": 8000,
        "mode": "sim, or gated real SO-101",
        "what": "One dashboard: manual controls, live sim and wrist streams, and a chat agent that plans and runs tasks."
      }
    ],
    "quiz": [
      {
        "q": "Why does lesson 03 read object positions straight from MuJoCo?",
        "options": [
          "Oracle baseline: if it fails, perception isn't to blame",
          "The simulated camera is broken",
          "It is faster than IK",
          "LLMs need exact coordinates"
        ],
        "answer": 0,
        "why": "The lesson is the oracle baseline on purpose: if this layer fails, perception is not the problem."
      },
      {
        "q": "How many robot tools can the LLM choose from in lesson 06?",
        "options": [
          "Three",
          "Ten",
          "Six",
          "Any Python function"
        ],
        "answer": 2,
        "why": "Six narrow tools (scan, find, plan, simulate, execute, verify); deterministic Python validates and runs every motion."
      },
      {
        "q": "What must you type before guided_demo.py --execute moves the arm?",
        "options": [
          "yes",
          "go",
          "sudo",
          "APPROVE"
        ],
        "answer": 3,
        "why": "The script pauses for the exact word APPROVE. --approve exists only for simulation tests, never for hardware."
      }
    ],
    "funFact": "The simulated wrist camera is mounted 90° counter-clockwise, which is why its picture-in-picture view is portrait.",
    "readme": "week_agentic_robotic/README.md",
    "minutes": 240
  }
]

# App 06 · NemoClaw — build specialized agents

This app teaches NVIDIA's NemoClaw pattern for building specialized agents by composition: an agent is an authored SPEC — base model + persona + skills + tools + signed policy — not hand-written code. Across five chapters you define a specialist, equip it with skills and tools from the Agent Skills catalog, run it safely inside an OpenShell-style sandbox where a signed policy and egress allowlist gate every tool call, and finally compose a supervised fleet of experts. Works REAL against any OpenAI-compatible endpoint (local Ollama, a DGX Spark over Tailscale, or build.nvidia.com) or in SIM mode with no GPU at all — cloud cost $0 on the sovereign paths.

Launch:

```bash
.venv/bin/python week23/06_nemoclaw/tutorial_server.py   # → http://127.0.0.1:8105
```

Hands-on side: see **[TUTORIAL.md](TUTORIAL.md)** for the copy-paste walkthrough and the three lab scripts in `labs/` (author + lint a spec, naive-vs-gated policy runtime, fleet routing + right-sizing).

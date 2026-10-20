# ▶ Spark Lab 19 — Multi-agent chatbot, RAG and knowledge graphs

> Part of Week 25 · DGX Spark: fine-tune, serve, and build sandboxed agents. You type the commands, you see the real output. Every lab also runs in **DRY** mode (no Spark, $0): commands are shown, and the output is either RECORDED from a real Spark, a REFERENCE quoted from NVIDIA's playbook, or a clearly marked EXAMPLE.

**What you'll actually do**
- Run NVIDIA's multi-agent chatbot: a gpt-oss-120B supervisor that hands work to coding, RAG and vision specialists over MCP.
- Turn text into a knowledge graph with txt2kg (ArangoDB or Neo4j, Ollama or vLLM), and see where it breaks.
- Start the agentic RAG example in NVIDIA AI Workbench, and learn which parts of it leave your Spark.
- Map the stacks' ports, GPUs and memory before you start them, so they don't collide with Modules 03–18.
- Build the core of each idea yourself on the laptop: a txt2kg-style triple extractor and a retriever that cites its sources.

**Time** ~50 min · **Difficulty** intermediate · **Hardware** 1 DGX Spark (or none: labs parse the playbooks and use your laptop's Ollama as a labelled stand-in)

**Official playbooks covered:** [Multi-agent chatbot](https://build.nvidia.com/spark/multi-agent-chatbot) · [txt2kg](https://build.nvidia.com/spark/txt2kg) · [RAG application in AI Workbench](https://build.nvidia.com/spark/rag-ai-workbench)

## 0 · Before you start

| Need | Check | Why |
|---|---|---|
| Module 18 done | you know what a tool call is and why it can fail | the supervisor agent lives on tool calls |
| Docker without sudo on the Spark | `docker ps` | both stacks are Docker Compose |
| ~75 GB free disk on the Spark | `df -h /` | the chatbot's default models |
| The playbooks cloned at the repo root | `ls dgx-spark-playbooks/nvidia/playbook-txt2kg` | labs 01–03 read the playbooks' own files |
| Ollama on the laptop (optional) | `ollama list` | stand-in for labs 02 and 03 |

```bash
# on: laptop
cd agenticaicodingfitness        # the root of your clone of this repo
ls dgx-spark-playbooks >/dev/null 2>&1 || git clone https://github.com/NVIDIA/dgx-spark-playbooks
ls dgx-spark-playbooks/nvidia | grep -cE '^playbook-'
```

**Expected output** (captured on this Mac; the count grows as NVIDIA adds playbooks)

```
65
```

> ⚠ The chatbot alone uses about 120 GB of memory. Before Section 2, stop what earlier modules left running: `ollama stop <model>` for Module 18's coding model, and `docker ps` to find vLLM, SGLang or LiteLLM containers.

✓ Checkpoint: `dgx-spark-playbooks/` exists at the repo root, and `docker ps` runs on the Spark without `sudo`.

## 1 · Three ways to put your documents to work

The three playbooks answer the same question ("let a model use my documents") with three different architectures:

| | Multi-agent chatbot | txt2kg | Agentic RAG (AI Workbench) |
|---|---|---|---|
| Core idea | a **supervisor** LLM routes each request to a specialist tool | an LLM turns text into **(subject, predicate, object)** triples in a graph DB | a **router** picks a retrieval path, then **graders** check relevance and hallucination and loop |
| Retrieval | vectors: Qwen3-Embedding-4B → Milvus | graph traversal (optional vectors with `--vector-search`: Qdrant + Sentence Transformers, `all-MiniLM-L6-v2`) | vector store + web search (Tavily) |
| Models | gpt-oss-120B (supervisor), Deepseek-Coder 6.7B, Qwen2-VL / Qwen2.5-VL 7B, Qwen3-Embedding-4B | llama3.1:8b (Ollama) or Nemotron Super 49B FP8 (vLLM) | NVIDIA-hosted endpoints by default, or self-hosted models |
| Runs as | Docker Compose, UI on :3000 | Docker Compose via `./start.sh`, UI on :3001 | an AI Workbench project, Gradio UI |
| Leaves the Spark? | no (only model downloads) | no (only downloads) | **yes by default**: needs `NVIDIA_API_KEY` and `TAVILY_API_KEY` |
| Good at | mixed tasks: code, docs, images in one chat | multi-hop questions ("how is X connected to Y?") | answer quality checks on open questions |

**When to use which:** use vector RAG to ask what a document says. Use a knowledge graph to ask how things connect across documents. Use a supervisor when one chat must do several kinds of work. Lab 03 below builds the retrieval half of RAG, and lab 02 builds the extraction half of txt2kg, so you can see each one's failure mode for yourself.

✓ Checkpoint: for "which rooms lose cooling if chiller 2 trips?" you can say why a graph beats plain vector search, and for "summarise this PDF" why it doesn't.

## 2 · The multi-agent chatbot: a supervisor and three specialists

Here is what the playbook's code wires together (`assets/backend/` and the two Compose files):

```text
 browser :3000 ─► frontend ─► backend :8000  (FastAPI · LangGraph · Postgres for chats)
                                   │  supervisor agent = gpt-oss-120b  (llama.cpp, http://gpt-oss-120b:8000/v1)
                                   │  tools come from 4 MCP servers over stdio (backend/client.py):
                                   ├─ write_code        → deepseek-coder  (llama.cpp)
                                   ├─ search_documents  → Milvus :19530 + qwen3-embedding (llama.cpp --embeddings)
                                   ├─ explain_image     → qwen2.5-vl container (trtllm-serve, TensorRT-LLM)
                                   └─ weather (test tool)
```

Four design choices are worth copying:

1. **The supervisor may not do the specialist's job.** Its system prompt (`backend/prompts.py`) says "NEVER EVER generate code yourself" and "DO NOT try to answer questions from documents yourself". Routing is only reliable when the router is not allowed to shortcut it.
2. **Batch independent calls, stage dependent ones.** The prompt's few-shot examples call two weather tools together, but run `search_documents` first and `write_code` second when the code depends on the search.
3. **A hard loop limit.** `agent.py` sets `max_iterations = 3`, so a confused tool loop ends instead of burning the GPU.
4. **Specialists are just OpenAI-compatible endpoints.** Each MCP tool calls `http://<container>:8000/v1`. Swap a specialist and the supervisor doesn't change.

The RAG specialist chunks uploads at 1000 characters with 200 overlap and retrieves `k=8` chunks (`vector_store.py`). One thing we noticed when reading the code: `image_understanding.py` asks for model `Qwen2.5-VL-7B-Instruct`, while `docker-compose-models.yml` serves `Qwen/Qwen2-VL-7B-Instruct` in a container named `qwen2.5-vl`. If image answers fail on your Spark, check this first.

The supervisor's routing decision is one tool call. Try it with the Module 18 model (to the Spark's Ollama, or the laptop stand-in). A good answer calls `search_documents` first, because the code depends on it:

```spark
{"target": "ollama", "which": "a", "model": "qwen3.6:35b-a3b-mtp-q4_K_M",
 "messages": [{"role": "system", "content": "You are a supervisor agent. ALWAYS use a tool when the request matches it. NEVER write code yourself. If a call depends on another call's output, make only the first call now."},
              {"role": "user", "content": "Search my documents for the design requirements, then build a website based on them."}],
 "max_tokens": 256,
 "tools": [{"type": "function", "function": {"name": "search_documents", "description": "Search the user's uploaded documents.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
           {"type": "function", "function": {"name": "write_code", "description": "Write complete code.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "programming_language": {"type": "string"}}, "required": ["query", "programming_language"]}}}]}
```

**Run it on the Spark** (playbook Steps 2–5). The model download is 30 minutes to 2 hours. The first `up` builds the llama.cpp CUDA image, which takes 10–20 minutes:

```bash
# on: spark
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/playbook-multi-agent-chatbot/assets
chmod +x model_download.sh
./model_download.sh
docker compose -f docker-compose.yml -f docker-compose-models.yml up -d --build
watch 'docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"'
```

The playbook notes that "the Qwen2.5-VL model container may report as unhealthy while starting up". Then forward the UI and API ports to your laptop, and open `http://localhost:3000`:

```bash
# on: laptop
ssh -L 3000:localhost:3000 -L 8000:localhost:8000 spark-a
```

To try the RAG agent, upload the NVIDIA Blackwell whitepaper PDF (linked in the playbook's Step 6) with **Upload Documents**, tick it under **Select Sources**, and click a tile. Need memory headroom? Playbook Step 8 switches the supervisor to gpt-oss-20B: edit `model_download.sh`, `docker-compose-models.yml` and the `MODELS=` variable in `docker-compose.yml`, then run `up` again. Clean up with:

```bash
# on: spark
cd ~/dgx-spark-playbooks/nvidia/playbook-multi-agent-chatbot/assets
docker compose -f docker-compose.yml -f docker-compose-models.yml down
docker volume rm "$(basename "$PWD")_postgres_data"      # deletes the chat history volume
```

✓ Checkpoint: you can trace one request from the browser to `deepseek-coder` and back, and name the file that forbids the supervisor from writing code.

## 3 · Map the stacks before you start them: lab 01

Two stacks, thirteen host ports and six GPU services, on a machine where you already ran Ollama and vLLM. Lab 01 reads the playbooks' own Compose files and finds the collisions for you:

```bash
# on: laptop
.venv/bin/python week25/19_multi_agent_rag_kg/labs/lab01_stack_map.py
```

**Expected output** (captured on this Mac, parsed from `dgx-spark-playbooks@3410c65`; Step 4's Spark commands print EXAMPLE shapes in DRY mode)

```
▣ STEP 1 · every service, per stack (from the Compose files)
│ stack                service                image                                         host ports  gpu  note
│ ───────────────────  ─────────────────────  ────────────────────────────────────────────  ──────────  ───  ──────────────────────
│ multi-agent chatbot  backend                (built locally)                               8000
│ multi-agent chatbot  frontend               (built locally)                               3000
│ multi-agent chatbot  postgres               postgres:15-alpine                            5432
│ multi-agent chatbot  milvus                 milvusdb/milvus:v2.5.15-20250718-3a3b374f-gp  19530,9091
│ multi-agent chatbot  qwen2.5-vl             nvcr.io/nvidia/tensorrt-llm/release:spark-si  —           GPU
│ multi-agent chatbot  gpt-oss-120b           local/llama.cpp:server-cuda                   —           GPU
│ txt2kg (./start.sh)  app                    (built locally)                               3001
│ txt2kg (./start.sh)  arangodb               arangodb:latest                               8529
│ txt2kg (./start.sh)  ollama                 ollama-custom:latest                          11434       GPU
│ txt2kg --neo4j       neo4j                  neo4j:5-community                             7474,7687
│ txt2kg --vllm        vllm                   (built locally)                               8001        GPU
│ …

▣ STEP 2 · port collisions — between stacks, and with servers earlier modules started
│ port   published by                                         both stacks?  course server on that port
│ ─────  ───────────────────────────────────────────────────  ────────────  ──────────────────────────────────
│ 8000   chatbot:backend, txt2kg:sentence-transformers (opt)  yes           Module 05 (vLLM) / Module 06 (NIM)
│ 11434  txt2kg:ollama                                        —             Modules 03/18 (host Ollama)
◆ (opt) = only with --vector-search. Docker refuses to start a container whose host port is taken
  ('port is already allocated'). Stop the earlier server first, or run one stack at a time.

▣ STEP 3 · memory: what the playbooks say each stack needs
│ multi-agent chatbot     ~120 GB (playbook: 'uses ~120 GB of memory by defau…  REFERENCE
│   downloads             gpt-oss-120B ~63 GB · Deepseek-Coder 6.7B ~7 GB · Q…  REFERENCE
│ txt2kg (Ollama stacks)  llama3.1:8b default → ~4.8 GB of Q4_K_M weights + K…  arithmetic
│ txt2kg --vllm           nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8 → ~49 …  arithmetic
│ Module 18 coding model  qwen3.6:35b-a3b-mtp-q4_K_M ~23 GB                     REFERENCE
```

(Step 1 is trimmed here. The lab prints all 29 service rows.) What it tells you:

- **:11434.** txt2kg's default stack starts its *own* Ollama container (`ollama-compose`) on the host's Ollama port. If the Spark's system Ollama from Modules 03 and 18 is running, stop it first (`sudo systemctl stop ollama`), or the txt2kg `ollama` container can't bind the port.
- **:8000.** The chatbot backend takes vLLM's and NIM's port. Stop Module 05/06 servers first.
- **Memory.** The chatbot alone fills most of a Spark. Run the chatbot or txt2kg, not both.

✓ Checkpoint: before `docker compose up`, you know which earlier server to stop for each stack.

## 4 · txt2kg: text → triples → graph

txt2kg is a Next.js app (UI on :3001) that chunks your documents, asks an LLM for triples, stores them in ArangoDB or Neo4j, and renders the graph with Three.js WebGPU. Its query page walks the graph to add related entities to the prompt before the LLM answers. The playbook's stacks:

| Start command | Graph DB | LLM | Default model |
|---|---|---|---|
| `./start.sh` | ArangoDB (:8529) | Ollama (:11434) | `llama3.1:8b` |
| `./start.sh --neo4j` | Neo4j (:7474, bolt :7687) | Ollama (:11434) | `llama3.1:8b` |
| `./start.sh --vllm` | Neo4j | vLLM (:8001) | `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-FP8` |

Add `--vector-search` to any of them for Qdrant + Sentence Transformers. On the Spark (playbook Steps 1–3):

```bash
# on: spark
cd ~/dgx-spark-playbooks/nvidia/playbook-txt2kg/assets
./start.sh --help
./start.sh --neo4j
docker exec ollama-compose ollama pull llama3.1:8b
```

This course starts `--neo4j` because Neo4j is the graph database Week 15 already taught. The playbook's default is plain `./start.sh` (ArangoDB), and both work on a Spark. The playbook notes that the 64 KB page-size problem with ArangoDB affects DGX Station, not DGX Spark. Open the UI through a tunnel, then upload markdown, text or CSV files (the formats the playbook lists):

```bash
# on: laptop
ssh -N -L 3001:localhost:3001 -L 7474:localhost:7474 -L 7687:localhost:7687 spark-a
# UI: http://localhost:3001 · Neo4j Browser: http://localhost:7474
```

The extraction prompt lives in `frontend/app/api/ollama/route.ts`: "Extract subject-predicate-object triples … Normalize entity names to their canonical form … Return results in JSON format as an array of objects with "subject", "predicate", "object" fields". When the JSON doesn't parse, a fallback reads lines like `a - b - c`. `utils/text-processing.ts` lowercases every part, then de-duplicates triples on `subject|predicate|object`. Its chunk size is 20,000 characters with 1,000 overlap, with a comment that it is "Optimized for Gemma3:27b on DGX Spark".

For speed, the playbook's Troubleshooting suggests `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KEEP_ALIVE=30m`, `OLLAMA_MAX_LOADED_MODELS=1` and `OLLAMA_KV_CACHE_TYPE=q8_0`. Stop with the same flags you started with: `./stop.sh --neo4j`.

✓ Checkpoint: the txt2kg UI opens on `localhost:3001`, and you can say which file holds the extraction prompt.

## 5 · Triples on your laptop: lab 02

Lab 02 is txt2kg in one screen of Python. The input is the "Basic idea" section of the three playbooks in this module. The system prompt is txt2kg's, word for word. The parser, lowercasing and de-duplication follow txt2kg's code. The lab then builds the graph and asks it questions. Three LLM calls, 300 tokens each.

```bash
# on: laptop
.venv/bin/python week25/19_multi_agent_rag_kg/labs/lab02_txt2kg_mini.py
```

**Expected output** (captured on this Mac, LAPTOP STAND-IN `gemma4:12b`, not txt2kg's `llama3.1:8b` on a Spark. Extraction is sampled, so your triples will differ.)

```
▣ STEP 2 · extract triples — one LLM call per chunk, txt2kg's system prompt
◆ multi-agent-chatbot   7 triples · 300 tok in 23.3s (gemma4:12b) · reply cut at 300 tokens, kept the complete objects
◆ txt2kg                8 triples · 300 tok in 35.3s (gemma4:12b) · reply cut at 300 tokens, kept the complete objects
◆ rag-ai-workbench      7 triples · 300 tok in 48.1s (gemma4:12b) · reply cut at 300 tokens, kept the complete objects

▣ STEP 3 · normalise + de-duplicate (txt2kg: lowercase, then mergeTriples on subject|predicate|object)
◆ 22 raw triples → 22 unique · 33 entities
│ supervisor agent                is powered by               gpt-oss-120b                        multi-agent-chatbot
│ supervisor agent                orchestrates                specialized downstream agents       multi-agent-chatbot
│ ollama                          is a                        local llm inference engine          txt2kg
│ neo4j                           is a                        graph database                      txt2kg
│ nvidia ai workbench             is used to clone and run    pre-built agentic rag application   rag-ai-workbench
│ …

▣ STEP 4 · the graph: hubs, entities that link documents, and a path
│ entity                             links  mentioned in
│ pre-built agentic rag application  4      rag-ai-workbench
│ supervisor agent                   3      multi-agent-chatbot
│ playbook                           2      multi-agent-chatbot, txt2kg
◆ entities that appear in more than one playbook: playbook
◆ connected components: 11 (1 would mean every fact is reachable from every other)
→ no multi-hop path between the top hubs: the extracted facts do not share entity names
→ wrote week25/19_multi_agent_rag_kg/.runs/triples.json and triples.cypher
```

The triples read well. The *graph* is poor: 22 facts in 11 islands, and the only entity that links two documents is the useless word "playbook". Two runs on this Mac gave the same pattern. That is the real lesson of txt2kg, and you can see it here in a minute rather than after an hour of uploads:

1. **Entity resolution decides graph quality.** "agentic retrieval-augmented generation" and "pre-built agentic rag application" are one thing to you and two nodes to the graph. Lowercasing is not enough. Real pipelines add an alias or merge step, a schema of allowed entity types, or a bigger model (the txt2kg `--vllm` stack uses Nemotron Super 49B).
2. **Token budget truncates.** Every reply hit the 300-token cap. The parser keeps the complete JSON objects, which is why txt2kg's own route allows `maxTokens = 4096`.
3. **Output you can load.** `triples.cypher` contains `MERGE` statements. Paste them into Neo4j Browser on the `--neo4j` stack (`localhost:7474`) to see the islands for yourself.

✓ Checkpoint: you ran lab 02 and can explain why its graph has more than one component.

## 6 · Agentic RAG in AI Workbench

The third playbook is not a Compose stack. It is an **NVIDIA AI Workbench** project ([workbench-example-agentic-rag](https://github.com/NVIDIA/workbench-example-agentic-rag)) with a Gradio chat. The "agentic" part is a loop around retrieval: a **router** chooses how to answer each query, then the system "evaluates responses for relevancy and hallucination, and iterates through evaluation and generation cycles". The **Monitor** tab shows those decisions.

The steps are GUI steps on the Spark's desktop (playbook Steps 1–7):

1. Open **NVIDIA AI Workbench** → **Begin Installation** → **Let's Get Started**. If you see "container tool failed to reach ready state … docker is not running", reboot and reopen.
2. Get two keys: an NVIDIA API key from NGC with **Public API Endpoints** permission, and a Tavily key.
3. **Local** → **Clone Project** → `https://github.com/NVIDIA/workbench-example-agentic-rag` → **Clone**.
4. In the yellow banner click **Configure**, then enter `NVIDIA_API_KEY` and `TAVILY_API_KEY`.
5. **Environment → Project Container → Apps → Chat**. Ask `How do I add an integration in the CLI?`, then follow the in-app quickstart to upload the sample dataset.

> ⚠ **Sovereignty check.** With its defaults this project calls NVIDIA-hosted endpoints and Tavily web search, so queries and retrieved text leave the Spark. The playbook says it "can work with NVIDIA-hosted API endpoints or self-hosted models". For a fully local version, point it at a model on your Spark (Modules 05–08) and drop web search. Enter the keys only in Workbench's **Project Secrets**, never in code or a notebook.

✓ Checkpoint: the Gradio chat answers a sample query and the Monitor tab shows the router's decision, or you can name the two external services the default setup calls.

## 7 · Retrieval you can inspect: lab 03

Every RAG system is "retrieve, then generate", and retrieval sets the ceiling on what generation can get right. Lab 03 builds the retriever with nothing but the standard library: TF-IDF over every section of every DGX Spark playbook README. It scores itself on eight questions with known answers, then gives the top 3 sections to an LLM that has to cite them, and checks the citations.

```bash
# on: laptop
.venv/bin/python week25/19_multi_agent_rag_kg/labs/lab03_playbook_finder.py
.venv/bin/python week25/19_multi_agent_rag_kg/labs/lab03_playbook_finder.py --ask "serve a VLM for live video" --no-llm
```

**Expected output** (captured on this Mac. Retrieval is arithmetic and identical everywhere. The answer is from the LAPTOP STAND-IN.)

```
▣ STEP 1 · index: split every playbook README on '## ' headings
◆ 65 playbooks · 1510 sections · 9,030 distinct terms

▣ STEP 2 · eval: eight questions with known answers
│    question                                        top 3 playbooks                                       score
│ ─  ──────────────────────────────────────────────  ────────────────────────────────────────────────────  ─────
│ ≈  fine-tune a vision language model on my own im  live-vlm-webui, vlm-finetuning, unsloth               0.25
│ ✓  high-throughput LLM serving with tensor parall  vllm, connect-two-sparks, trt-llm                     0.24
│ ✓  turn my documents into a knowledge graph I can  txt2kg, llms, vss                                     0.35
│ ✓  run Claude Code against a local model           local-coding-agent, cli-coding-agent, vscode          0.52
│ ✓  cable two Sparks together for distributed work  connect-two-sparks, connect-multiple-sparks, connec…  0.38
│ ✓  sandbox and govern an AI agent's network and f  openshell, tailscale, nemoclaw-applications           0.31
│ ✓  a supervisor agent that delegates to coding an  multi-agent-chatbot, cli-coding-agent, rag-ai-workb…  0.24
│ ≈  code completion in VS Code from a model on my   vscode, vibe-coding, local-coding-agent               0.34
◆ hit@1 = 6/8 · hit@3 = 8/8   (✓ right at #1 · ≈ right in top 3 · ✕ missed)

▣ STEP 3 · retrieve for: 'Which playbook should I use to build a knowledge graph from my PDFs and ask it questions?'
[1] 0.258  Build Knowledge Graphs with txt2kg  —  nvidia/playbook-txt2kg/README.md § Step 5. Upload documents and build knowledge graphs
[2] 0.136  Set Up Example NemoClaw Agents  —  nvidia/playbook-nemoclaw-applications/README.md § Step 2. Agent prompt
[3] 0.129  Deploy a Video Search and Summarization Agent  —  nvidia/playbook-vss/README.md § Step 10. Next steps

▣ STEP 4 · grounded answer: the LLM may use ONLY these three sources, and must cite them
· ANSWER  To build a knowledge graph and ask questions about your documents, you should use the **Build Knowledge Graphs with txt2kg** playbook [1]. This process involves uploading documents, extracting triples, and using a query interface where the LLM generates responses based on enriched graph context [1].
◆ LAPTOP STAND-IN (not Spark numbers) · gemma4:12b · TTFT 6268 ms · 58 tok in 9.1s · 20.2 tok/s
✓ every citation [1] points at a source it was given: nvidia/playbook-txt2kg/README.md § Step 5. Upload documents and build knowledge graphs
```

Read the misses, not just the score. The two ≈ rows lost to playbooks that share the words ("VLM", "VS Code") but not the intent. Bag-of-words can't tell "fine-tune a VLM" from "stream video to a VLM". That is the gap the chatbot's Qwen3-Embedding vectors are meant to close.

Then read the answer. It is **grounded** (one real citation, checked), but not **complete**: the question said PDFs, and txt2kg's playbook lists markdown, text and CSV. The chatbot's RAG agent is the playbook that takes a PDF. Citations tell you where a claim came from. They don't tell you what the retriever missed.

✓ Checkpoint: you can explain both ≈ rows, and why a correctly cited answer can still be the wrong recommendation.

## Labs — run them here

**labs/lab01_stack_map.py** — Parse the chatbot and txt2kg Compose files: every service, host port and GPU, the port collisions with earlier modules, and each stack's memory.

**labs/lab02_txt2kg_mini.py** — txt2kg in miniature: extract triples from three playbooks with txt2kg's own prompt, merge them, and query the graph.

**labs/lab03_playbook_finder.py** — Stdlib TF-IDF retrieval over every playbook README, scored with hit@k, with an LLM answer whose citations are checked.

Lab 01 is offline apart from its read-only Spark step. Labs 02 and 03 use the Spark's Ollama when it answers, else the laptop's (labelled LAPTOP STAND-IN).

## Try it yourself

**Exercise 19 — the graph behind GraphRAG.** Open `week25/19_multi_agent_rag_kg/exercises/ex19_graph_context.py`. Its 13 triples describe the chatbot's wiring (Section 2), with the messy casing and duplicates an extractor produces. It has three `TODO`s:

1. `normalise(triple)`: lowercase, strip, collapse inner whitespace.
2. `build_graph(triples)`: de-duplicate in first-seen order, and build neighbour sets in both directions.
3. `path(graph, a, b)`: breadth-first shortest path, or `None`.

```bash
# on: laptop
.venv/bin/python week25/19_multi_agent_rag_kg/exercises/ex19_graph_context.py
```

**Expected output** (once all three TODOs are done)

```
✓ normalise: lowercase · strip · collapse inner spaces
✓ build_graph: 13 raw → 11 unique · backend links frontend, supervisor agent, postgres
✓ path: frontend → milvus in 4 hops · deepseek-coder → qwen3-embedding in 5 · unknown → None

▣ your graph, applied: the context GraphRAG adds for 'How does a chat reach the vector database?'
│ (frontend) -[sends chats to]-> (backend)
│ (backend) -[runs]-> (supervisor agent)
│ (supervisor agent) -[calls tool]-> (search_documents)
│ (search_documents) -[retrieves from]-> (milvus)
```

<details><summary>Hint — why does "13 raw → 11 unique" need normalise() first?</summary>

`("Backend", "stores chats in", "Postgres")` and `(" backend ", "Stores Chats In", "postgres")` are the same fact. De-duplicate the normalised tuples. `list(dict.fromkeys(...))` keeps first-seen order.

</details>

<details><summary>Stretch — fix lab 02's islands</summary>

Add an alias table to lab 02 (`{"pre-built agentic rag application": "agentic rag", "retrieval-augmented generation": "rag", …}`) and apply it in `norm()`. How many connected components are left? Then load `.runs/triples.cypher` into Neo4j and count them with Cypher.

</details>

✓ Checkpoint: all three checker lines are ✓.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `address already in use` or `port is already allocated` on :11434 (txt2kg) | The Spark's own Ollama holds the port: `sudo systemctl stop ollama`, then `./start.sh` again (lab 01, Step 2) |
| `port is already allocated` on :8000 (chatbot) | A vLLM/NIM container from Modules 05–06 is running: `docker ps`, then `docker stop <name>` |
| Chatbot containers unhealthy or OOM | Stop other GPU work (`nvidia-smi`), switch the supervisor to gpt-oss-20B (playbook Step 8), flush the cache: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` |
| `qwen2.5-vl` shows unhealthy right after `up` | Expected while it starts, per the playbook. If image answers still fail, check the model name mismatch noted in Section 2 |
| Model download stops partway | Delete the partial file under `models/`, then run `./model_download.sh` again |
| "Cannot access gated repo" during download | Run `hf auth login` on the Spark and request access to the model on Hugging Face |
| txt2kg UI shows an initializing banner | The LLM is still loading. For `--vllm`, follow `docker logs vllm-service -f` (30+ minutes on first start) |
| AI Workbench: 401 / 403 | Wrong key, or the NGC key lacks **Public API Endpoints**. Replace it in Project Secrets and restart |
| Lab 02 parses 0 triples | The model didn't return JSON. Try `--laptop-model gemma4:12b`, or a bigger model on the Spark |

## Next

Continue to [Lab 20 — capstone: a sovereign hotel agent](../20_capstone_sovereign_agent/TUTORIAL.md): fine-tune, serve, route through the gateway and run a NAT agent in a sandbox, with the retrieval and graph ideas from this module as its tools.

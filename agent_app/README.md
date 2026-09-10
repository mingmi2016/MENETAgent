# MENET Agent Application and MCP Server

This directory contains the product layer around the MENET algorithm:

```text
agent_app/
├── agent/       # FastAPI API, intent parsing, workflow, state and tools
├── agent/mcp_server.py # MCP adapter for Codex/Claude; no embedded LLM
├── web/         # Researcher-facing conversational interface
├── data/        # Shared demo datasets and local benchmark metadata
├── scripts/     # API and Celery worker launchers
├── tests/       # Agent integration and workflow tests
├── runs/        # SQLite state and task outputs (not committed)
└── uploads/     # User datasets (not committed)
```

The application depends on the sibling `../MENET` algorithm directory. The startup scripts set `MENET_CORE_ROOT` and `PYTHONPATH` explicitly, so the service does not depend on the shell's current directory.

## Local Service

```bash
cd /home/mingmi/workspace/MenetAgent
./agent_app/scripts/start_agent.sh
```

The default address is `http://127.0.0.1:8010`. Override it with `MENET_AGENT_HOST` and `MENET_AGENT_PORT`.

The current 8010 service provides the REST/Web Agent interfaces. A local `stdio` MCP server is available now. It needs no login or Bearer Token because Codex starts it as a local child process. Streamable HTTP is deferred; a future remote endpoint must validate a Bearer token or OAuth identity rather than relying on the client setting alone.

## Local MCP

Start it manually for a protocol check:

```bash
cd /home/mingmi/workspace/MenetAgent
./agent_app/scripts/start_mcp.sh
```

For Codex, create a local STDIO MCP entry with command `/home/mingmi/workspace/MenetAgent/agent_app/scripts/start_mcp.sh`; do not configure a URL or Bearer token. The server exposes dataset inspection, asynchronous training, new-genotype prediction, evaluation, explanation, task polling, cancellation and temporary-task cleanup.

MCP state is held in memory for the lifetime of the local process. No user account, chat history or LLM call is involved. If `output_dir` is omitted, computation artifacts use `/tmp/menet-mcp/<task_id>` and can be removed with `menet_cleanup_task`. An explicitly supplied result directory is never deleted by that cleanup tool.

For Redis/Celery execution:

```bash
cd /home/mingmi/workspace/MenetAgent
MENET_QUEUE_BACKEND=celery ./agent_app/scripts/start_worker.sh
MENET_QUEUE_BACKEND=celery ./agent_app/scripts/start_agent.sh
```

## Responsibilities

- **Web Agent**: understand supported natural-language requests, use an LLM when configured, maintain conversation context, and select an allow-listed workflow.
- **MCP Server**: expose structured MENET tools to external AI clients. The external client supplies the LLM, planning and conversation context; this server does not call another LLM.
- **Shared services**: validate parameters and select an allow-listed workflow.
- Isolate users, conversations, datasets, tasks, models and artifacts.
- Run MENET jobs asynchronously and report real progress and timing.
- Restore historical conversations and their task results.
- Register completed training runs as model assets with metrics and provenance.
- Let users explicitly select, rename and archive models for prediction, evaluation and explanation; fall back to the latest compatible model only when none is selected.

The two entry points must call shared application services directly. MCP should not call the public REST endpoints through loopback, and the MCP layer must not duplicate MENET workflow logic.

For the local stdio MCP mode, there is no login, conversation history, user memory or long-term user file library. The caller provides accessible input data for the current analysis. MENET may create temporary run files while training, then clean them according to the retention policy. MCP returns task status, metric summaries, warnings, errors and artifact references to Codex or Claude; it does not persist a chat session.

MENET model definitions and training mathematics belong in `../MENET`. Application concerns such as HTTP, SQLite, uploads, task state and web rendering belong here.

## Configuration

The Web UI contains an **AI Model** settings panel. It can discover models, persist the selected endpoint under `runs/llm-settings.json`, and test a real JSON response. Two model roles are supported: the intent model handles request classification and parameter extraction, while the analysis model explains MENET results. Ollama does not require an API key. OpenAI-compatible, Anthropic-compatible, and mixed third-party endpoints can be configured; API keys are never returned by the settings API and the local settings file is mode `0600`.

For a third-party relay, enter either the service root or its `/v1` URL; the client normalizes the model-list, Anthropic Messages, and OpenAI Chat Completions paths. Auto mode selects Anthropic `/v1/messages` for Claude models and OpenAI `/v1/chat/completions` for other models. Keep the real key in the ignored local settings file or `MENET_LLM_API_KEY`; never add it to this repository. A visible model name does not guarantee an active upstream channel, so always run the connection test before enabling a model for users.

Without an enabled or reachable LLM, the service uses the deterministic rule parser. The LLM may extract intent and explain results, but workflow validation and execution remain controlled by application code.

## Model assets

Successful training and report tasks automatically create a model record backed by the immutable task output directory. Existing successful runs are registered during service startup. The Web UI shows the selected model's trait, training sample/SNP counts and test R². Archiving removes a model from normal selection without deleting its reproducibility artifacts.

The REST surface includes `GET /api/v1/models`, `GET /api/v1/models/{model_id}`, `PATCH /api/v1/models/{model_id}` and `DELETE /api/v1/models/{model_id}`. Prediction, evaluation and explanation requests can pass `model_id`; the task then records the model ID and name and rejects trait mismatches.

Every completed workflow writes `quality_report.json`. Deterministic checks currently cover blocking data errors, small effective sample sizes, negative or weak test R², training/validation overfitting gaps, constant phenotypes, duplicate genotype profiles, and overlap between train/validation/test sample IDs. New training runs also fit genomic ridge, random forest, and (when installed) XGBoost baselines on the identical train/test split and report MENET's R² gain over each baseline. The model panel compares repeated runs for the same user, dataset and trait; two or more runs are required before it reports a stability range.

The **重复验证** action submits 2–5 independent runs with different split and training seeds. Runs reuse the selected model's trait, dataset, epoch count and split ratios, and each seed is recorded in `task.json`. The default single GPU worker executes them sequentially to avoid concurrent GPU memory exhaustion.

The current development target is this WSL workstation. Generic installers, server multi-tenancy and a C++ wrapper are deferred until the local Agent and MCP workflows are complete.

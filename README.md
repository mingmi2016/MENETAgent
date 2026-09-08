# MENET Agent and MCP Server

This project provides two complementary ways to use MENET:

- **MENET Agent**: a web-facing conversational Agent that combines an LLM, context management, planning and controlled MENET workflows.
- **MENET MCP Server**: an AI-facing protocol adapter for Codex, Claude and other MCP clients. It exposes structured MENET tools and does not contain a second LLM.

MENET is a mixed-effects deep neural network architecture for multi-environment agronomic traits prediction. The algorithm and the Agent application are intentionally separated:

```text
MenetAgent/
├── MENET/          # Algorithm core, model definitions, training utilities and configs
├── agent_app/      # FastAPI Agent, web UI, tests, demos and deployment scripts
├── .venv/          # Shared local Python environment (not committed)
└── README.md
```

## Overall Architecture

The two entry points share the same application services, workflow and MENET core:

```text
Web user ── REST ──> MENET Agent ── LLM/context/planning ──┐
                                                           ├─> Application services
Codex/Claude ─ MCP ─> MENET MCP Server ─ structured tools ─┘   ├─ dataset/task/model/result
                                                               └─ workflow -> worker/GPU -> MENET
```

The web Agent is responsible for natural-language interaction and conversation context. The MCP Server is intentionally narrower: the external AI already provides the LLM, planning and conversation context, so MCP only validates arguments, submits or queries tasks, and returns structured results. In local stdio mode it does not require login or persist chat history and user files; MENET may still use temporary files during computation. Neither entry point calls the other over HTTP; both call the shared service layer.

REST remains available for the web UI, C++ clients, file upload/download and conventional integrations. MCP is the AI integration surface, not a replacement for REST.

For Codex connections, use `STDIO` for a local single-user MCP process, which normally needs no login or Bearer Token. For a remote Streamable HTTP MCP endpoint, Codex can read a variable such as `MCP_BEARER_TOKEN` and send it as `Authorization: Bearer <token>`. The server must implement token validation; setting the variable in Codex alone does not secure the service. The 8010 service is REST/Web Agent. Local stdio MCP is implemented in `agent_app/agent/mcp_server.py` and starts with `./agent_app/scripts/start_mcp.sh`; it does not use the 8010 port.

## Current Development Scope

The project is local-first. The current milestone is a reliable end-to-end system on the existing WSL workstation:

```text
Web UI -> MENET Agent -> local Ollama -> MENET workflow -> local GPU
Codex/Claude -> local MCP Server -> the same MENET workflow
```

Work on generic Windows installers, arbitrary-computer compatibility, production multi-tenancy, cloud deployment and a C++ desktop wrapper is intentionally deferred. These are packaging and deployment concerns and should not distract from validating the Agent, model management, new-material prediction, result quality and MCP integration.

## Start the Agent

```bash
cd /home/mingmi/workspace/MenetAgent
./agent_app/scripts/start_agent.sh
```

Open:

- Web application: http://127.0.0.1:8010/app/
- OpenAPI documentation: http://127.0.0.1:8010/docs

The launcher runs from `agent_app/`, adds `MENET/` as the explicit algorithm dependency, and stores local state under `agent_app/runs/` and `agent_app/uploads/`.

## Run Tests

```bash
cd /home/mingmi/workspace/MenetAgent
source .venv/bin/activate
pytest -q
```

## Install Dependencies

```bash
cd /home/mingmi/workspace/MenetAgent
python3 -m venv .venv
.venv/bin/python -m pip install -r agent_app/requirements.txt
```

The Web UI exposes local LLM settings. Ollama can be used without an API key; when the LLM is disabled or unavailable, the application falls back to deterministic intent parsing. Environment-variable defaults are documented in `agent_app/.env.example`.

Completed training runs are registered in the local model library. Users can inspect metrics and provenance, choose the exact model used by prediction/evaluation/explanation, rename it, or archive it without deleting the underlying run artifacts.

## Documentation

- [MENET algorithm](MENET/README.md)
- [Agent application](agent_app/README.md)
- [Agent architecture](agent_app/docs/MENET-Agent-Design.md)
- [MCP integration](agent_app/docs/MENET-Agent-Design.md#mcp-server-and-external-ai)
- [Local GPU benchmark](agent_app/docs/Local-Benchmark.md)
- [Usability checklist](agent_app/docs/Usability-Checklist.md)

The current profile selector is for local development and is not authentication. Add verified accounts, HTTPS, authorization, quotas, and production storage before exposing the service outside the local machine.

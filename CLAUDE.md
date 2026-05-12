# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the AutoGPT monorepo containing:
- **`autogpts/forge/`** — Forge: a reusable agent SDK and template for building new agents
- **`autogpts/autogpt/`** — The original AutoGPT agent (uses Forge as a dependency)
- **`benchmark/`** — `agbenchmark`: pytest-based benchmarking harness for agent protocol–compatible agents
- **`frontend/`** — Flutter web UI for interacting with agents
- **`arena/`** — JSON entries for the AutoGPT Arena leaderboard

## Root CLI

The primary entrypoint for all developer tasks is `./run` (a thin wrapper around `cli.py`):

```bash
./run setup                          # Install system dependencies
./run agent create YOUR_AGENT_NAME   # Scaffold a new agent from Forge template
./run agent start YOUR_AGENT_NAME    # Start agent on http://localhost:8000
./run agent stop                     # Kill the process on port 8000
./run benchmark start YOUR_AGENT_NAME
./run benchmark categories list
./run benchmark tests list
./run arena enter YOUR_AGENT_NAME    # Submit to leaderboard (requires GitHub token)
```

## Forge Agent Development

Forge agents live in `autogpts/forge/`. Dependencies are managed with Poetry.

```bash
cd autogpts/forge
poetry install
poetry run python -m forge            # Start agent server (port 8000)
poetry run agbenchmark                # Run benchmark against the running agent (port 8080)
poetry run pytest                     # Run unit tests
```

### Linting & formatting (Forge)
```bash
poetry run black forge/
poetry run isort forge/
poetry run flake8 forge/
poetry run mypy forge/
```

Run a single test:
```bash
poetry run pytest forge/sdk/agent_test.py::test_create_task -v
```

## Benchmark

```bash
cd benchmark
poetry install
poetry run agbenchmark start          # Run all challenges (agent must be running on :8000)
poetry run agbenchmark start --test TestName
poetry run pytest tests/              # Unit tests for the benchmark package itself
```

## Frontend (Flutter)

```bash
cd frontend
flutter pub get
flutter run -d chrome                 # Dev server
flutter build web                     # Build → frontend/build/web (served by the agent)
flutter test
```

## Architecture

### Agent Protocol

All agents expose a REST API at `http://localhost:8000/ap/v1/` conforming to the [Agent Protocol](https://agentprotocol.ai/) spec. The benchmark and frontend both communicate exclusively through this API. Key routes (defined in `autogpts/forge/forge/sdk/routes/agent_protocol.py`):
- `POST /agent/tasks` — create a task
- `POST /agent/tasks/{task_id}/steps` — execute next step
- `GET /agent/tasks/{task_id}/artifacts` — list output files
- `POST /agent/tasks/{task_id}/artifacts` — upload artifact

### Forge SDK (`autogpts/forge/forge/sdk/`)

The SDK is the foundation for all Forge-based agents:

| Module | Role |
|---|---|
| `agent.py` | Base `Agent` class; wires FastAPI app, mounts frontend static files, calls subclass hooks |
| `db.py` | `AgentDB`: SQLAlchemy/SQLite storage for `Task`, `Step`, `Artifact` models |
| `workspace.py` | `Workspace` ABC + `LocalWorkspace`: sandboxed per-task file I/O (prevents directory traversal) |
| `prompting.py` | `PromptEngine`: loads and renders Jinja2 templates from `forge/prompts/<model>/` |
| `model.py` | Pydantic models for the agent protocol (`Task`, `Step`, `Artifact`, etc.) |
| `forge_log.py` | Colorised logger |

### ForgeAgent & Actions (`autogpts/forge/forge/`)

`forge/agent.py` contains `ForgeAgent`, a subclass of the SDK `Agent`. The key customization point is `execute_step()`.

Actions are defined with the `@action` decorator and **auto-discovered** at startup: `ActionRegister` (`forge/actions/registry.py`) globs all `.py` files under `forge/actions/`, imports them, and registers any function decorated with `@action`. Adding a new `.py` file in that directory is sufficient to register new actions — no manual registration needed.

```python
from forge.sdk import Agent
from forge.actions import action, ActionParameter

@action(
    name="my_action",
    description="What it does",
    parameters=[ActionParameter(name="arg", description="...", type="string", required=True)],
    output_type="string",
)
async def my_action(agent: Agent, task_id: str, arg: str) -> str:
    ...
```

### LLM (`forge/llm.py`)

All LLM calls go through **LiteLLM** via `chat_completion_request()`, which supports any provider. Embeddings use the OpenAI API directly. Both are wrapped with tenacity retry logic (3 attempts, exponential backoff).

### Memory (`forge/memory/`)

`MemStore` is the abstract interface; `ChromaMemStore` is the ChromaDB-backed implementation used by default. It stores embeddings keyed by task ID.

### Prompt Templates (`forge/prompts/`)

Jinja2 templates organised by model name (e.g. `gpt-3.5-turbo/`) with reusable technique partials in `prompts/techniques/` (`expert.j2`, `chain-of-thought.j2`, `few-shot.j2`). `PromptEngine` is initialised with a model name and performs fuzzy-matched template lookup.

### Frontend Architecture

Flutter web app using MVVM with Provider:
- **Views** (`lib/views/`) — pure UI widgets
- **ViewModels** (`lib/viewmodels/`) — `TaskViewModel`, `ChatViewModel`, `SkillTreeViewModel`, etc.
- **Services** (`lib/services/`) — `TaskService`, `ChatService`, `BenchmarkService`, etc. make HTTP calls to the agent protocol API via `RestApiUtility`

Firebase is used for authentication (Google + GitHub). The built Flutter app is served as static files by the FastAPI agent server at `/app/`.

### Benchmark Challenges

Challenges are JSON files in `benchmark/agbenchmark/challenges/`. They are discovered by pytest. Each challenge defines inputs, expected outputs, and scoring criteria. The benchmark runs tasks against the agent at `http://localhost:8000/ap/v1` using the `AgentApi` client.

## Environment Setup

Each agent subproject requires a `.env` file. Forge will copy `.env.example` automatically on first run. Minimum required variable:

```
OPENAI_API_KEY=sk-...
DATABASE_STRING=sqlite:///agent.db
AGENT_WORKSPACE=./workspace
```

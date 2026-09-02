# Agent Core

This package is the first non-LLM layer of MENET Agent. It defines structured
tasks, a controlled workflow, and the allow-listed MENET tool boundary.

Run a validation-only task from the repository root:

```python
from agent import MenetTask, MenetWorkflow, TaskIntent

task = MenetTask(
    intent=TaskIntent.INSPECT_DATA,
    trait="culmlength",
    dataset_dir="data",
)
result = MenetWorkflow().run(task)
print(result.to_dict())
```

The training tools call the existing MENET computation modules and save their
artifacts under the task output directory. If a required artifact or input is
missing, the tool returns a structured failure instead of claiming completion.

The current bundled example contains too few SNPs for the legacy convolutional
network and is intended to document file format only. Use a real dataset with
enough markers for training.

Start the HTTP service from the repository root:

```bash
source .venv/bin/activate
uvicorn agent.api:app --host 127.0.0.1 --port 8010
```

Or use the repository launcher:

```bash
./scripts/start_agent.sh
```

Submit a validation task with `POST /api/v1/tasks`, then query
`GET /api/v1/tasks/{task_id}`. Long-running work is executed in a background
worker and persisted in `runs/agent.db`.

For production-style queue execution, start Redis and run:

```bash
MENET_QUEUE_BACKEND=celery ./scripts/start_worker.sh
```

Then start the API with the same `MENET_QUEUE_BACKEND=celery` setting. The
default `thread` backend remains useful for local development.

Upload a dataset instead of placing files manually under the service root:

```bash
curl -X POST "http://localhost:8010/api/v1/datasets/upload?trait=culmlength" \
  -F "genotype=@genotype.csv" -F "phenotype=@culmlength.csv"
```

Use `GET /api/v1/tasks/{task_id}/explanation` after completion to obtain a
human-readable explanation. The endpoint uses the configured LLM when
available and otherwise returns a deterministic summary.

The natural-language entry point is `POST /api/v1/chat`:

```json
{
  "message": "请用 GPU 自动划分数据训练 MENET，性状是 culmlength，并解释重要 SNP",
  "dataset_dir": "data",
  "output_dir": "runs/example"
}
```

The first version includes a rule-based parser so it can run without an API
key. It accepts the same `ParsedIntent` contract as a future LLM parser. The
LLM should extract intent and arguments only; `MenetWorkflow` remains the
authority for validation and execution.

To enable an OpenAI-compatible model endpoint, set these variables before
starting the service:

```bash
export MENET_LLM_BASE_URL="https://your-endpoint/v1"
export MENET_LLM_API_KEY="your-key"
export MENET_LLM_MODEL="your-model"
```

Without these variables the service uses the built-in rule parser. If the
configured endpoint fails, it falls back to that parser so a transient LLM
outage does not bypass the deterministic validation layer.

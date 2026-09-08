"""Optional Celery entry point for production workers."""

import os
from pathlib import Path

from celery import Celery


celery = Celery(
    "menet_agent",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
)


@celery.task(name="menet_agent.run_task")
def run_task(task_data):
    from .runner import run_task as execute

    app_root = Path(__file__).resolve().parents[1]
    database_path = Path(os.environ.get("MENET_AGENT_DB", "runs/agent.db"))
    if not database_path.is_absolute():
        database_path = app_root / database_path
    return execute(task_data, str(database_path))

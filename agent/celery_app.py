"""Optional Celery entry point for production workers."""

import os

from celery import Celery


celery = Celery(
    "menet_agent",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
)


@celery.task(name="menet_agent.run_task")
def run_task(task_data):
    from .runner import run_task as execute

    return execute(task_data, os.environ.get("MENET_AGENT_DB", "runs/agent.db"))

"""Shared task runner used by local and queued workers."""

import json
from pathlib import Path
from typing import Any, Dict

from .models import MenetTask
from .store import TaskStore
from .workflow import MenetWorkflow
from .quality import write_quality_report


def run_task(task_data: Dict[str, Any], database_path: str) -> Dict[str, Any]:
    task = MenetTask(**task_data)
    store = TaskStore(database_path)
    cancellation = Path(task.output_dir) / "cancel.requested"
    if cancellation.is_file():
        result = {"task_id": task.task_id, "status": "cancelled", "errors": [], "warnings": []}
        store.update(task.task_id, "cancelled", result)
        return result
    store.update(task.task_id, "running")
    try:
        output_dir = Path(task.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "task.json").write_text(
            json.dumps(task.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result = MenetWorkflow().run(
            task,
            on_status=lambda status: store.update(task.task_id, status.value),
        ).to_dict()
        if cancellation.is_file():
            result["status"] = "cancelled"
            result["errors"] = []
            result.setdefault("warnings", []).append("任务已按用户请求停止")
        result["quality_report"] = write_quality_report(task, result)
        validation = next((step.get("data", {}) for step in result.get("steps", []) if step.get("name") == "validate_dataset"), {})
        if validation.get("matched_sample_count") is not None and validation.get("snp_count") is not None:
            store.update_dataset_stats(task.dataset_dir, validation["matched_sample_count"], validation["snp_count"])
        if result.get("status") == "completed" and task.intent.value in {"train_model", "generate_report"}:
            try:
                result["model"] = store.register_model(task.to_dict(), result)
            except ValueError as exc:
                result.setdefault("warnings", []).append(f"模型登记失败：{exc}")
        store.update(task.task_id, result["status"], result)
        return result
    except Exception as exc:
        result = {"task_id": task.task_id, "status": "failed", "errors": [str(exc)]}
        store.update(task.task_id, "failed", result)
        return result

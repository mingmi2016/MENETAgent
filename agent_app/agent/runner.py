"""Shared task runner used by local and queued workers."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .models import MenetTask
from .store import TaskStore
from .workflow import MenetWorkflow
from .quality import write_quality_report


def write_json_atomic(path: Path, value: Dict[str, Any]) -> None:
    """Publish complete files; readers must never see half-written JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_file_task(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute-node runner: deliberately never creates or opens a TaskStore."""
    task = MenetTask(**task_data)
    output_dir = Path(task.output_dir)
    started_at = datetime.now(timezone.utc).isoformat()
    def publish(status: str) -> None:
        write_json_atomic(output_dir / "worker_status.json", {
            "task_id": task.task_id, "status": status, "started_at": started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
    publish("running")
    try:
        if (output_dir / "cancel.requested").is_file():
            result = {"task_id": task.task_id, "status": "cancelled", "steps": [], "errors": [], "warnings": []}
        else:
            result = MenetWorkflow().run(task, on_status=lambda status: publish(status.value)).to_dict()
            if (output_dir / "cancel.requested").is_file():
                result["status"] = "cancelled"
                result["errors"] = []
                result.setdefault("warnings", []).append("任务已按用户请求停止")
            result["quality_report"] = write_quality_report(task, result)
    except Exception as exc:
        result = {"task_id": task.task_id, "status": "failed", "errors": [str(exc)], "warnings": []}
    result["worker_timing"] = {
        "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(output_dir / "result.json", result)
    publish(result["status"])
    return result


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
        result = store.finalize_task(task.to_dict(), result)
        write_json_atomic(output_dir / "result.json", result)
        return result
    except Exception as exc:
        result = {"task_id": task.task_id, "status": "failed", "errors": [str(exc)]}
        try:
            Path(task.output_dir).mkdir(parents=True, exist_ok=True)
            (Path(task.output_dir) / "result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass
        store.update(task.task_id, "failed", result)
        return result

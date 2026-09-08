"""Stateless local MCP entry point for MENET tools."""

import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from .models import MenetTask, TaskIntent
from .quality import write_quality_report
from .tools import MenetTools
from .workflow import MenetWorkflow


mcp = FastMCP(
    "MENET",
    instructions=(
        "MENET predicts one agronomic trait per trained model. Inspect data before training. "
        "Prediction requires a trained model directory and a new genotype CSV with exactly "
        "the same SNP columns and order as the training genotype file."
    ),
)

_temporary_root = Path("/tmp/menet-mcp").resolve()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="menet-mcp")
_jobs: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _job_snapshot(task_id: str) -> Dict[str, Any]:
    with _lock:
        job = _jobs.get(task_id)
        return dict(job) if job else {"task_id": task_id, "status": "not_found"}


def _artifact_manifest(output_dir: str) -> list[Dict[str, Any]]:
    root = Path(output_dir)
    if not root.is_dir():
        return []
    return [
        {"name": path.name, "path": str(path.resolve()), "size_bytes": path.stat().st_size}
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "cancel.requested"
    ]


def _execute(task: MenetTask) -> None:
    with _lock:
        _jobs[task.task_id].update({"status": "running", "started_at": _now()})
    output = Path(task.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "task.json").write_text(json.dumps(task.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        result = MenetWorkflow().run(
            task,
            on_status=lambda status: _set_job_status(task.task_id, status.value),
        ).to_dict()
        if (output / "cancel.requested").is_file():
            result["status"] = "cancelled"
            result["errors"] = []
            result.setdefault("warnings", []).append("任务已按调用方请求取消")
        result["quality_report"] = write_quality_report(task, result)
    except Exception as exc:
        result = {"task_id": task.task_id, "status": "failed", "errors": [str(exc)], "warnings": []}
    result["artifacts"] = _artifact_manifest(task.output_dir)
    with _lock:
        _jobs[task.task_id].update({
            "status": result["status"], "finished_at": _now(), "result": result,
            "artifacts": result["artifacts"],
        })


def _set_job_status(task_id: str, status: str) -> None:
    with _lock:
        if task_id in _jobs:
            _jobs[task_id]["status"] = status


def _submit_task(task: MenetTask) -> Dict[str, Any]:
    errors = task.validate()
    if errors:
        return {"accepted": False, "status": "invalid", "errors": errors}
    if not task.output_dir:
        task.output_dir = str(_temporary_root / task.task_id)
    job = {
        "task_id": task.task_id,
        "status": "queued",
        "intent": task.intent.value,
        "trait": task.trait,
        "output_dir": str(Path(task.output_dir).resolve()),
        "created_at": _now(),
        "result": None,
        "artifacts": [],
    }
    with _lock:
        _jobs[task.task_id] = job
    _executor.submit(_execute, task)
    return {"accepted": True, **job}


@mcp.tool()
def menet_inspect_dataset(dataset_dir: str, trait: str) -> Dict[str, Any]:
    """Inspect MENET genotype/phenotype files, matching samples, SNPs, missingness, and leakage risks."""
    task = MenetTask(dataset_dir=dataset_dir, trait=trait, intent=TaskIntent.INSPECT_DATA)
    return MenetTools().validate_dataset(task)


@mcp.tool()
def menet_submit_training(
    dataset_dir: str,
    trait: str,
    output_dir: str = "",
    device: str = "auto",
    epochs: Optional[int] = None,
    split_strategy: str = "existing",
    split_seed: int = 42,
    training_seed: int = 42,
    explain_snp: bool = False,
) -> Dict[str, Any]:
    """Submit one-trait MENET training; poll menet_get_task because this can take minutes or hours."""
    task = MenetTask(
        dataset_dir=dataset_dir, trait=trait, output_dir=output_dir, device=device,
        epochs=epochs, split_strategy=split_strategy, explain_snp=explain_snp,
        intent=TaskIntent.TRAIN_MODEL,
        metadata={"split_seed": split_seed, "seed": training_seed},
    )
    return _submit_task(task)


@mcp.tool()
def menet_submit_prediction(
    dataset_dir: str,
    trait: str,
    model_output_dir: str,
    prediction_genotype_path: str,
    output_dir: str = "",
    device: str = "auto",
) -> Dict[str, Any]:
    """Predict a trait for a new genotype CSV using an explicit trained MENET model directory."""
    task = MenetTask(
        dataset_dir=dataset_dir, trait=trait, output_dir=output_dir,
        source_output_dir=model_output_dir, prediction_genotype_path=prediction_genotype_path,
        device=device, intent=TaskIntent.PREDICT_TRAIT,
    )
    return _submit_task(task)


@mcp.tool()
def menet_submit_evaluation(dataset_dir: str, trait: str, model_output_dir: str) -> Dict[str, Any]:
    """Read metrics and quality evidence from an explicit trained model directory."""
    task = MenetTask(
        dataset_dir=dataset_dir, trait=trait, output_dir="",
        source_output_dir=model_output_dir, intent=TaskIntent.EVALUATE_MODEL,
    )
    return _submit_task(task)


@mcp.tool()
def menet_submit_explanation(dataset_dir: str, trait: str, model_output_dir: str) -> Dict[str, Any]:
    """Read MENET training history and model contribution evidence for an explicit model directory."""
    task = MenetTask(
        dataset_dir=dataset_dir, trait=trait, output_dir="",
        source_output_dir=model_output_dir, intent=TaskIntent.EXPLAIN_MODEL,
    )
    return _submit_task(task)


@mcp.tool()
def menet_get_task(task_id: str) -> Dict[str, Any]:
    """Return current status, structured result, warnings, errors, and artifact paths for an MCP task."""
    return _job_snapshot(task_id)


@mcp.tool()
def menet_cancel_task(task_id: str) -> Dict[str, Any]:
    """Request cooperative cancellation of a queued or running MCP task."""
    job = _job_snapshot(task_id)
    if job["status"] == "not_found":
        return job
    if job["status"] in {"completed", "failed", "cancelled"}:
        return {"task_id": task_id, "status": job["status"], "message": "任务已经结束。"}
    Path(job["output_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(job["output_dir"]) / "cancel.requested").touch()
    return {"task_id": task_id, "status": "cancellation_requested"}


@mcp.tool()
def menet_cleanup_task(task_id: str) -> Dict[str, Any]:
    """Delete a finished task only when it used the default /tmp MENET MCP workspace."""
    job = _job_snapshot(task_id)
    if job["status"] == "not_found":
        return job
    output = Path(job["output_dir"]).resolve()
    if _temporary_root not in output.parents:
        return {"task_id": task_id, "status": "refused", "message": "显式指定的结果目录不会由 MCP 自动删除。"}
    if job["status"] not in {"completed", "failed", "cancelled"}:
        return {"task_id": task_id, "status": "refused", "message": "任务仍在运行。"}
    shutil.rmtree(output, ignore_errors=True)
    with _lock:
        _jobs.pop(task_id, None)
    return {"task_id": task_id, "status": "cleaned"}


if __name__ == "__main__":
    mcp.run(transport="stdio")

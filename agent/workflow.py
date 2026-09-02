"""Controlled MENET workflow orchestration."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .models import MenetTask, TaskIntent, TaskStatus
from .tools import MenetTools


@dataclass
class WorkflowResult:
    task_id: str
    status: TaskStatus
    steps: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "steps": self.steps,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class MenetWorkflow:
    """Run only predefined, validated workflows."""

    def __init__(self, tools: Optional[MenetTools] = None):
        self.tools = tools or MenetTools()

    def run(self, task: MenetTask, on_status: Optional[Callable[[TaskStatus], None]] = None) -> WorkflowResult:
        result = WorkflowResult(task_id=task.task_id, status=TaskStatus.CREATED)
        validation_errors = task.validate()
        if validation_errors:
            result.status = TaskStatus.FAILED
            result.errors.extend(validation_errors)
            return result

        self._step(result, TaskStatus.VALIDATING, "validate_dataset", self.tools.validate_dataset, task, on_status)
        if result.errors:
            result.status = TaskStatus.FAILED
            return result

        if task.intent == TaskIntent.INSPECT_DATA:
            result.status = TaskStatus.COMPLETED
            return result

        if task.intent == TaskIntent.PREDICT_TRAIT:
            self._step(result, TaskStatus.EVALUATING, "predict_trait", self.tools.predict_trait, task, on_status)
            result.status = TaskStatus.FAILED if result.errors else TaskStatus.COMPLETED
            return result

        if task.intent == TaskIntent.EVALUATE_MODEL:
            self._step(result, TaskStatus.EVALUATING, "evaluate_model", self.tools.evaluate_model, task, on_status)
            result.status = TaskStatus.FAILED if result.errors else TaskStatus.COMPLETED
            return result

        if task.intent == TaskIntent.EXPLAIN_MODEL:
            self._step(result, TaskStatus.EXPLAINING, "explain_model", self.tools.explain_model, task, on_status)
            result.status = TaskStatus.FAILED if result.errors else TaskStatus.COMPLETED
            return result

        self._step(result, TaskStatus.PREPARING, "create_split", self.tools.create_split, task, on_status)
        if result.errors:
            result.status = TaskStatus.FAILED
            return result

        steps = [
            (TaskStatus.TRAINING_ENCODER, "train_trait_encoder", self.tools.train_trait_encoder),
            (TaskStatus.BUILDING_RELATEDNESS, "build_relatedness", self.tools.build_relatedness),
            (TaskStatus.TRAINING_MENET, "train_menet", self.tools.train_menet),
            (TaskStatus.EVALUATING, "evaluate_model", self.tools.evaluate_model),
        ]
        if task.explain_snp or task.intent == TaskIntent.EXPLAIN_MODEL:
            steps.append((TaskStatus.EXPLAINING, "explain_model", self.tools.explain_model))

        for status, name, handler in steps:
            self._step(result, status, name, handler, task, on_status)
            if result.errors:
                result.status = TaskStatus.FAILED
                return result

        if task.intent == TaskIntent.GENERATE_REPORT:
            self._step(result, TaskStatus.EXPLAINING, "generate_report", self.tools.generate_report, task, on_status)
        result.status = TaskStatus.COMPLETED
        return result

    @staticmethod
    def _step(result: WorkflowResult, status: TaskStatus, name: str, handler, task: MenetTask, on_status=None) -> None:
        result.status = status
        if on_status:
            on_status(status)
        output = handler(task)
        result.steps.append({"name": name, "status": output.get("status"), "data": output.get("data", {})})
        result.errors.extend(output.get("errors", []))
        result.warnings.extend(output.get("warnings", []))

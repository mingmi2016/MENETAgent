"""Stable data contracts used by the MENET Agent."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


class TaskIntent(str, Enum):
    INSPECT_DATA = "inspect_data"
    TRAIN_MODEL = "train_model"
    PREDICT_TRAIT = "predict_trait"
    EVALUATE_MODEL = "evaluate_model"
    EXPLAIN_MODEL = "explain_model"
    GENERATE_REPORT = "generate_report"


class TaskStatus(str, Enum):
    CREATED = "created"
    VALIDATING = "validating"
    PREPARING = "preparing"
    TRAINING_ENCODER = "training_encoder"
    BUILDING_RELATEDNESS = "building_relatedness"
    TRAINING_MENET = "training_menet"
    EVALUATING = "evaluating"
    EXPLAINING = "explaining"
    COMPLETED = "completed"
    WAITING_FOR_USER = "waiting_for_user"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class MenetTask:
    """A validated, serializable request for one MENET workflow."""

    trait: str
    owner_user_id: str = "user_local"
    dataset_dir: str = "data"
    output_dir: str = "runs"
    source_output_dir: Optional[str] = None
    prediction_genotype_path: Optional[str] = None
    device: str = "auto"
    split_strategy: str = "existing"
    train_ratio: float = 0.7
    valid_ratio: float = 0.15
    test_ratio: float = 0.15
    epochs: Optional[int] = None
    explain_snp: bool = False
    intent: TaskIntent = TaskIntent.TRAIN_MODEL
    task_id: str = field(default_factory=lambda: f"task_{uuid4().hex[:12]}")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.intent, str):
            self.intent = TaskIntent(self.intent)

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.trait.strip():
            errors.append("trait 不能为空")
        if self.device not in {"auto", "cpu", "cuda"}:
            errors.append("device 必须是 auto、cpu 或 cuda")
        if self.split_strategy not in {"existing", "random", "group", "year"}:
            errors.append("split_strategy 不受支持")
        ratios = (self.train_ratio, self.valid_ratio, self.test_ratio)
        if any(r <= 0 or r >= 1 for r in ratios):
            errors.append("数据划分比例必须在 0 和 1 之间")
        elif abs(sum(ratios) - 1.0) > 1e-6:
            errors.append("train_ratio、valid_ratio、test_ratio 之和必须为 1")
        if self.epochs is not None and self.epochs <= 0:
            errors.append("epochs 必须为正整数")
        if self.intent in {TaskIntent.TRAIN_MODEL, TaskIntent.PREDICT_TRAIT, TaskIntent.EVALUATE_MODEL} and self.split_strategy == "existing":
            # The data tool gives the precise missing-file message.
            pass
        return errors

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["intent"] = self.intent.value
        return value

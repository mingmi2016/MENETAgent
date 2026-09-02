"""Intent parsing boundary.

The rule-based parser makes the service usable without an API key. A hosted
LLM parser can implement the same interface and be plugged in later.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .models import TaskIntent


@dataclass
class ParsedIntent:
    intent: Optional[TaskIntent]
    arguments: Dict[str, Any]
    missing_fields: List[str]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value if self.intent else None,
            "arguments": self.arguments,
            "missing_fields": self.missing_fields,
            "confidence": self.confidence,
        }


class IntentParser:
    """Parse the supported MENET intents without executing anything."""

    def parse(self, message: str) -> ParsedIntent:
        text = message.strip().lower()
        if not text:
            return ParsedIntent(None, {}, ["message"], 0.0)

        if any(word in text for word in ("检查", "校验", "能不能用", "数据质量")):
            intent = TaskIntent.INSPECT_DATA
        elif any(word in text for word in ("训练", "train")):
            # Training is the primary workflow when the request also asks for
            # explanation or evaluation.
            intent = TaskIntent.TRAIN_MODEL
        elif any(word in text for word in ("预测", "predict")):
            intent = TaskIntent.PREDICT_TRAIT
        elif any(word in text for word in ("解释", "重要snp", "重要位点", "重要标记")):
            intent = TaskIntent.EXPLAIN_MODEL
        elif any(word in text for word in ("评估", "效果", "r2", "r²")):
            intent = TaskIntent.EVALUATE_MODEL
        elif any(word in text for word in ("报告", "report")):
            intent = TaskIntent.GENERATE_REPORT
        elif any(word in text for word in ("训练", "train", "分析", "menet")):
            intent = TaskIntent.TRAIN_MODEL
        else:
            return ParsedIntent(None, {}, ["intent"], 0.0)

        arguments: Dict[str, Any] = {}
        trait = self._extract_trait(message)
        if trait:
            arguments["trait"] = trait
        if "gpu" in text or "cuda" in text:
            arguments["device"] = "cuda"
        elif "cpu" in text:
            arguments["device"] = "cpu"
        if any(word in text for word in ("自动划分", "随机划分")):
            arguments["split_strategy"] = "random"
        if any(word in text for word in ("重要snp", "重要位点", "解释")):
            arguments["explain_snp"] = True
        epochs = re.search(r"(?:训练|运行)?\s*(\d+)\s*(?:个)?\s*(?:epoch|epochs|轮)", text)
        if epochs:
            arguments["epochs"] = int(epochs.group(1))

        missing = [] if trait else ["trait"]
        return ParsedIntent(intent, arguments, missing, 0.85 if trait else 0.55)

    @staticmethod
    def _extract_trait(message: str) -> Optional[str]:
        patterns = (
            r"(?:性状|表型|trait)\s*(?:是|为|=|:|：)?\s*([\w-]+)",
            r"(?:分析|训练|预测)\s*([a-zA-Z][\w-]*)\s*(?:数据|性状|表型)?",
        )
        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip(" ，,。")
                if value.lower() not in {"menet", "this", "the"}:
                    return value
        return None

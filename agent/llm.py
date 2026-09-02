"""Optional OpenAI-compatible language-model integration.

The service remains usable without credentials. Configure a compatible chat
endpoint with MENET_LLM_BASE_URL, MENET_LLM_API_KEY, and MENET_LLM_MODEL.
"""

import json
import os
import re
from typing import Any, Dict, Optional
from urllib import request

from .intent import IntentParser, ParsedIntent
from .models import TaskIntent


class CompatibleLLMClient:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None, timeout: int = 60):
        self.base_url = (base_url or os.getenv("MENET_LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("MENET_LLM_API_KEY", "")
        self.model = model or os.getenv("MENET_LLM_MODEL", "")
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("LLM 未配置")
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        endpoint = self.base_url if self.base_url.endswith("/chat/completions") else self.base_url + "/chat/completions"
        req = request.Request(endpoint, data=body, headers=headers, method="POST")
        with request.urlopen(req, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"]
        return self._parse_json(content)

    @staticmethod
    def _parse_json(content: str) -> Dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                raise ValueError("LLM 返回内容不是合法 JSON")
            return json.loads(match.group(0))


class AgentIntentParser:
    """Use an LLM when configured, with a deterministic fallback."""

    def __init__(self, client: Optional[CompatibleLLMClient] = None, fallback: Optional[IntentParser] = None):
        self.client = client or CompatibleLLMClient()
        self.fallback = fallback or IntentParser()

    def parse(self, message: str) -> ParsedIntent:
        if not self.client.enabled:
            return self.fallback.parse(message)
        try:
            data = self.client.complete_json(
                "You are a MENET domain agent. Return JSON only. Allowed intents: inspect_data, train_model, predict_trait, evaluate_model, explain_model, generate_report. Extract only facts present in the user request.",
                message,
            )
            intent = TaskIntent(data["intent"]) if data.get("intent") else None
            raw_arguments = data.get("arguments") or {}
            allowed = {"trait", "device", "split_strategy", "explain_snp", "epochs"}
            arguments = {key: value for key, value in raw_arguments.items() if key in allowed}
            missing = data.get("missing_fields") or ([] if arguments.get("trait") else ["trait"])
            confidence = min(1.0, max(0.0, float(data.get("confidence", 0.7))))
            return ParsedIntent(intent, arguments, missing, confidence)
        except Exception:
            return self.fallback.parse(message)


class ResultInterpreter:
    def __init__(self, client: Optional[CompatibleLLMClient] = None):
        self.client = client or CompatibleLLMClient()

    def explain(self, result: Dict[str, Any]) -> str:
        if self.client.enabled:
            try:
                return self.client.complete_json(
                    "Explain the MENET result in concise Chinese. Do not invent values. Return JSON with an answer string and caveats array.",
                    json.dumps(result, ensure_ascii=False),
                ).get("answer", "")
            except Exception:
                pass
        if result.get("status") == "cancelled":
            return "任务已取消，已完成的中间轮次不会作为正式模型发布。可以修改参数后重新提交。"
        if result.get("status") == "completed":
            steps = result.get("steps") or []
            validation = next((step.get("data", {}) for step in steps if step.get("name") == "validate_dataset"), {})
            metrics = next((step.get("data", {}) for step in reversed(steps) if step.get("data", {}).get("test_r2") is not None), {})
            if metrics:
                return (
                    f"分析完成。测试集 R² 为 {float(metrics['test_r2']):.3f}，"
                    f"测试损失为 {float(metrics.get('test_loss', 0)):.3f}。"
                    "R² 需要结合数据划分、重复训练和育种场景判断，不能直接视为因果证据。"
                )
            if validation:
                return (
                    f"数据检查完成：匹配 {validation.get('matched_sample_count', 0)} 个样本，"
                    f"包含 {validation.get('snp_count', 0)} 个 SNP。"
                )
            return "MENET 任务已完成，结果文件可在下方下载。"
        errors = result.get("errors") or []
        detail = "；".join(errors)
        suggestions = []
        if "缺少必要文件" in detail:
            suggestions.append("请重新选择或上传包含 genotype、phenotype 和所需 split 的数据集")
        if "样本 ID" in detail or "匹配" in detail:
            suggestions.append("请确认两个 CSV 第一列的样本 ID 完全一致")
        if "CUDA" in detail or "GPU" in detail:
            suggestions.append("可改用自动设备或 CPU 后重试")
        if "模型" in detail:
            suggestions.append("请先完成一次兼容的模型训练")
        suffix = "；建议：" + "；".join(dict.fromkeys(suggestions)) if suggestions else ""
        return "MENET 任务未完成：" + detail + suffix

"""Optional language-model integration.

The service remains usable without credentials. Ollama, OpenAI-compatible and
Anthropic-compatible endpoints are supported.
"""

import json
import gzip
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import request

from .intent import IntentParser, ParsedIntent
from .models import TaskIntent


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
COMPATIBLE_API_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; MENET-Agent/0.1)",
}


def load_llm_settings(path: Path) -> Dict[str, Any]:
    defaults = {
        "enabled": bool(os.getenv("MENET_LLM_BASE_URL") and os.getenv("MENET_LLM_MODEL")),
        "provider": os.getenv("MENET_LLM_PROVIDER", "ollama"),
        "base_url": os.getenv("MENET_LLM_BASE_URL", DEFAULT_OLLAMA_URL),
        "api_key": os.getenv("MENET_LLM_API_KEY", ""),
        "model": os.getenv("MENET_LLM_MODEL", ""),
        "intent_model": os.getenv("MENET_LLM_INTENT_MODEL", os.getenv("MENET_LLM_MODEL", "")),
        "analysis_model": os.getenv("MENET_LLM_ANALYSIS_MODEL", os.getenv("MENET_LLM_MODEL", "")),
    }
    if not path.is_file():
        return defaults
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults
    settings = {**defaults, **{key: saved[key] for key in defaults if key in saved}}
    settings["intent_model"] = settings.get("intent_model") or settings.get("model", "")
    settings["analysis_model"] = settings.get("analysis_model") or settings.get("model", "")
    return settings


def save_llm_settings(path: Path, settings: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


class CompatibleLLMClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 120,
        provider: Optional[str] = None,
        enabled: Optional[bool] = None,
    ):
        self.base_url = (base_url or os.getenv("MENET_LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("MENET_LLM_API_KEY", "")
        self.model = model or os.getenv("MENET_LLM_MODEL", "")
        self.timeout = timeout
        self.provider = (provider or os.getenv("MENET_LLM_PROVIDER", "openai_compatible")).lower()
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        configured = bool(self.base_url and self.model)
        return configured if self._enabled is None else bool(self._enabled and configured)

    @property
    def protocol_provider(self) -> str:
        if self.provider != "compatible_auto":
            return self.provider
        return "anthropic_compatible" if self.model.lower().startswith("claude") else "openai_compatible"

    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("LLM 未配置")
        if self.protocol_provider == "ollama":
            return self._complete_ollama(system_prompt, user_prompt)
        if self.protocol_provider == "anthropic_compatible":
            return self._complete_anthropic(system_prompt, user_prompt)
        payload = {
            "model": self.model,
            "temperature": 0,
            "stream": False,
            "max_tokens": 2048,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {**COMPATIBLE_API_HEADERS, "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        endpoint = self._versioned_endpoint("chat/completions")
        req = request.Request(endpoint, data=body, headers=headers, method="POST")
        with request.urlopen(req, timeout=self.timeout) as response:
            result = self._read_json_response(response)
        content = result["choices"][0]["message"]["content"]
        return self._parse_json(content)

    def _complete_anthropic(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "max_tokens": 2048,
            "temperature": 0,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        headers = {
            **COMPATIBLE_API_HEADERS,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(
            self._versioned_endpoint("messages"),
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as response:
            result = self._read_json_response(response)
        content = "".join(
            block.get("text", "")
            for block in result.get("content", [])
            if block.get("type") == "text"
        )
        return self._parse_json(content)

    def _complete_ollama(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        req = request.Request(
            self.base_url + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as response:
            result = self._read_json_response(response)
        return self._parse_json(result["message"]["content"])

    def list_models(self) -> list[str]:
        if not self.base_url:
            return []
        if self.protocol_provider == "ollama":
            endpoint = self.base_url + "/api/tags"
            with request.urlopen(endpoint, timeout=min(self.timeout, 10)) as response:
                result = self._read_json_response(response)
            return [item["name"] for item in result.get("models", []) if item.get("name")]
        endpoint = self._versioned_endpoint("models")
        headers = dict(COMPATIBLE_API_HEADERS)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            if self.protocol_provider == "anthropic_compatible" or self.provider == "compatible_auto":
                headers["x-api-key"] = self.api_key
                headers["anthropic-version"] = "2023-06-01"
        with request.urlopen(request.Request(endpoint, headers=headers), timeout=min(self.timeout, 10)) as response:
            result = self._read_json_response(response)
        return [item["id"] for item in result.get("data", []) if item.get("id")]

    def _versioned_endpoint(self, resource: str) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith(f"/v1/{resource}"):
            return base
        if base.endswith("/v1"):
            return f"{base}/{resource}"
        return f"{base}/v1/{resource}"

    @staticmethod
    def _read_json_response(response) -> Dict[str, Any]:
        body = response.read()
        headers = getattr(response, "headers", None)
        encoding = headers.get("Content-Encoding", "").lower() if headers else ""
        if encoding == "gzip" or body.startswith(b"\x1f\x8b"):
            body = gzip.decompress(body)
        return json.loads(body.decode("utf-8"))

    def test_connection(self) -> Dict[str, Any]:
        started = time.monotonic()
        result = self.complete_json(
            'Return JSON only in this exact shape: {"ok": true}.',
            "Check the connection.",
        )
        return {"ok": result.get("ok") is True, "latency_seconds": round(time.monotonic() - started, 2)}

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
                'You are a MENET domain agent. Return JSON only with keys "intent", "arguments", '
                '"missing_fields", and "confidence". Allowed intents: inspect_data, train_model, '
                "predict_trait, evaluate_model, explain_model, generate_report. Arguments may contain "
                "trait, device, split_strategy, explain_snp, and epochs. Extract only facts present in "
                "the user request. Only trait may be required for these intents; device, split_strategy, "
                "epochs, and explain_snp are optional and must never appear in missing_fields. "
                "Use null for an unknown intent.",
                message,
            )
            intent = TaskIntent(data["intent"]) if data.get("intent") else None
            raw_arguments = data.get("arguments") or {}
            allowed = {"trait", "device", "split_strategy", "explain_snp", "epochs"}
            arguments = {key: value for key, value in raw_arguments.items() if key in allowed}
            required_fields = {"trait"}
            reported_missing = data.get("missing_fields") or []
            missing = [field for field in reported_missing if field in required_fields]
            if not arguments.get("trait") and "trait" not in missing:
                missing.append("trait")
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

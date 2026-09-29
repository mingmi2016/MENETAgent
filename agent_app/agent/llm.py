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
        self.last_source = {"type": "rules", "label": "规则引擎"}

    def parse(self, message: str) -> ParsedIntent:
        if not self.client.enabled:
            return self.fallback.parse(message)
        try:
            data = self.client.complete_json(
                'You are a MENET domain agent. Return JSON only with keys "intent", "arguments", '
                '"missing_fields", and "confidence". Allowed intents: list_models, inspect_data, train_model, '
                "predict_trait, evaluate_model, explain_model, generate_report. Arguments may contain "
                "trait, dataset_reference, model_id, device, split_strategy, explain_snp, and epochs. "
                "dataset_reference is the dataset name or species explicitly mentioned by the user; "
                "model_id is a model_xxx identifier explicitly mentioned by the user. list_models means the user asks "
                "which trained models are available for the current dataset or prediction. Extract only facts present in "
                "the user request. Only trait may be required for analysis intents; device, split_strategy, "
                "epochs, and explain_snp are optional and must never appear in missing_fields. "
                "Use null for an unknown intent.",
                message,
            )
            self.last_source = {"type": "llm", "label": f"模型服务 · {self.client.model}"}
            intent = TaskIntent(data["intent"]) if data.get("intent") else None
            raw_arguments = data.get("arguments") or {}
            allowed = {"trait", "dataset_reference", "model_id", "device", "split_strategy", "explain_snp", "epochs"}
            arguments = {key: value for key, value in raw_arguments.items() if key in allowed}
            required_fields = set() if intent == TaskIntent.LIST_MODELS else {"trait"}
            reported_missing = data.get("missing_fields") or []
            missing = [field for field in reported_missing if field in required_fields]
            if intent != TaskIntent.LIST_MODELS and not arguments.get("trait") and "trait" not in missing:
                missing.append("trait")
            confidence = min(1.0, max(0.0, float(data.get("confidence", 0.7))))
            return ParsedIntent(intent, arguments, missing, confidence)
        except Exception:
            self.last_source = {"type": "rules", "label": "规则引擎"}
            return self.fallback.parse(message)


class ResultInterpreter:
    def __init__(self, client: Optional[CompatibleLLMClient] = None):
        self.client = client or CompatibleLLMClient()

    def explain(self, result: Dict[str, Any]) -> str:
        return self.explain_with_source(result)[0]

    def recommend_prediction_model(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Ask the LLM for a model proposal, then validate its contract locally.

        The LLM may weigh the evidence and provide a rationale, but it cannot
        invent methods or metrics. Execution remains outside this method.
        """
        fallback = self._fallback_model_recommendation(evidence)
        if not self.client.enabled:
            fallback["recommendation_source"] = "规则引擎（模型服务不可用时的降级）"
            return fallback
        system_prompt = (
            "你是 MENET 模型选择助手。根据输入的真实结构化评估证据提出预测模型建议。"
            "必须只返回 JSON："
            '{"recommended_method":"menet|genomic_ridge|random_forest|xgboost",'
            '"reference_method":"menet", "confidence":"high|moderate|low",'
            '"reason":"中文理由", "requires_repeat_validation":true}。'
            "只能选择 metrics 中存在且有 test_r2 的方法；不能修改或编造任何数值。"
            "MENET 可以作为主要模型，也可以作为对照。若只有一次数据划分，必须将 requires_repeat_validation 设为 true。"
        )
        try:
            proposal = self.client.complete_json(system_prompt, json.dumps(evidence, ensure_ascii=False))
            method = proposal.get("recommended_method")
            metrics = evidence.get("metrics") or {}
            if method not in metrics or not isinstance(metrics[method], dict) or "test_r2" not in metrics[method]:
                raise ValueError("LLM 推荐了不存在或没有评估指标的模型")
            if proposal.get("reference_method", "menet") != "menet":
                raise ValueError("MENET 必须作为独立参考模型")
            confidence = proposal.get("confidence")
            if confidence not in {"high", "moderate", "low"}:
                raise ValueError("LLM 推荐置信度无效")
            reason = proposal.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("LLM 推荐缺少理由")
            if not isinstance(proposal.get("requires_repeat_validation"), bool):
                raise ValueError("LLM 推荐缺少重复验证标记")
            return {
                "primary_method": method,
                "reference_method": "menet",
                "confidence": confidence,
                "reason": reason.strip(),
                "requires_repeat_validation": proposal["requires_repeat_validation"],
                "recommendation_source": f"模型服务 · {self.client.model}",
                "candidates": {key: value.get("test_r2") for key, value in metrics.items()},
            }
        except Exception:
            fallback["recommendation_source"] = "规则引擎（模型服务推荐无效时的降级）"
            return fallback

    @staticmethod
    def _fallback_model_recommendation(evidence: Dict[str, Any]) -> Dict[str, Any]:
        metrics = evidence.get("metrics") or {}
        available = {
            method: float(item["test_r2"])
            for method, item in metrics.items()
            if isinstance(item, dict) and isinstance(item.get("test_r2"), (int, float))
        }
        menet_r2 = available.get("menet")
        best_method = max(available, key=available.get) if available else "menet"
        if menet_r2 is None:
            best_method = "menet"
        return {
            "primary_method": best_method,
            "reference_method": "menet",
            "confidence": "low" if not evidence.get("same_test_split", False) else "moderate",
            "reason": "模型服务不可用或推荐无效，使用结构化评估结果生成候选建议。",
            "requires_repeat_validation": not bool(evidence.get("repeated_validation", False)),
            "candidates": available,
        }

    def explain_with_source(self, result: Dict[str, Any]) -> tuple[str, Dict[str, str]]:
        if result.get("task_intent") == "inspect_data":
            if self.client.enabled:
                try:
                    answer = self._inspect_with_harness(result)
                    return answer, {
                        "type": "llm",
                        "label": f"模型服务 · {self.client.model}",
                    }
                except Exception:
                    pass
            return self._explain_inspection(result), {
                "type": "rules",
                "label": "结构化数据检查（模型服务不可用时的降级）",
            }
        if self.client.enabled:
            try:
                answer = self.client.complete_json(
                    "Explain the MENET result in concise Chinese. Respect task_intent. Do not invent values. Only discuss R² or baselines when they are present in the supplied result and the task is model training, evaluation, prediction, or report generation. Return JSON with an answer string and caveats array.",
                    json.dumps(result, ensure_ascii=False),
                ).get("answer", "")
                if answer:
                    return answer, {
                        "type": "llm",
                        "label": f"模型服务 · {self.client.model}",
                    }
            except Exception:
                pass
        return self._explain_rules(result), {
            "type": "rules",
            "label": "规则引擎",
        }

    def _inspect_with_harness(self, result: Dict[str, Any]) -> str:
        """Ask the LLM to explain validated evidence under a strict contract."""
        steps = result.get("steps") or []
        step = next((item for item in steps if item.get("name") == "validate_dataset"), {})
        data = step.get("data", {})
        evidence = {
            "task_intent": "inspect_data",
            "trait": result.get("task", {}).get("trait") or data.get("trait"),
            "validation_status": step.get("status") or step.get("step"),
            "errors": step.get("errors") or result.get("errors") or [],
            "warnings": step.get("warnings") or [],
            "measurements": {
                key: data.get(key)
                for key in (
                    "sample_count_genotype", "sample_count_phenotype", "matched_sample_count",
                    "snp_count", "genotype_missing_count", "genotype_missing_rate",
                    "phenotype_missing_count", "duplicate_marker_count",
                    "duplicate_genotype_sample_count",
                )
                if key in data
            },
        }
        system_prompt = (
            "你是 MENET 数据质量分析助手。当前任务类型固定为 inspect_data（数据检查），"
            "只能解释输入证据，不能进行模型性能评价。必须返回 JSON，格式为："
            '{"answer":"中文简洁说明","status":"pass|warning|failed",'
            '"findings":[{"severity":"info|warning|error","text":"..."}],'
            '"recommendation":"..."}。'
            "answer 必须说明样本匹配、SNP 数量、缺失和重复情况，并给出是否适合进入训练的建议。"
            "严禁提及 R²、测试损失、基线模型、模型排名或捏造输入中没有的数字；"
            "没有证据的内容必须写为未知。"
        )
        raw = self.client.complete_json(system_prompt, json.dumps(evidence, ensure_ascii=False))
        answer = raw.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("数据检查解释缺少 answer")
        if raw.get("status") not in {"pass", "warning", "failed"}:
            raise ValueError("数据检查解释 status 不符合约定")
        findings = raw.get("findings")
        if not isinstance(findings, list) or any(
            not isinstance(item, dict) or item.get("severity") not in {"info", "warning", "error"}
            or not isinstance(item.get("text"), str)
            for item in findings
        ):
            raise ValueError("数据检查解释 findings 不符合约定")
        if any(term in answer for term in ("R²", "R2", "测试损失", "基线模型", "XGBoost", "随机森林", "genomic_ridge")):
            raise ValueError("数据检查解释越过证据边界")
        return answer.strip()

    @staticmethod
    def _explain_inspection(result: Dict[str, Any]) -> str:
        steps = result.get("steps") or []
        step = next((item for item in steps if item.get("name") == "validate_dataset"), {})
        data = step.get("data", {})
        errors = step.get("errors") or result.get("errors") or []
        warnings = step.get("warnings") or []
        summary = (
            f"数据检查完成：基因型 {data.get('sample_count_genotype', 0)} 个样本，"
            f"表型 {data.get('sample_count_phenotype', 0)} 个样本，"
            f"成功匹配 {data.get('matched_sample_count', 0)} 个样本，"
            f"包含 {data.get('snp_count', 0)} 个 SNP；"
            f"基因型缺失率 {float(data.get('genotype_missing_rate', 0)) * 100:.2f}%，"
            f"表型缺失 {data.get('phenotype_missing_count', 0)} 个。"
        )
        duplicate_count = data.get("duplicate_genotype_sample_count", 0)
        if duplicate_count:
            warnings.append(f"检测到 {duplicate_count} 个基因型完全相同的样本")
        if errors:
            return summary + "数据存在问题，暂不建议直接训练：" + "；".join(errors) + "。"
        if warnings:
            return summary + "数据整体可以用于后续分析，但需要注意：" + "；".join(dict.fromkeys(warnings)) + "。"
        return summary + "数据完整性通过，可以进入 MENET 训练流程。"

    def _explain_rules(self, result: Dict[str, Any]) -> str:
        if result.get("status") == "cancelled":
            return "任务已取消，已完成的中间轮次不会作为正式模型发布。可以修改参数后重新提交。"
        if result.get("status") == "completed":
            steps = result.get("steps") or []
            validation = next((step.get("data", {}) for step in steps if step.get("name") == "validate_dataset"), {})
            metrics = next((step.get("data", {}) for step in reversed(steps) if step.get("data", {}).get("test_r2") is not None), {})
            if metrics:
                baseline_data = metrics.get("baselines") or ({"genomic_ridge": metrics.get("baseline")} if metrics.get("baseline") else {})
                labels = {"genomic_ridge": "基因组岭回归", "random_forest": "随机森林", "xgboost": "XGBoost"}
                comparisons = []
                for method, baseline in baseline_data.items():
                    if isinstance(baseline, dict) and "test_r2" in baseline:
                        r2 = float(baseline["test_r2"])
                        comparisons.append(f"{labels.get(method, method)} R² 为 {r2:.3f}，MENET 差值 {float(metrics['test_r2']) - r2:+.3f}")
                    elif isinstance(baseline, dict) and baseline.get("status") == "unavailable":
                        comparisons.append(f"{labels.get(method, method)} 当前不可用")
                comparison_text = "；".join(comparisons)
                available = [(method, float(item["test_r2"])) for method, item in baseline_data.items() if isinstance(item, dict) and "test_r2" in item]
                best_method, best_r2 = max(available, key=lambda item: item[1]) if available else (None, None)
                menet_r2 = float(metrics["test_r2"])
                if best_method and best_r2 > menet_r2 + 0.05:
                    recommendation = f"当前划分下建议优先使用{labels.get(best_method, best_method)}进行预测；MENET 可保留作对照，正式选择前建议重复验证。"
                elif best_method:
                    recommendation = "当前各模型差距不大，暂不做强制推荐，建议通过重复划分或交叉验证后再确定默认模型。"
                else:
                    recommendation = "暂时没有足够的基线结果，无法给出模型推荐。"
                return (f"分析完成。MENET 测试集 R² 为 {menet_r2:.3f}，测试损失为 {float(metrics.get('test_loss', 0)):.3f}。"
                        + (f"基线比较：{comparison_text}。" if comparison_text else "")
                        + f"模型建议：{recommendation}R² 需要结合数据划分、重复训练和育种场景判断，不能直接视为因果证据。")
            if validation:
                return f"数据检查完成：匹配 {validation.get('matched_sample_count', 0)} 个样本，包含 {validation.get('snp_count', 0)} 个 SNP。"
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

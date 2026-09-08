"""Deterministic data and model quality gates for MENET runs."""

import json
from pathlib import Path
from typing import Any, Dict

from .models import MenetTask


def build_quality_report(task: MenetTask, result: Dict[str, Any]) -> Dict[str, Any]:
    checks = []
    validation = next(
        (step.get("data", {}) for step in result.get("steps", []) if step.get("name") == "validate_dataset"),
        {},
    )
    errors = result.get("errors", [])
    warnings = result.get("warnings", [])
    if errors:
        checks.append(_check("workflow", "fail", "工作流未通过", "；".join(errors), "修正错误后重新运行。"))
    elif warnings:
        checks.append(_check("data_warnings", "warning", "数据检查存在提醒", "；".join(warnings), "确认提醒不会影响当前研究设计。"))
    else:
        checks.append(_check("data_validation", "pass", "数据格式检查通过", "未发现阻断性格式问题。", "继续保留原始数据和检查记录。"))

    matched = validation.get("matched_sample_count")
    if matched is not None:
        level = "warning" if matched < 100 else "pass"
        checks.append(_check("sample_size", level, "有效样本规模", f"匹配样本数为 {matched}。", "样本少于 100 时谨慎解释泛化能力。"))

    metrics_path = Path(task.output_dir) / "metrics.json"
    history_path = Path(task.output_dir) / "training_history.json"
    metrics = _read_json(metrics_path)
    history = _read_json(history_path, default=[])
    if metrics:
        test_r2 = float(metrics.get("test_r2", 0))
        level = "pass" if test_r2 >= 0.3 else "warning" if test_r2 >= 0 else "fail"
        checks.append(_check("test_r2", level, "测试集泛化表现", f"测试集 R² 为 {test_r2:.3f}。", "结合基线、重复划分和育种目标判断是否可用。"))
        baseline = metrics.get("baseline") or {}
        if "test_r2" in baseline:
            baseline_r2 = float(baseline["test_r2"])
            gain = test_r2 - baseline_r2
            level = "pass" if gain > 0 else "warning"
            checks.append(_check(
                "baseline_comparison", level, "与岭回归基线比较",
                f"MENET 相对基线的测试集 R² 差值为 {gain:.3f}（基线 {baseline_r2:.3f}）。",
                "若 MENET 未超过基线，应优先检查数据划分、超参数和样本规模。",
            ))
    if history:
        best = max(history, key=lambda item: float(item.get("val_r2", float("-inf"))))
        gap = float(best.get("train_r2", 0)) - float(best.get("val_r2", 0))
        level = "warning" if gap > 0.2 else "pass"
        checks.append(_check("overfitting_gap", level, "训练与验证差距", f"最佳验证轮次的 R² 差距为 {gap:.3f}。", "差距超过 0.2 时考虑早停、正则化或增加样本。"))

    overall = "fail" if any(item["level"] == "fail" for item in checks) else "warning" if any(item["level"] == "warning" for item in checks) else "pass"
    return {
        "overall_status": overall,
        "checks": checks,
        "limitations": [
            "单次数据划分不能证明模型稳定性，需要重复训练或交叉验证。",
            "预测表现和 SNP 梯度重要性不能直接解释为因果效应。",
        ],
    }


def write_quality_report(task: MenetTask, result: Dict[str, Any]) -> Dict[str, Any]:
    report = build_quality_report(task, result)
    output = Path(task.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _check(check_id: str, level: str, title: str, detail: str, action: str) -> Dict[str, str]:
    return {"id": check_id, "level": level, "title": title, "detail": detail, "action": action}


def _read_json(path: Path, default=None):
    if not path.is_file():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {} if default is None else default

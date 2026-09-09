"""HTTP entry point for the MENET Agent MVP."""

import os
import json
import hashlib
import re
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .models import MenetTask, TaskIntent
from .llm import (
    AgentIntentParser,
    CompatibleLLMClient,
    ResultInterpreter,
    load_llm_settings,
    save_llm_settings,
)
from .store import DEFAULT_USER_ID, TaskStore
from .runner import run_task
from .demo_data import register_demo_datasets
from .quality import build_quality_report
from .tools import MenetTools


class TaskRequest(BaseModel):
    trait: str = Field(min_length=1)
    user_id: str = DEFAULT_USER_ID
    dataset_dir: str = "data"
    output_dir: str = "runs"
    source_output_dir: Optional[str] = None
    device: str = "auto"
    split_strategy: str = "existing"
    train_ratio: float = 0.7
    valid_ratio: float = 0.15
    test_ratio: float = 0.15
    epochs: Optional[int] = None
    explain_snp: bool = False
    intent: TaskIntent = TaskIntent.TRAIN_MODEL
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    user_id: str = DEFAULT_USER_ID
    dataset_id: Optional[str] = None
    dataset_dir: Optional[str] = None
    output_dir: Optional[str] = None
    conversation_id: Optional[str] = None
    prediction_genotype_path: Optional[str] = None
    model_id: Optional[str] = None


class UserRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=80)


class ConversationRequest(BaseModel):
    user_id: str = DEFAULT_USER_ID


class ModelUpdateRequest(BaseModel):
    user_id: str = DEFAULT_USER_ID
    name: str = Field(min_length=1, max_length=120)


class RepeatTrainingRequest(BaseModel):
    user_id: str = DEFAULT_USER_ID
    conversation_id: Optional[str] = None
    repeats: int = Field(default=3, ge=2, le=5)
    epochs: Optional[int] = Field(default=None, ge=1)
    device: str = "auto"


class LLMSettingsRequest(BaseModel):
    enabled: bool = True
    provider: str = "ollama"
    base_url: str = Field(min_length=1)
    model: str = ""
    intent_model: str = ""
    analysis_model: str = ""
    api_key: Optional[str] = None


app = FastAPI(title="MENET Agent", version="0.1.0")
service_root = Path(
    os.environ.get("MENET_AGENT_ROOT", Path(__file__).resolve().parents[1])
).resolve()
database_path = Path(os.environ.get("MENET_AGENT_DB", "runs/agent.db"))
if not database_path.is_absolute():
    database_path = service_root / database_path
store = TaskStore(str(database_path))
legacy_root = service_root.parent / "MENET"
store.rebase_paths({
    str(legacy_root / "runs"): str(service_root / "runs"),
    str(legacy_root / "uploads"): str(service_root / "uploads"),
    str(legacy_root / "data" / "demo"): str(service_root / "data" / "demo"),
})
register_demo_datasets(store, service_root)
store.backfill_models()
store.recover_interrupted()
executor = ThreadPoolExecutor(max_workers=int(os.environ.get("MENET_AGENT_WORKERS", "1")))
llm_settings_path = service_root / "runs" / "llm-settings.json"
llm_settings = load_llm_settings(llm_settings_path)


def _llm_client(settings: Dict[str, Any], role: str = "intent") -> CompatibleLLMClient:
    role_model = settings.get(f"{role}_model") or settings.get("model")
    return CompatibleLLMClient(
        base_url=settings.get("base_url"),
        api_key=settings.get("api_key"),
        model=role_model,
        provider=settings.get("provider"),
        enabled=settings.get("enabled", False),
    )


intent_parser = AgentIntentParser(client=_llm_client(llm_settings, "intent"))
result_interpreter = ResultInterpreter(client=_llm_client(llm_settings, "analysis"))

web_root = service_root / "web"
if web_root.is_dir():
    app.mount("/app", StaticFiles(directory=web_root, html=True), name="web")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "menet-agent", "queue": os.environ.get("MENET_QUEUE_BACKEND", "thread")}


@app.get("/", include_in_schema=False)
def web_app() -> RedirectResponse:
    return RedirectResponse(url="/app/")


def _mask_api_key(api_key: str) -> str:
    if not api_key:
        return ""
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return f"{api_key[:3]}{'*' * min(16, len(api_key) - 7)}{api_key[-4:]}"


@app.get("/api/v1/settings/llm")
def get_llm_settings() -> Dict[str, Any]:
    return {
        "enabled": bool(llm_settings.get("enabled")),
        "provider": llm_settings.get("provider", "ollama"),
        "base_url": llm_settings.get("base_url", ""),
        "model": llm_settings.get("model", ""),
        "intent_model": llm_settings.get("intent_model") or llm_settings.get("model", ""),
        "analysis_model": llm_settings.get("analysis_model") or llm_settings.get("model", ""),
        "api_key_configured": bool(llm_settings.get("api_key")),
        "api_key_masked": _mask_api_key(llm_settings.get("api_key", "")),
        "mode": "llm" if intent_parser.client.enabled else "rules",
    }


@app.put("/api/v1/settings/llm")
def update_llm_settings(request_data: LLMSettingsRequest) -> Dict[str, Any]:
    global llm_settings
    provider = request_data.provider.strip().lower()
    if provider not in {"ollama", "openai_compatible", "anthropic_compatible", "compatible_auto"}:
        raise HTTPException(status_code=422, detail="不支持的模型服务类型")
    base_url = request_data.base_url.strip().rstrip("/")
    if not re.match(r"^https?://", base_url):
        raise HTTPException(status_code=422, detail="模型服务地址必须以 http:// 或 https:// 开头")
    api_key = request_data.api_key
    if api_key is None:
        api_key = llm_settings.get("api_key", "")
    intent_model = request_data.intent_model.strip() or request_data.model.strip()
    analysis_model = request_data.analysis_model.strip() or intent_model
    llm_settings = {
        "enabled": request_data.enabled,
        "provider": provider,
        "base_url": base_url,
        "model": intent_model,
        "intent_model": intent_model,
        "analysis_model": analysis_model,
        "api_key": api_key.strip(),
    }
    save_llm_settings(llm_settings_path, llm_settings)
    intent_parser.client = _llm_client(llm_settings, "intent")
    result_interpreter.client = _llm_client(llm_settings, "analysis")
    return get_llm_settings()


@app.get("/api/v1/settings/llm/models")
def list_llm_models(provider: str = "ollama", base_url: str = "http://127.0.0.1:11434") -> Dict[str, Any]:
    try:
        client = CompatibleLLMClient(
            provider=provider,
            base_url=base_url.rstrip("/"),
            api_key=llm_settings.get("api_key", ""),
            model=llm_settings.get("model", ""),
            enabled=True,
        )
        return {"models": client.list_models()}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"无法读取模型列表: {exc}") from exc


@app.post("/api/v1/settings/llm/test")
def test_llm_connection() -> Dict[str, Any]:
    try:
        tested = []
        for role in ("intent", "analysis"):
            client = _llm_client({**llm_settings, "enabled": True}, role)
            if any(item["model"] == client.model for item in tested):
                continue
            result = client.test_connection()
            if not result["ok"]:
                raise HTTPException(status_code=502, detail=f"{client.model} 未通过 JSON 测试")
            tested.append({"role": role, "model": client.model, **result})
        return {"ok": True, "provider": llm_settings["provider"], "models": tested}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"模型连接失败: {exc}") from exc


@app.get("/api/v1/users")
def list_users() -> Dict[str, Any]:
    return {"users": store.list_users()}


@app.post("/api/v1/users", status_code=201)
def create_user(request: UserRequest) -> Dict[str, Any]:
    try:
        return store.create_user(request.display_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/conversations", status_code=201)
def create_conversation(request: ConversationRequest = ConversationRequest()) -> Dict[str, Any]:
    try:
        return store.ensure_conversation(user_id=request.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/conversations")
def list_conversations(user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
    _require_user(user_id)
    return {"conversations": store.list_conversations(user_id)}


@app.get("/api/v1/conversations/{conversation_id}")
def get_conversation(conversation_id: str, user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
    conversation = store.get_conversation(conversation_id, user_id=user_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return conversation


@app.get("/api/v1/datasets")
def list_datasets(user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
    _require_user(user_id)
    return {"datasets": store.list_datasets(user_id)}


@app.get("/api/v1/datasets/{dataset_id}")
def get_dataset(dataset_id: str, user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
    dataset = store.get_dataset(dataset_id, user_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="数据集不存在")
    return dataset


@app.get("/api/v1/datasets/{dataset_id}/inspection")
def inspect_registered_dataset(dataset_id: str, user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
    dataset = store.get_dataset(dataset_id, user_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="数据集不存在")
    task = MenetTask(
        owner_user_id=user_id,
        dataset_dir=dataset["dataset_dir"],
        trait=dataset["trait"],
        intent=TaskIntent.INSPECT_DATA,
    )
    inspection = MenetTools().validate_dataset(task)
    data = inspection.get("data", {})
    if data.get("matched_sample_count") is not None and data.get("snp_count") is not None:
        store.update_dataset_stats(dataset["dataset_dir"], data["matched_sample_count"], data["snp_count"])
    return {"dataset": dataset, "inspection": inspection}


@app.post("/api/v1/datasets/upload", status_code=201)
async def upload_dataset(
    trait: str,
    genotype: UploadFile = File(...),
    phenotype: UploadFile = File(...),
    conversation_id: Optional[str] = None,
    user_id: str = DEFAULT_USER_ID,
    name: str = "",
    species: str = "",
) -> Dict[str, Any]:
    _require_user(user_id)
    if not trait or Path(trait).name != trait or any(char in trait for char in "/\\"):
        raise HTTPException(status_code=400, detail="非法性状名称")
    allowed = {".csv"}
    if Path(genotype.filename or "").suffix.lower() not in allowed or Path(phenotype.filename or "").suffix.lower() not in allowed:
        raise HTTPException(status_code=400, detail="当前仅支持 CSV 文件")
    dataset_id = f"dataset_{uuid4().hex[:12]}"
    dataset_dir = service_root / "uploads" / user_id / dataset_id
    (dataset_dir / "genotype").mkdir(parents=True, exist_ok=False)
    (dataset_dir / "phenotype").mkdir(parents=True, exist_ok=False)
    genotype_path = dataset_dir / "genotype" / "genotype.csv"
    phenotype_path = dataset_dir / "phenotype" / f"{trait}.csv"
    max_bytes = int(os.environ.get("MENET_MAX_UPLOAD_BYTES", str(512 * 1024 * 1024)))
    try:
        genotype_meta = await _save_upload(genotype, genotype_path, max_bytes)
        phenotype_meta = await _save_upload(phenotype, phenotype_path, max_bytes)
    except Exception:
        shutil.rmtree(dataset_dir, ignore_errors=True)
        raise
    dataset = {
        "dataset_id": dataset_id,
        "user_id": user_id,
        "name": name.strip() or f"{species.strip() or 'MENET'} {trait}",
        "species": species.strip(),
        "trait": trait,
        "dataset_dir": str(dataset_dir),
        "genotype_filename": genotype.filename or "genotype.csv",
        "phenotype_filename": phenotype.filename or f"{trait}.csv",
        "genotype_size": genotype_meta["size"],
        "phenotype_size": phenotype_meta["size"],
        "genotype_sha256": genotype_meta["sha256"],
        "phenotype_sha256": phenotype_meta["sha256"],
    }
    store.create_dataset(dataset)
    if conversation_id:
        current = store.get_conversation(conversation_id, user_id=user_id)
        if current is None:
            raise HTTPException(status_code=404, detail="会话不存在或不属于当前用户")
        recent = [dataset_id] + [item for item in current.get("state", {}).get("recent_dataset_ids", []) if item != dataset_id]
        store.update_conversation(conversation_id, {
            "active_dataset_id": dataset_id,
            "active_dataset_dir": str(dataset_dir),
            "active_trait": trait,
            "recent_dataset_ids": recent[:10],
        }, user_id)
    return dataset


@app.post("/api/v1/prediction-genotypes/upload", status_code=201)
async def upload_prediction_genotype(
    file: UploadFile = File(...),
    user_id: str = DEFAULT_USER_ID,
) -> Dict[str, Any]:
    """Store a genotype file for a one-off new-material prediction."""
    _require_user(user_id)
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="预测基因型文件必须是 CSV")
    destination_dir = service_root / "uploads" / user_id / "prediction_inputs"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{uuid4().hex[:12]}_genotype.csv"
    try:
        info = await _save_upload(file, destination, 2 * 1024 * 1024 * 1024)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"保存预测基因型失败: {exc}") from exc
    return {"path": str(destination), "filename": file.filename, **info}


@app.get("/api/v1/models")
def list_trained_models(user_id: str = DEFAULT_USER_ID, include_archived: bool = False) -> Dict[str, Any]:
    _require_user(user_id)
    return {"models": [_with_model_quality(model) for model in store.list_models(user_id, include_archived=include_archived)]}


@app.get("/api/v1/models/compare")
def compare_trained_models(
    user_id: str = DEFAULT_USER_ID,
    trait: str = "",
    dataset_id: str = "",
) -> Dict[str, Any]:
    _require_user(user_id)
    models = [_with_model_quality(model) for model in store.list_models(user_id)]
    if trait:
        models = [model for model in models if model["trait"] == trait]
    if dataset_id:
        models = [model for model in models if model.get("dataset_id") == dataset_id]
    comparable = [model for model in models if model.get("metrics", {}).get("test_r2") is not None]
    scores = [float(model["metrics"]["test_r2"]) for model in comparable]
    score_range = max(scores) - min(scores) if len(scores) >= 2 else None
    stability = (
        "insufficient" if score_range is None else
        "stable" if score_range <= 0.1 else
        "variable" if score_range <= 0.25 else
        "unstable"
    )
    summary = {
        "run_count": len(comparable),
        "stable_enough_to_estimate": len(scores) >= 2,
        "test_r2_mean": sum(scores) / len(scores) if scores else None,
        "test_r2_min": min(scores) if scores else None,
        "test_r2_max": max(scores) if scores else None,
        "test_r2_std": statistics.stdev(scores) if len(scores) >= 2 else None,
        "test_r2_range": score_range,
        "stability": stability,
    }
    return {"models": comparable, "summary": summary}


@app.get("/api/v1/models/{model_id}")
def get_trained_model(model_id: str, user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
    model = store.get_model(model_id, user_id)
    if model is None:
        raise HTTPException(status_code=404, detail="模型不存在")
    return _with_model_quality(model)


@app.patch("/api/v1/models/{model_id}")
def update_trained_model(model_id: str, request_data: ModelUpdateRequest) -> Dict[str, Any]:
    try:
        model = store.rename_model(model_id, request_data.user_id, request_data.name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if model is None:
        raise HTTPException(status_code=404, detail="模型不存在")
    return model


@app.delete("/api/v1/models/{model_id}")
def archive_trained_model(model_id: str, user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
    model = store.archive_model(model_id, user_id)
    if model is None:
        raise HTTPException(status_code=404, detail="模型不存在")
    return {"model_id": model_id, "status": "archived", "message": "模型已从可选列表移除，训练文件仍被保留。"}


@app.post("/api/v1/models/{model_id}/repeat-training", status_code=202)
def repeat_model_training(model_id: str, request_data: RepeatTrainingRequest) -> Dict[str, Any]:
    _require_user(request_data.user_id)
    model = store.get_model(model_id, request_data.user_id)
    if model is None or model.get("status") != "active":
        raise HTTPException(status_code=404, detail="模型不存在或已归档")
    source = store.get(model["task_id"], request_data.user_id)
    if source is None:
        raise HTTPException(status_code=404, detail="找不到模型的原始训练任务")
    source_task = source["task"]
    group_id = f"repeat_{uuid4().hex[:12]}"
    base_seed = int(uuid4().hex[:8], 16)
    tasks = []
    for index in range(request_data.repeats):
        task = MenetTask(
            owner_user_id=request_data.user_id,
            dataset_dir=model["dataset_dir"],
            output_dir="runs",
            trait=model["trait"],
            device=request_data.device,
            epochs=request_data.epochs or source_task.get("epochs"),
            split_strategy="random",
            train_ratio=float(source_task.get("train_ratio", 0.7)),
            valid_ratio=float(source_task.get("valid_ratio", 0.15)),
            test_ratio=float(source_task.get("test_ratio", 0.15)),
            intent=TaskIntent.TRAIN_MODEL,
            metadata={
                "dataset_id": model.get("dataset_id", ""),
                "conversation_id": request_data.conversation_id,
                "repeated_from_model_id": model_id,
                "repeat_group_id": group_id,
                "repeat_index": index + 1,
                "repeat_total": request_data.repeats,
                "split_seed": base_seed + index,
                "seed": base_seed + 1000 + index,
            },
        )
        _prepare_task_directories(task, "runs")
        store.create(task.to_dict())
        _submit_task(task)
        tasks.append({
            "task_id": task.task_id,
            "repeat_index": index + 1,
            "split_seed": task.metadata["split_seed"],
            "training_seed": task.metadata["seed"],
        })
    if request_data.conversation_id:
        store.update_conversation(
            request_data.conversation_id,
            {"active_task_id": tasks[-1]["task_id"], "active_trait": model["trait"],
             "active_dataset_dir": model["dataset_dir"], "active_model_id": model_id},
            request_data.user_id,
        )
    return {"repeat_group_id": group_id, "model_id": model_id, "tasks": tasks, "status": "queued"}


@app.post("/api/v1/chat")
def chat(request: ChatRequest) -> Dict[str, Any]:
    _require_user(request.user_id)
    try:
        conversation = store.ensure_conversation(request.conversation_id, request.user_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    conversation_id = conversation["conversation_id"]
    state = conversation.get("state", {})
    dataset = _resolve_dataset(request.message, request.user_id, request.dataset_id, state)
    dataset_dir = (dataset or {}).get("dataset_dir") or request.dataset_dir or state.get("active_dataset_dir") or "data"
    output_base_dir = request.output_dir or state.get("active_output_base_dir") or "runs"
    _validate_paths(dataset_dir, output_base_dir)
    store.add_message(conversation_id, "user", request.message)
    parsed = intent_parser.parse(request.message)
    requested_model = store.get_model(request.model_id, request.user_id) if request.model_id else None
    context_trait = (requested_model or {}).get("trait") or (dataset or {}).get("trait") or state.get("active_trait")
    if "trait" in parsed.missing_fields and context_trait:
        parsed.arguments["trait"] = context_trait
        parsed.missing_fields.remove("trait")
    if parsed.intent is None or parsed.missing_fields:
        reply = "请补充以下信息：" + "、".join(parsed.missing_fields)
        store.add_message(conversation_id, "assistant", reply)
        return {"type": "needs_input", "conversation_id": conversation_id, "parsed": parsed.to_dict()}
    source_run = None
    selected_model = None
    if parsed.intent in {TaskIntent.PREDICT_TRAIT, TaskIntent.EVALUATE_MODEL, TaskIntent.EXPLAIN_MODEL}:
        if request.model_id:
            selected_model = requested_model
            if selected_model is None or selected_model.get("status") != "active":
                raise HTTPException(status_code=404, detail="所选模型不存在或已归档")
            if selected_model["trait"] != parsed.arguments["trait"]:
                reply = f"所选模型用于 {selected_model['trait']}，不能预测 {parsed.arguments['trait']}。请更换模型或性状。"
                store.add_message(conversation_id, "assistant", reply)
                return {"type": "needs_model", "conversation_id": conversation_id, "message": reply, "parsed": parsed.to_dict()}
            source_run = store.get(selected_model["task_id"], request.user_id)
            dataset_dir = selected_model["dataset_dir"]
        else:
            source_run = _latest_compatible_run(request.user_id, dataset_dir, parsed.arguments["trait"])
            if source_run:
                selected_model = store.get_model(f"model_{source_run['task_id'].removeprefix('task_')}", request.user_id)
        if source_run is None:
            reply = "当前数据和性状还没有可复用的已完成模型。请先训练模型，再进行预测、评估或解释。"
            store.add_message(conversation_id, "assistant", reply)
            return {
                "type": "needs_model",
                "conversation_id": conversation_id,
                "message": reply,
                "parsed": parsed.to_dict(),
            }
    payload = {
        "intent": parsed.intent,
        "trait": parsed.arguments["trait"],
        "owner_user_id": request.user_id,
        "dataset_dir": dataset_dir,
        "output_dir": output_base_dir,
        **({"source_output_dir": source_run["task"]["output_dir"]} if source_run else {}),
        **parsed.arguments,
        "metadata": {
            "conversation_id": conversation_id,
            **({"dataset_id": dataset["dataset_id"]} if dataset else {}),
            **({"model_id": selected_model["model_id"], "model_name": selected_model["name"]} if selected_model else {}),
            **({"prediction_genotype_path": request.prediction_genotype_path} if request.prediction_genotype_path else {}),
        },
    }
    task = MenetTask(**payload)
    _prepare_task_directories(task, output_base_dir)
    errors = task.validate()
    if errors:
        store.add_message(conversation_id, "assistant", "任务参数无效：" + "；".join(errors))
        return {"type": "invalid_task", "conversation_id": conversation_id, "parsed": parsed.to_dict(), "errors": errors}
    store.create(task.to_dict())
    conversation_update = {
        "active_trait": task.trait,
        "active_dataset_dir": task.dataset_dir,
        "active_output_base_dir": output_base_dir,
        "active_task_id": task.task_id,
        "active_run_dir": task.output_dir,
        **({"active_model_id": selected_model["model_id"]} if selected_model else {}),
        **({"active_dataset_id": dataset["dataset_id"]} if dataset else {}),
    }
    if dataset:
        recent = [dataset["dataset_id"]] + [
            item for item in state.get("recent_dataset_ids", []) if item != dataset["dataset_id"]
        ]
        conversation_update["recent_dataset_ids"] = recent[:10]
    store.update_conversation(conversation_id, conversation_update, request.user_id)
    _submit_task(task)
    return {"type": "task_submitted", "conversation_id": conversation_id, "task_id": task.task_id, "output_dir": task.output_dir, "parsed": parsed.to_dict()}


@app.post("/api/v1/tasks", status_code=202)
def submit_task(request: TaskRequest) -> Dict[str, Any]:
    _require_user(request.user_id)
    _validate_paths(request.dataset_dir, request.output_dir)
    payload = request.model_dump()
    payload["owner_user_id"] = payload.pop("user_id")
    task = MenetTask(**payload)
    _prepare_task_directories(task, request.output_dir)
    errors = task.validate()
    if errors:
        raise HTTPException(status_code=422, detail=errors)
    store.create(task.to_dict())
    _submit_task(task)
    return {"task_id": task.task_id, "status": "queued"}


@app.get("/api/v1/tasks")
def list_tasks(user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
    _require_user(user_id)
    return {"tasks": [_with_runtime(task) for task in store.list_tasks(user_id)]}


@app.get("/api/v1/tasks/{task_id}")
def get_task(task_id: str, user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
    task = store.get(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return _with_runtime(task)


@app.post("/api/v1/tasks/{task_id}/cancel", status_code=202)
def cancel_task(task_id: str, user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
    task = store.get(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task["status"] in {"completed", "failed", "cancelled"}:
        raise HTTPException(status_code=409, detail="任务已经结束，无法取消")
    output = Path(task["task"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "cancel.requested").write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    return {"task_id": task_id, "status": "cancel_requested", "message": "已请求停止；当前轮次结束后生效。"}


@app.get("/api/v1/tasks/{task_id}/explanation")
def explain_task(task_id: str, user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
    task = store.get(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task["result"] is None:
        raise HTTPException(status_code=409, detail="任务尚未完成")
    return {"task_id": task_id, "answer": result_interpreter.explain(task["result"])}


@app.get("/api/v1/tasks/{task_id}/artifacts/{filename}")
def download_artifact(task_id: str, filename: str, user_id: str = DEFAULT_USER_ID) -> FileResponse:
    task = store.get(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="非法文件名")
    output_dir = Path(task["task"]["output_dir"]).resolve()
    artifact = (output_dir / filename).resolve()
    if output_dir not in artifact.parents or not artifact.is_file():
        raise HTTPException(status_code=404, detail="工件不存在")
    return FileResponse(artifact, filename=filename)


@app.get("/api/v1/tasks/{task_id}/artifacts")
def list_artifacts(task_id: str, user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
    task = store.get(task_id, user_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    output_dir = Path(task["task"]["output_dir"]).resolve()
    files = [
        {"name": path.name, "size": path.stat().st_size}
        for path in sorted(output_dir.iterdir()) if path.is_file() and path.name != "cancel.requested"
    ] if output_dir.is_dir() else []
    return {"task_id": task_id, "files": files}


def _submit_task(task: MenetTask) -> None:
    if os.environ.get("MENET_QUEUE_BACKEND", "thread").lower() == "celery":
        try:
            from .celery_app import run_task as celery_task
            celery_task.delay(task.to_dict())
            return
        except Exception as exc:
            store.update(task.task_id, "failed", {"task_id": task.task_id, "status": "failed", "errors": [f"Celery任务提交失败: {exc}"]})
            return
    executor.submit(run_task, task.to_dict(), str(store.path))


def _validate_paths(dataset_dir: str, output_dir: str) -> None:
    for label, value in (("dataset_dir", dataset_dir), ("output_dir", output_dir)):
        resolved = (service_root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
        if resolved != service_root and service_root not in resolved.parents:
            raise HTTPException(status_code=400, detail=f"{label} 必须位于 MENET_AGENT_ROOT 目录内")


def _prepare_task_directories(task: MenetTask, output_base_dir: str) -> None:
    base = (service_root / output_base_dir).resolve() if not Path(output_base_dir).is_absolute() else Path(output_base_dir).resolve()
    if task.intent in {TaskIntent.PREDICT_TRAIT, TaskIntent.EVALUATE_MODEL, TaskIntent.EXPLAIN_MODEL}:
        task.source_output_dir = task.source_output_dir or str(base)
        base = service_root / "runs"
    task.metadata.setdefault("output_base_dir", str(base))
    task.output_dir = str(base / task.owner_user_id / task.task_id)
    _validate_paths(task.dataset_dir, task.output_dir)


async def _save_upload(upload: UploadFile, destination: Path, max_bytes: int) -> Dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with destination.open("wb") as handle:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(status_code=413, detail="上传文件超过大小限制")
            digest.update(chunk)
            handle.write(chunk)
    return {"size": size, "sha256": digest.hexdigest()}


def _require_user(user_id: str) -> Dict[str, Any]:
    user = store.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    return user


def _latest_compatible_run(user_id: str, dataset_dir: str, trait: str) -> Optional[Dict[str, Any]]:
    target = Path(dataset_dir).resolve()
    for record in store.list_tasks(user_id, limit=200):
        task = record["task"]
        if record["status"] != "completed" or task.get("trait") != trait:
            continue
        if task.get("intent") not in {"train_model", "generate_report"}:
            continue
        if Path(task.get("dataset_dir", "")).resolve() != target:
            continue
        output = Path(task.get("output_dir", ""))
        if (output / "menet_model.pt").is_file():
            return record
    return None


def _with_runtime(record: Dict[str, Any]) -> Dict[str, Any]:
    """Add honest elapsed time and a locally learned duration estimate."""
    terminal = record["status"] in {"completed", "failed", "cancelled"}
    created = datetime.fromisoformat(record["created_at"])
    started = datetime.fromisoformat(record["started_at"]) if record.get("started_at") else None
    ended_value = record.get("finished_at") or (record["updated_at"] if terminal else None)
    ended = datetime.fromisoformat(ended_value) if ended_value else datetime.now(timezone.utc)
    elapsed = max(0, int((ended - created).total_seconds()))
    if started:
        queue_seconds = max(0, int((started - created).total_seconds()))
        execution_seconds = max(0, int((ended - started).total_seconds()))
    elif record["status"] == "created":
        queue_seconds, execution_seconds = elapsed, 0
    else:
        queue_seconds, execution_seconds = 0, elapsed
    task = record["task"]
    durations = []
    for previous in store.list_tasks(record["user_id"], limit=200):
        if previous["task_id"] == record["task_id"] or previous["status"] != "completed":
            continue
        candidate = previous["task"]
        comparable_fields = ("intent", "dataset_dir", "device", "epochs", "explain_snp")
        if any(candidate.get(field) != task.get(field) for field in comparable_fields):
            continue
        first = datetime.fromisoformat(previous.get("started_at") or previous["created_at"])
        last = datetime.fromisoformat(previous.get("finished_at") or previous["updated_at"])
        durations.append(max(1, int((last - first).total_seconds())))
    runtime: Dict[str, Any] = {
        "elapsed_seconds": elapsed,
        "queue_seconds": queue_seconds,
        "execution_seconds": execution_seconds,
        "terminal": terminal,
    }
    progress_path = Path(task.get("output_dir", "")) / "progress.json"
    if progress_path.is_file():
        try:
            runtime["epoch_progress"] = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    if terminal:
        runtime.update({"actual_seconds": elapsed, "message": "任务已结束，显示的是实际总耗时。"})
    elif durations:
        median = statistics.median(durations)
        runtime.update({
            "estimated_seconds_low": max(1, int(median * 0.7)),
            "estimated_seconds_high": max(2, int(median * 1.5)),
            "estimate_source": "local_history",
            "history_samples": len(durations),
            "message": "预计区间来自当前用户在本机运行的同数据、同类型历史任务。",
        })
    elif benchmark := _local_benchmark(task):
        seconds = benchmark["execution_seconds"]
        runtime.update({
            "estimated_seconds_low": max(1, int(seconds * 0.8)),
            "estimated_seconds_high": max(2, int(seconds * 1.3)),
            "estimate_source": "local_benchmark",
            "history_samples": 1,
            "message": "预计区间来自本机 RTX 5060 对该示例数据和配置的实测基准，不含排队时间。",
        })
    elif task.get("intent") in {"train_model", "generate_report"}:
        runtime.update({
            "estimate_source": "pending_benchmark",
            "history_samples": 0,
            "message": "首次同类训练暂无本机基准；可能需要数分钟到数小时，请保持服务和 Worker 运行。",
        })
    else:
        runtime.update({
            "estimate_source": "pending_benchmark",
            "history_samples": 0,
            "message": "首次同类任务暂无本机基准；完成后将自动积累实际耗时。",
        })
    return {**record, "runtime": runtime}


def _with_model_quality(model: Dict[str, Any]) -> Dict[str, Any]:
    record = store.get(model["task_id"], model["user_id"])
    if not record or not record.get("result"):
        return model
    path = Path(model["output_dir"]) / "quality_report.json"
    try:
        quality = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else build_quality_report(
            MenetTask(**record["task"]), record["result"]
        )
    except (OSError, ValueError, TypeError):
        return model
    return {**model, "quality_report": quality}


def _local_benchmark(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    path = service_root / "data" / "benchmarks" / "local_gpu.json"
    if not path.is_file():
        return None
    try:
        benchmarks = json.loads(path.read_text(encoding="utf-8")).get("benchmarks", [])
        dataset_dir = Path(task.get("dataset_dir", ""))
        dataset_dir = dataset_dir if dataset_dir.is_absolute() else service_root / dataset_dir
        for benchmark in benchmarks:
            benchmark_dir = (service_root / benchmark["dataset_dir"]).resolve()
            device_matches = task.get("device") == benchmark["device"] or task.get("device") == "auto"
            if (
                dataset_dir.resolve() == benchmark_dir
                and task.get("intent") == benchmark["intent"]
                and task.get("epochs") == benchmark["epochs"]
                and bool(task.get("explain_snp")) == benchmark["explain_snp"]
                and device_matches
            ):
                return benchmark
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return None


def _resolve_dataset(message: str, user_id: str, requested_id: Optional[str], state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    datasets = store.list_datasets(user_id)
    by_id = {item["dataset_id"]: item for item in datasets}
    text = message.lower()
    recent = state.get("recent_dataset_ids", [])
    if any(word in text for word in ("上一个数据", "前一个数据", "之前的数据")) and len(recent) > 1:
        return by_id.get(recent[1])
    for item in datasets:
        if item.get("name") and item["name"].lower() in text:
            return item
    normalized_text = re.sub(r"(示例|数据集|数据|[\s_-])", "", text)
    for item in datasets:
        normalized_name = re.sub(r"(示例|数据集|数据|[\s_-])", "", item.get("name", "").lower())
        if normalized_name and normalized_name in normalized_text:
            return item
        trait = item.get("trait", "").lower()
        if trait and trait in text:
            return item
    species_terms = {
        "rice": ("水稻", "rice", "oryza"),
        "wheat": ("小麦", "wheat", "triticum"),
    }
    for species, terms in species_terms.items():
        if any(term in text for term in terms):
            match = next((item for item in datasets if species in item.get("species", "").lower()), None)
            if match:
                return match
    if requested_id:
        dataset = by_id.get(requested_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="数据集不存在或不属于当前用户")
        return dataset
    return by_id.get(state.get("active_dataset_id"))

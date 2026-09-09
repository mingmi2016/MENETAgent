import json
import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from agent import MenetTask, MenetWorkflow, TaskIntent
from agent.intent import IntentParser
from agent.llm import AgentIntentParser, CompatibleLLMClient, load_llm_settings, save_llm_settings
from agent.store import TaskStore
from agent.tools import MenetTools
from agent.api import ChatRequest, _latest_compatible_run, _prepare_task_directories, _resolve_dataset, _validate_paths, _with_runtime
from agent.demo_data import register_demo_datasets
from agent.runner import run_task
from agent.quality import build_quality_report
from utils.dataset import create_dual_scale_dataloader
from utils.train import _write_progress, evaluate_ridge_baseline
from utils.utils import split_data


def make_dataset(root: Path, samples: int = 12, snps: int = 128) -> Path:
    data = root / "data"
    (data / "genotype").mkdir(parents=True)
    (data / "phenotype").mkdir(parents=True)
    rng = np.random.default_rng(42)
    ids = [f"S{i:03d}" for i in range(samples)]
    markers = [f"{(i // 32) + 1}_{1000 + i}" for i in range(snps)]
    genotype = pd.DataFrame(rng.choice([-1, 0, 1], size=(samples, snps)), index=ids, columns=markers)
    phenotype = pd.DataFrame({"culmlength": genotype.iloc[:, :8].sum(axis=1)}, index=ids)
    genotype.to_csv(data / "genotype/genotype.csv", index_label="FID")
    phenotype.to_csv(data / "phenotype/culmlength.csv", index_label="ID")
    return data


def test_intent_parser_extracts_training_request():
    parsed = IntentParser().parse("请用 GPU 自动划分数据训练 MENET 50 轮，性状是 culmlength，并解释重要 SNP")
    assert parsed.intent == TaskIntent.TRAIN_MODEL
    assert parsed.arguments == {
        "trait": "culmlength",
        "device": "cuda",
        "split_strategy": "random",
        "explain_snp": True,
        "epochs": 50,
    }


def test_llm_prediction_does_not_require_training_only_fields():
    class FakeClient:
        enabled = True

        @staticmethod
        def complete_json(_system_prompt, _message):
            return {
                "intent": "predict_trait",
                "arguments": {"trait": "flowering_arkansas"},
                "missing_fields": ["device", "split_strategy"],
                "confidence": 0.9,
            }

    parsed = AgentIntentParser(client=FakeClient()).parse("请预测性状为 flowering_arkansas")

    assert parsed.intent == TaskIntent.PREDICT_TRAIT
    assert parsed.arguments == {"trait": "flowering_arkansas"}
    assert parsed.missing_fields == []


def test_menet_task_normalizes_gpu_device_alias():
    task = MenetTask(trait="culmlength", device="GPU")
    assert task.device == "cuda"


def test_local_ollama_client_does_not_require_api_key():
    client = CompatibleLLMClient(
        provider="ollama",
        base_url="http://127.0.0.1:11434",
        model="qwen3:8b",
        api_key="",
        enabled=True,
    )
    assert client.enabled


def test_llm_settings_are_persisted_locally(tmp_path):
    path = tmp_path / "llm-settings.json"
    expected = {
        "enabled": True,
        "provider": "ollama",
        "base_url": "http://127.0.0.1:11434",
        "api_key": "",
        "model": "qwen3:8b",
        "intent_model": "qwen3:8b",
        "analysis_model": "qwen3:8b",
    }
    save_llm_settings(path, expected)
    assert load_llm_settings(path) == expected
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_anthropic_compatible_client_uses_messages_api(monkeypatch):
    import agent.llm as llm

    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps({"content": [{"type": "text", "text": '{"ok": true}'}]}).encode()

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["payload"] = json.loads(req.data)
        return Response()

    monkeypatch.setattr(llm.request, "urlopen", fake_urlopen)
    client = CompatibleLLMClient(
        provider="anthropic_compatible",
        base_url="https://relay.example",
        api_key="secret",
        model="claude-haiku-4-5",
        enabled=True,
    )

    assert client.complete_json("Return JSON", "test") == {"ok": True}
    assert captured["url"] == "https://relay.example/v1/messages"
    assert captured["payload"]["model"] == "claude-haiku-4-5"
    assert captured["headers"]["Anthropic-version"] == "2023-06-01"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert "MENET-Agent" in captured["headers"]["User-agent"]


def test_auto_compatible_provider_selects_protocol_by_model():
    claude = CompatibleLLMClient(
        provider="compatible_auto", base_url="https://relay.example", model="claude-haiku-4-5", enabled=True
    )
    gemini = CompatibleLLMClient(
        provider="compatible_auto", base_url="https://relay.example", model="gemini-2.5-pro", enabled=True
    )

    assert claude.protocol_provider == "anthropic_compatible"
    assert gemini.protocol_provider == "openai_compatible"


def test_openai_compatible_endpoint_adds_v1_prefix():
    client = CompatibleLLMClient(
        provider="openai_compatible", base_url="https://relay.example", model="gemini-test", enabled=True
    )

    assert client._versioned_endpoint("chat/completions") == "https://relay.example/v1/chat/completions"


def test_supported_examples_have_complete_intents():
    examples = {
        TaskIntent.INSPECT_DATA: "请检查数据，性状为 culmlength",
        TaskIntent.TRAIN_MODEL: "请训练模型，性状为 culmlength，随机划分，使用 GPU，训练 30 轮",
        TaskIntent.PREDICT_TRAIT: "请预测性状为 culmlength",
        TaskIntent.EVALUATE_MODEL: "请评估模型，性状为 culmlength",
        TaskIntent.EXPLAIN_MODEL: "请解释模型，性状为 culmlength",
        TaskIntent.GENERATE_REPORT: "请生成报告，性状为 culmlength，随机划分",
    }
    for expected, message in examples.items():
        parsed = IntentParser().parse(message)
        assert parsed.intent == expected
        assert parsed.arguments["trait"] == "culmlength"
        assert not parsed.missing_fields


def test_random_split_is_reproducible(tmp_path):
    data = make_dataset(tmp_path)
    task = MenetTask(trait="culmlength", dataset_dir=str(data), output_dir=str(tmp_path / "run"), split_strategy="random")
    first = MenetTools().create_split(task)
    split_dir = tmp_path / "run/input/split"
    expected = {name: (split_dir / name).read_text() for name in ("train_index.txt", "valid_index.txt", "test_index.txt")}
    second = MenetTools().create_split(task)
    actual = {name: (split_dir / name).read_text() for name in expected}
    assert first["success"] and second["success"]
    assert actual == expected
    assert not (data / "split").exists()


def test_training_configuration_snapshot_is_saved_with_model(tmp_path):
    task = MenetTask(
        trait="culmlength", output_dir=str(tmp_path), device="cuda", split_strategy="random",
        train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, explain_snp=True,
        metadata={"training_mode": "research", "split_seed": 17, "seed": 23},
    )
    path = MenetTools._write_training_snapshot(task, "menet", {"lr": 0.001, "batch_size": 16})
    MenetTools._write_training_snapshot(task, "trait_encoder", {"margin": 0.1, "batch_size": 32})
    snapshot = json.loads(Path(path).read_text(encoding="utf-8"))
    assert snapshot["training_mode"] == "research"
    assert snapshot["split_seed"] == 17
    assert snapshot["training_seed"] == 23
    assert snapshot["split_ratios"] == {"train": 0.7, "validation": 0.15, "test": 0.15}
    assert snapshot["components"]["menet"]["lr"] == 0.001
    assert snapshot["components"]["trait_encoder"]["margin"] == 0.1


def test_training_loader_can_drop_singleton_tail_batch():
    rows = 17
    snp = pd.DataFrame({"trait": np.arange(rows), **{f"s{i}": np.ones(rows) for i in range(8)}})
    relatedness = pd.DataFrame({"trait": np.arange(rows), **{f"r{i}": np.ones(rows) for i in range(8)}})
    loader = create_dual_scale_dataloader(
        snp, relatedness, {"batch_size": 8}, shuffle=False, drop_last=rows % 8 == 1
    )
    assert [batch[0].shape[0] for batch in loader] == [8, 8]


def test_dataset_validation_rejects_split_leakage(tmp_path):
    data = make_dataset(tmp_path)
    split = data / "split"
    split.mkdir()
    (split / "train_index.txt").write_text("S000\nS001\nS002\n", encoding="utf-8")
    (split / "valid_index.txt").write_text("S002\nS003\n", encoding="utf-8")
    (split / "test_index.txt").write_text("S004\nS005\n", encoding="utf-8")
    task = MenetTask(trait="culmlength", dataset_dir=str(data), intent=TaskIntent.TRAIN_MODEL)

    result = MenetTools().validate_dataset(task)

    assert not result["success"]
    assert any("数据泄漏" in message for message in result["errors"])


def test_quality_report_flags_overfitting_and_negative_test_r2(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "metrics.json").write_text('{"test_r2": -0.2}', encoding="utf-8")
    (output / "training_history.json").write_text(
        '[{"epoch": 1, "train_r2": 0.9, "val_r2": 0.4}]', encoding="utf-8"
    )
    task = MenetTask(trait="culmlength", output_dir=str(output))
    result = {"status": "completed", "steps": [{"name": "validate_dataset", "data": {"matched_sample_count": 80}}]}

    report = build_quality_report(task, result)

    assert report["overall_status"] == "fail"
    levels = {item["id"]: item["level"] for item in report["checks"]}
    assert levels["sample_size"] == "warning"
    assert levels["test_r2"] == "fail"
    assert levels["overfitting_gap"] == "warning"


def test_split_data_matches_numeric_csv_ids_to_text_split_ids(tmp_path):
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    (split_dir / "train_index.txt").write_text("101\n102\n", encoding="utf-8")
    (split_dir / "valid_index.txt").write_text("103\n", encoding="utf-8")
    (split_dir / "test_index.txt").write_text("104\n", encoding="utf-8")
    phenotype = pd.DataFrame({"trait": [1, 2, 3, 4]}, index=[101, 102, 103, 104])
    genotype = pd.DataFrame({"snp": [0, 1, 0, 1]}, index=["101", "102", "103", "104"])
    _, train, valid, test = split_data(phenotype, genotype, tmp_path)
    assert list(train.index) == ["101", "102"]
    assert list(valid.index) == ["103"]
    assert list(test.index) == ["104"]


def test_store_round_trip(tmp_path):
    store = TaskStore(str(tmp_path / "agent.db"))
    task = MenetTask(trait="culmlength", intent=TaskIntent.INSPECT_DATA)
    store.create(task.to_dict())
    store.update(task.task_id, "completed", {"status": "completed"})
    saved = store.get(task.task_id)
    assert saved["status"] == "completed"
    assert saved["result"]["status"] == "completed"


def test_trained_model_registry_supports_metrics_rename_and_archive(tmp_path):
    store = TaskStore(str(tmp_path / "agent.db"))
    user = store.create_user("Researcher")
    output = tmp_path / "runs" / "trained"
    output.mkdir(parents=True)
    (output / "menet_model.pt").write_bytes(b"model")
    (output / "metrics.json").write_text('{"test_r2": 0.72, "test_loss": 0.18}', encoding="utf-8")
    task = MenetTask(
        trait="plant_height", owner_user_id=user["user_id"], dataset_dir=str(tmp_path / "rice"),
        output_dir=str(output), intent=TaskIntent.TRAIN_MODEL, metadata={"dataset_id": "rice_a"},
    )
    store.create(task.to_dict())
    result = {
        "status": "completed",
        "steps": [{"name": "validate_dataset", "data": {"matched_sample_count": 120, "snp_count": 500}}],
    }
    store.update(task.task_id, "completed", result)

    model = store.register_model(task.to_dict(), result)
    assert model["metrics"]["test_r2"] == 0.72
    assert model["sample_count"] == 120
    assert store.rename_model(model["model_id"], user["user_id"], "株高正式模型")["name"] == "株高正式模型"
    assert store.archive_model(model["model_id"], user["user_id"])["status"] == "archived"
    assert store.list_models(user["user_id"]) == []
    assert len(store.list_models(user["user_id"], include_archived=True)) == 1


def test_store_persists_dataset_and_conversation_state(tmp_path):
    store = TaskStore(str(tmp_path / "agent.db"))
    dataset = {
        "dataset_id": "dataset_example",
        "trait": "culmlength",
        "dataset_dir": str(tmp_path / "uploads/dataset_example"),
        "genotype_filename": "genotype.csv",
        "phenotype_filename": "culmlength.csv",
        "genotype_size": 120,
        "phenotype_size": 30,
        "genotype_sha256": "a" * 64,
        "phenotype_sha256": "b" * 64,
    }
    store.create_dataset(dataset)
    conversation = store.ensure_conversation("conv_example")
    store.add_message("conv_example", "user", "检查数据")
    store.update_conversation("conv_example", {"active_dataset_id": dataset["dataset_id"], "active_trait": "culmlength"})

    saved = store.get_conversation(conversation["conversation_id"])
    assert store.get_dataset("dataset_example")["genotype_sha256"] == "a" * 64
    assert saved["state"]["active_trait"] == "culmlength"
    assert saved["messages"][0]["content"] == "检查数据"


def test_conversation_history_has_titles_and_is_user_scoped(tmp_path):
    store = TaskStore(str(tmp_path / "agent.db"))
    alice = store.create_user("Alice")
    bob = store.create_user("Bob")
    first = store.ensure_conversation(user_id=alice["user_id"])
    second = store.ensure_conversation(user_id=alice["user_id"])
    store.ensure_conversation(user_id=bob["user_id"])
    store.add_message(first["conversation_id"], "user", "请检查第一批水稻数据")
    store.add_message(second["conversation_id"], "user", "请训练小麦产量模型")

    history = store.list_conversations(alice["user_id"])
    assert {item["conversation_id"] for item in history} == {first["conversation_id"], second["conversation_id"]}
    assert {item["title"] for item in history} == {"请检查第一批水稻数据", "请训练小麦产量模型"}
    assert all(item["user_id"] == alice["user_id"] for item in history)


def test_chat_task_is_linked_to_conversation_without_placeholder_reply(tmp_path, monkeypatch):
    import agent.api as api

    task_store = TaskStore(str(tmp_path / "agent.db"))
    user = task_store.create_user("Researcher")
    dataset = dataset_record(
        "rice_chat",
        user["user_id"],
        "水稻株高",
        "rice",
        Path(__file__).resolve().parents[1] / "data/demo",
    )
    task_store.create_dataset(dataset)
    conversation = task_store.ensure_conversation(user_id=user["user_id"])
    monkeypatch.setattr(api, "store", task_store)
    monkeypatch.setattr(api, "_submit_task", lambda task: None)

    response = api.chat(ChatRequest(
        message="请检查数据，性状为 plant_height",
        user_id=user["user_id"],
        dataset_id=dataset["dataset_id"],
        conversation_id=conversation["conversation_id"],
    ))

    saved_task = task_store.get(response["task_id"], user["user_id"])
    saved_conversation = task_store.get_conversation(conversation["conversation_id"], user_id=user["user_id"])
    assert saved_task["task"]["metadata"]["conversation_id"] == conversation["conversation_id"]
    assert saved_task["task"]["metadata"]["dataset_id"] == dataset["dataset_id"]
    assert [message["role"] for message in saved_conversation["messages"]] == ["user"]


def test_chat_prediction_uses_explicit_registered_model(tmp_path, monkeypatch):
    import agent.api as api

    task_store = TaskStore(str(tmp_path / "agent.db"))
    user = task_store.create_user("Researcher")
    dataset = dataset_record("rice_model", user["user_id"], "水稻株高", "rice", tmp_path)
    Path(dataset["dataset_dir"]).mkdir(parents=True)
    task_store.create_dataset(dataset)
    output = tmp_path / "runs" / "source"
    output.mkdir(parents=True)
    (output / "menet_model.pt").write_bytes(b"model")
    training = MenetTask(
        trait="plant_height", owner_user_id=user["user_id"], dataset_dir=dataset["dataset_dir"],
        output_dir=str(output), intent=TaskIntent.TRAIN_MODEL, metadata={"dataset_id": dataset["dataset_id"]},
    )
    task_store.create(training.to_dict())
    training_result = {"status": "completed", "steps": []}
    task_store.update(training.task_id, "completed", training_result)
    model = task_store.register_model(training.to_dict(), training_result)
    conversation = task_store.ensure_conversation(user_id=user["user_id"])
    monkeypatch.setattr(api, "store", task_store)
    monkeypatch.setattr(api, "service_root", tmp_path)
    monkeypatch.setattr(api, "intent_parser", IntentParser())
    monkeypatch.setattr(api, "_submit_task", lambda task: None)

    response = api.chat(ChatRequest(
        message="请预测性状为 plant_height", user_id=user["user_id"], dataset_id=dataset["dataset_id"],
        conversation_id=conversation["conversation_id"], model_id=model["model_id"],
    ))

    prediction = task_store.get(response["task_id"], user["user_id"])["task"]
    assert prediction["source_output_dir"] == str(output)
    assert prediction["metadata"]["model_id"] == model["model_id"]
    assert prediction["metadata"]["model_name"] == model["name"]


def test_task_output_is_isolated_and_source_is_preserved():
    training = MenetTask(trait="culmlength", output_dir="runs", intent=TaskIntent.TRAIN_MODEL)
    _prepare_task_directories(training, "runs")
    assert Path(training.output_dir).name == training.task_id
    assert training.source_output_dir is None

    prediction = MenetTask(trait="culmlength", output_dir="runs/model_a", intent=TaskIntent.PREDICT_TRAIT)
    _prepare_task_directories(prediction, "runs/model_a")
    assert Path(prediction.output_dir).name == prediction.task_id
    assert Path(prediction.source_output_dir).name == "model_a"


def test_task_round_trip_restores_intent_enum():
    original = MenetTask(trait="culmlength", intent=TaskIntent.INSPECT_DATA)
    restored = MenetTask(**original.to_dict())
    assert restored.intent == TaskIntent.INSPECT_DATA
    assert restored.to_dict()["intent"] == "inspect_data"


def test_queued_task_honors_cancellation_marker(tmp_path):
    store = TaskStore(str(tmp_path / "agent.db"))
    output = tmp_path / "run"
    output.mkdir()
    (output / "cancel.requested").write_text("now", encoding="utf-8")
    task = MenetTask(trait="plant_height", output_dir=str(output), intent=TaskIntent.TRAIN_MODEL)
    store.create(task.to_dict())
    result = run_task(task.to_dict(), str(store.path))
    assert result["status"] == "cancelled"
    assert store.get(task.task_id)["status"] == "cancelled"


def test_epoch_progress_is_written_atomically(tmp_path):
    _write_progress({"model_path": str(tmp_path)}, "training_menet", 3, 10, {"val_r2": 0.4})
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["current_epoch"] == 3
    assert progress["percent"] == 30.0
    assert not (tmp_path / "progress.json.tmp").exists()


def dataset_record(dataset_id: str, user_id: str, name: str, species: str, root: Path, is_demo: bool = False):
    return {
        "dataset_id": dataset_id,
        "user_id": user_id,
        "name": name,
        "species": species,
        "trait": "plant_height",
        "dataset_dir": str(root / dataset_id),
        "genotype_filename": "genotype.csv",
        "phenotype_filename": "plant_height.csv",
        "genotype_size": 10,
        "phenotype_size": 10,
        "genotype_sha256": "a" * 64,
        "phenotype_sha256": "b" * 64,
        "source_url": "",
        "license": "",
        "is_demo": is_demo,
    }


def test_multiple_users_are_isolated_but_share_demo_data(tmp_path):
    store = TaskStore(str(tmp_path / "agent.db"))
    alice = store.create_user("Alice")
    bob = store.create_user("Bob")
    store.create_dataset(dataset_record("alice_rice", alice["user_id"], "Alice 水稻", "rice", tmp_path))
    store.create_dataset(dataset_record("bob_wheat", bob["user_id"], "Bob 小麦", "wheat", tmp_path))
    store.create_dataset(dataset_record("shared_rice", "user_local", "共享水稻", "rice", tmp_path, True))

    assert {item["dataset_id"] for item in store.list_datasets(alice["user_id"])} == {"alice_rice", "shared_rice"}
    assert {item["dataset_id"] for item in store.list_datasets(bob["user_id"])} == {"bob_wheat", "shared_rice"}
    conversation = store.ensure_conversation(user_id=alice["user_id"])
    assert store.get_conversation(conversation["conversation_id"], user_id=bob["user_id"]) is None
    with __import__("pytest").raises(PermissionError):
        store.ensure_conversation(conversation["conversation_id"], bob["user_id"])


def test_dataset_resolution_understands_history_name_and_species(tmp_path, monkeypatch):
    import agent.api as api

    store = TaskStore(str(tmp_path / "agent.db"))
    user = store.create_user("Researcher")
    store.create_dataset(dataset_record("rice_old", user["user_id"], "水稻株高第一批", "rice", tmp_path))
    store.create_dataset(dataset_record("rice_new", user["user_id"], "水稻株高第二批", "rice", tmp_path))
    store.create_dataset(dataset_record("wheat", user["user_id"], "小麦产量", "wheat", tmp_path))
    monkeypatch.setattr(api, "store", store)
    state = {"active_dataset_id": "rice_new", "recent_dataset_ids": ["rice_new", "rice_old"]}

    assert _resolve_dataset("检查上一个数据集", user["user_id"], "rice_new", state)["dataset_id"] == "rice_old"
    assert _resolve_dataset("检查小麦数据", user["user_id"], "rice_new", state)["dataset_id"] == "wheat"
    assert _resolve_dataset("检查水稻株高第一批", user["user_id"], None, state)["dataset_id"] == "rice_old"
    assert _resolve_dataset("检查当前数据", user["user_id"], None, state)["dataset_id"] == "rice_new"


def test_dataset_resolution_prefers_named_environment_over_species_fallback(tmp_path, monkeypatch):
    import agent.api as api

    store = TaskStore(str(tmp_path / "agent.db"))
    user = store.create_user("Researcher")
    env1 = dataset_record("wheat_env1", user["user_id"], "小麦环境1产量示例", "wheat", tmp_path)
    env1["trait"] = "grain_yield_env1"
    env2 = dataset_record("wheat_env2", user["user_id"], "小麦环境2产量示例", "wheat", tmp_path)
    env2["trait"] = "grain_yield_env2"
    store.create_dataset(env1)
    store.create_dataset(env2)
    monkeypatch.setattr(api, "store", store)

    selected = _resolve_dataset("请检查小麦环境1产量数据", user["user_id"], "wheat_env2", {})
    assert selected["dataset_id"] == "wheat_env1"


def test_task_lookup_is_scoped_to_owner(tmp_path):
    store = TaskStore(str(tmp_path / "agent.db"))
    alice = store.create_user("Alice")
    bob = store.create_user("Bob")
    task = MenetTask(trait="plant_height", owner_user_id=alice["user_id"], intent=TaskIntent.INSPECT_DATA)
    store.create(task.to_dict())
    assert store.get(task.task_id, alice["user_id"]) is not None
    assert store.get(task.task_id, bob["user_id"]) is None


def test_latest_compatible_model_run_is_found_automatically(tmp_path, monkeypatch):
    import agent.api as api

    store = TaskStore(str(tmp_path / "agent.db"))
    monkeypatch.setattr(api, "store", store)
    user = store.create_user("Researcher")
    output = tmp_path / "runs/model"
    output.mkdir(parents=True)
    (output / "menet_model.pt").write_bytes(b"model")
    task = MenetTask(
        trait="plant_height", owner_user_id=user["user_id"], dataset_dir=str(tmp_path / "rice"),
        output_dir=str(output), intent=TaskIntent.TRAIN_MODEL,
    )
    store.create(task.to_dict())
    store.update(task.task_id, "completed", {"status": "completed"})
    found = _latest_compatible_run(user["user_id"], str(tmp_path / "rice"), "plant_height")
    assert found["task_id"] == task.task_id


def test_runtime_metadata_learns_from_matching_completed_tasks(tmp_path, monkeypatch):
    import agent.api as api

    store = TaskStore(str(tmp_path / "agent.db"))
    monkeypatch.setattr(api, "store", store)
    previous = MenetTask(trait="plant_height", dataset_dir="data/rice", intent=TaskIntent.TRAIN_MODEL)
    store.create(previous.to_dict())
    store.update(previous.task_id, "completed", {"status": "completed"})
    with store._connect() as connection:
        connection.execute(
            "UPDATE tasks SET created_at='2026-01-01T00:00:00+00:00', "
            "started_at='2026-01-01T00:00:00+00:00', updated_at='2026-01-01T00:10:00+00:00', "
            "finished_at='2026-01-01T00:10:00+00:00' WHERE task_id=?",
            (previous.task_id,),
        )
    current = MenetTask(trait="plant_height", dataset_dir="data/rice", intent=TaskIntent.TRAIN_MODEL)
    store.create(current.to_dict())

    runtime = _with_runtime(store.get(current.task_id))["runtime"]
    assert runtime["estimate_source"] == "local_history"
    assert runtime["estimated_seconds_low"] == 420
    assert runtime["estimated_seconds_high"] == 900


def test_runtime_metadata_uses_project_benchmark_for_demo(tmp_path, monkeypatch):
    import agent.api as api

    store = TaskStore(str(tmp_path / "agent.db"))
    monkeypatch.setattr(api, "store", store)
    task = MenetTask(
        trait="plant_height",
        dataset_dir=str(Path(__file__).resolve().parents[1] / "data/demo/demo_rice_plant_height"),
        device="cuda",
        epochs=50,
        intent=TaskIntent.TRAIN_MODEL,
    )
    store.create(task.to_dict())
    runtime = _with_runtime(store.get(task.task_id))["runtime"]
    assert runtime["estimate_source"] == "local_benchmark"
    assert runtime["estimated_seconds_low"] == 34
    assert runtime["estimated_seconds_high"] == 55


def test_prepared_crop_demo_datasets_pass_validation():
    project_root = Path(__file__).resolve().parents[1]
    catalog = json.loads((project_root / "data/demo/catalog.json").read_text(encoding="utf-8"))
    assert len(catalog["datasets"]) == 4
    for dataset in catalog["datasets"]:
        dataset_dir = Path(dataset["dataset_dir"])
        if not dataset_dir.is_absolute():
            dataset_dir = project_root / dataset_dir
        task = MenetTask(
            trait=dataset["trait"], dataset_dir=str(dataset_dir), intent=TaskIntent.INSPECT_DATA
        )
        result = MenetTools().validate_dataset(task)
        assert result["success"], result["errors"]
        assert result["data"]["matched_sample_count"] >= 300
        assert result["data"]["snp_count"] >= 1200


def test_demo_catalog_is_registered_for_new_users(tmp_path):
    store = TaskStore(str(tmp_path / "agent.db"))
    created = register_demo_datasets(store, Path(__file__).resolve().parents[1])
    user = store.create_user("Researcher")
    assert created == 4
    demos = [item for item in store.list_datasets(user["user_id"]) if item["is_demo"]]
    assert len(demos) == 4
    assert all(item["sample_count"] and item["snp_count"] for item in demos)


def test_store_rebases_only_configured_path_prefixes(tmp_path):
    store = TaskStore(str(tmp_path / "agent.db"))
    old_root = "/workspace/old"
    new_root = "/workspace/new"
    task = MenetTask(
        trait="plant_height",
        dataset_dir=f"{old_root}/data/demo/rice",
        output_dir=f"{old_root}/runs/user/task",
    )
    store.create(task.to_dict())
    conversation = store.ensure_conversation("conv_rebase")
    store.update_conversation(
        conversation["conversation_id"],
        {"active_dataset_dir": f"{old_root}/data/demo/rice", "active_run_dir": f"{old_root}/runs/user/task"},
    )

    store.rebase_paths({
        f"{old_root}/data/demo": f"{new_root}/data/demo",
        f"{old_root}/runs": f"{new_root}/runs",
    })

    saved_task = store.get(task.task_id)
    saved_conversation = store.get_conversation(conversation["conversation_id"])
    assert saved_task["task"]["dataset_dir"] == f"{new_root}/data/demo/rice"
    assert saved_task["task"]["output_dir"] == f"{new_root}/runs/user/task"
    assert saved_conversation["state"]["active_run_dir"] == f"{new_root}/runs/user/task"


def test_inspection_workflow_returns_structured_failure_for_small_example(tmp_path):
    data = make_dataset(tmp_path, snps=16)
    task = MenetTask(trait="culmlength", dataset_dir=str(data), intent=TaskIntent.INSPECT_DATA)
    result = MenetWorkflow().run(task)
    assert result.status.value == "completed"
    assert result.steps[0]["status"] == "validated"
    assert result.steps[0]["data"]["snp_count"] == 16


def test_report_tool_writes_html_from_metrics(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "metrics.json").write_text(json.dumps({"test_loss": 1.2, "test_r2": 0.5, "device": "cpu"}))
    (output / "training_history.json").write_text(json.dumps([{"epoch": 1, "train_loss": 1.0, "val_loss": 1.1, "val_r2": 0.4}]))
    task = MenetTask(trait="culmlength", output_dir=str(output))
    result = MenetTools().generate_report(task)
    assert result["success"]
    assert (output / "report.html").is_file()


def test_api_path_guard_rejects_parent_directory():
    with pytest.raises(Exception):
        _validate_paths("../outside", "runs")


def test_genomic_ridge_baseline_uses_same_train_test_split():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(30, 1, 8)).astype("float32")
    y = (2 * x[:, 0, 0] - x[:, 0, 1]).astype("float32")
    relatedness = np.zeros((30, 1, 1), dtype="float32")
    train = TensorDataset(torch.from_numpy(x[:20]), torch.from_numpy(relatedness[:20]), torch.from_numpy(y[:20]))
    test = TensorDataset(torch.from_numpy(x[20:]), torch.from_numpy(relatedness[20:]), torch.from_numpy(y[20:]))

    metrics = evaluate_ridge_baseline(DataLoader(train, batch_size=4), DataLoader(test, batch_size=4))

    assert metrics["method"] == "genomic_ridge"
    assert metrics["train_sample_count"] == 20
    assert metrics["test_sample_count"] == 10
    assert metrics["test_r2"] > 0.9


def test_model_comparison_is_scoped_by_user_trait_and_dataset(tmp_path, monkeypatch):
    import agent.api as api

    store = TaskStore(str(tmp_path / "agent.db"))
    monkeypatch.setattr(api, "store", store)
    alice = store.create_user("Alice")
    bob = store.create_user("Bob")

    def add_model(user_id, trait, dataset_id, score):
        output = tmp_path / user_id / f"{trait}-{score}"
        output.mkdir(parents=True)
        (output / "menet_model.pt").write_bytes(b"model")
        (output / "metrics.json").write_text(json.dumps({"test_r2": score}), encoding="utf-8")
        task = MenetTask(
            trait=trait, owner_user_id=user_id, dataset_dir=str(tmp_path / dataset_id),
            output_dir=str(output), intent=TaskIntent.TRAIN_MODEL,
            metadata={"dataset_id": dataset_id},
        )
        store.create(task.to_dict())
        store.update(task.task_id, "completed", {"status": "completed"})
        store.register_model(task.to_dict(), {"status": "completed"})

    add_model(alice["user_id"], "height", "rice", 0.4)
    add_model(alice["user_id"], "height", "rice", 0.6)
    add_model(alice["user_id"], "yield", "rice", 0.8)
    add_model(bob["user_id"], "height", "rice", 0.9)

    result = api.compare_trained_models(alice["user_id"], "height", "rice")
    assert result["summary"]["run_count"] == 2
    assert result["summary"]["stable_enough_to_estimate"] is True
    assert result["summary"]["test_r2_mean"] == 0.5
    assert result["summary"]["test_r2_range"] == pytest.approx(0.2)


def test_repeat_training_creates_reproducible_independent_tasks(tmp_path, monkeypatch):
    import agent.api as api

    store = TaskStore(str(tmp_path / "agent.db"))
    monkeypatch.setattr(api, "store", store)
    monkeypatch.setattr(api, "service_root", tmp_path)
    submitted = []
    monkeypatch.setattr(api, "_submit_task", lambda task: submitted.append(task))
    user = store.create_user("Researcher")
    dataset = tmp_path / "data" / "rice"
    output = tmp_path / "runs" / "source"
    output.mkdir(parents=True)
    (output / "menet_model.pt").write_bytes(b"model")
    (output / "metrics.json").write_text(json.dumps({"test_r2": 0.5}), encoding="utf-8")
    source = MenetTask(
        trait="height", owner_user_id=user["user_id"], dataset_dir=str(dataset),
        output_dir=str(output), epochs=20, intent=TaskIntent.TRAIN_MODEL,
        metadata={"dataset_id": "rice"},
    )
    store.create(source.to_dict())
    store.update(source.task_id, "completed", {"status": "completed"})
    model = store.register_model(source.to_dict(), {"status": "completed"})

    response = api.repeat_model_training(
        model["model_id"], api.RepeatTrainingRequest(user_id=user["user_id"], repeats=3)
    )

    assert len(response["tasks"]) == 3
    assert len(submitted) == 3
    assert all(task.split_strategy == "random" and task.epochs == 20 for task in submitted)
    assert len({task.metadata["split_seed"] for task in submitted}) == 3
    assert len({task.metadata["seed"] for task in submitted}) == 3


def test_mcp_exposes_only_structured_menet_capabilities():
    from agent.mcp_server import mcp

    names = {tool.name for tool in asyncio.run(mcp.list_tools())}
    assert names == {
        "menet_inspect_dataset", "menet_submit_training", "menet_submit_prediction",
        "menet_submit_evaluation", "menet_submit_explanation", "menet_get_task",
        "menet_cancel_task", "menet_cleanup_task",
    }

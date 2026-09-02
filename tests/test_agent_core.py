import json
from pathlib import Path

import numpy as np
import pandas as pd

from agent import MenetTask, MenetWorkflow, TaskIntent
from agent.intent import IntentParser
from agent.store import TaskStore
from agent.tools import MenetTools
from agent.api import _latest_compatible_run, _prepare_task_directories, _resolve_dataset, _validate_paths, _with_runtime
from agent.demo_data import register_demo_datasets
from agent.runner import run_task
from utils.dataset import create_dual_scale_dataloader
from utils.train import _write_progress
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


def test_training_loader_can_drop_singleton_tail_batch():
    rows = 17
    snp = pd.DataFrame({"trait": np.arange(rows), **{f"s{i}": np.ones(rows) for i in range(8)}})
    relatedness = pd.DataFrame({"trait": np.arange(rows), **{f"r{i}": np.ones(rows) for i in range(8)}})
    loader = create_dual_scale_dataloader(
        snp, relatedness, {"batch_size": 8}, shuffle=False, drop_last=rows % 8 == 1
    )
    assert [batch[0].shape[0] for batch in loader] == [8, 8]


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
        task = MenetTask(
            trait=dataset["trait"], dataset_dir=dataset["dataset_dir"], intent=TaskIntent.INSPECT_DATA
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
    import pytest

    with pytest.raises(Exception):
        _validate_paths("../outside", "runs")

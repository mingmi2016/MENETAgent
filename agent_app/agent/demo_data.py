"""Register prepared shared demo datasets in the local metadata store."""

import json
from pathlib import Path

from .store import TaskStore


def register_demo_datasets(store: TaskStore, project_root: Path) -> int:
    catalog_path = project_root / "data" / "demo" / "catalog.json"
    if not catalog_path.is_file():
        return 0
    datasets = json.loads(catalog_path.read_text(encoding="utf-8")).get("datasets", [])
    created = 0
    for catalog_dataset in datasets:
        dataset = dict(catalog_dataset)
        dataset_path = Path(dataset["dataset_dir"])
        if not dataset_path.is_absolute():
            dataset["dataset_dir"] = str((project_root / dataset_path).resolve())
        existing = store.get_dataset(dataset["dataset_id"])
        if existing is None:
            store.create_dataset(dataset)
            created += 1
        else:
            store.update_dataset_metadata(dataset)
    return created

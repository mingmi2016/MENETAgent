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
    for dataset in datasets:
        existing = store.get_dataset(dataset["dataset_id"])
        if existing is None:
            store.create_dataset(dataset)
            created += 1
        else:
            store.update_dataset_stats(dataset["dataset_dir"], dataset["sample_count"], dataset["snp_count"])
    return created

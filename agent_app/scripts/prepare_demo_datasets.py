#!/usr/bin/env python3
"""Download pinned public crop datasets and convert them to MENET CSV layout."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import rdata


ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = ROOT / "data" / "demo"
SOURCE_ROOT = DEMO_ROOT / "_sources"
RAINBOWR_COMMIT = "aa4493db13bfc3a9b0ffbf593b1d4bf8b8eef7eb"
BGLR_COMMIT = "000536479fcdb9eb05a5c8e595174071c7d16e0c"
SOURCES = {
    "rice": {
        "filename": "Rice_Zhao_etal.rda",
        "url": f"https://raw.githubusercontent.com/KosukeHamazaki/RAINBOWR/{RAINBOWR_COMMIT}/data/Rice_Zhao_etal.rda",
        "page": "https://github.com/KosukeHamazaki/RAINBOWR",
        "license": "MIT (RAINBOWR package); cite Zhao et al. 2010 and 2011",
    },
    "wheat": {
        "filename": "wheat.RData",
        "url": f"https://raw.githubusercontent.com/gdlc/BGLR-R/{BGLR_COMMIT}/data/wheat.RData",
        "page": "https://github.com/gdlc/BGLR-R",
        "license": "GPL-3 (BGLR package); data source CIMMYT",
    },
}


def download(source: dict) -> Path:
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    destination = SOURCE_ROOT / source["filename"]
    if not destination.is_file():
        with urlopen(source["url"], timeout=120) as response:
            destination.write_bytes(response.read())
    return destination


def write_dataset(dataset_id: str, species: str, name: str, trait: str, genotype: pd.DataFrame,
                  phenotype: pd.Series, source: dict) -> dict:
    common = genotype.index.intersection(phenotype.dropna().index)
    genotype = genotype.loc[common].copy()
    phenotype = phenotype.loc[common].astype(float)
    dataset_dir = DEMO_ROOT / dataset_id
    genotype_dir = dataset_dir / "genotype"
    phenotype_dir = dataset_dir / "phenotype"
    split_dir = dataset_dir / "split"
    genotype_dir.mkdir(parents=True, exist_ok=True)
    phenotype_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    genotype.index.name = "sample_id"
    genotype.to_csv(genotype_dir / "genotype.csv")
    phenotype.to_frame(trait).rename_axis("sample_id").to_csv(phenotype_dir / f"{trait}.csv")
    rng = np.random.default_rng(42)
    ids = np.asarray(common.astype(str), dtype=object)
    rng.shuffle(ids)
    train_end = max(1, int(len(ids) * 0.7))
    valid_end = min(len(ids) - 1, train_end + max(1, int(len(ids) * 0.15)))
    for filename, values in {
        "train_index.txt": ids[:train_end],
        "valid_index.txt": ids[train_end:valid_end],
        "test_index.txt": ids[valid_end:],
    }.items():
        (split_dir / filename).write_text("\n".join(values.tolist()) + "\n", encoding="utf-8")
    genotype_path = genotype_dir / "genotype.csv"
    phenotype_path = phenotype_dir / f"{trait}.csv"
    return {
        "dataset_id": dataset_id,
        "user_id": "user_local",
        "name": name,
        "species": species,
        "trait": trait,
        "dataset_dir": str(dataset_dir),
        "genotype_filename": genotype_path.name,
        "phenotype_filename": phenotype_path.name,
        "genotype_size": genotype_path.stat().st_size,
        "phenotype_size": phenotype_path.stat().st_size,
        "genotype_sha256": hashlib.sha256(genotype_path.read_bytes()).hexdigest(),
        "phenotype_sha256": hashlib.sha256(phenotype_path.read_bytes()).hexdigest(),
        "source_url": source["page"],
        "license": source["license"],
        "is_demo": True,
        "sample_count": len(common),
        "snp_count": genotype.shape[1],
    }


def prepare_rice() -> list:
    source = SOURCES["rice"]
    rice = rdata.read_rda(download(source))["Rice_Zhao_etal"]
    genotype = rice["genoScore"].T
    marker_map = rice["genoMap"]
    genotype.columns = [
        f"{int(marker_map.loc[marker, 'chr'])}_{int(marker_map.loc[marker, 'pos'])}_{marker}"
        for marker in genotype.columns
    ]
    genotype.index = genotype.index.astype(str)
    phenotype = rice["pheno"]
    return [
        write_dataset("demo_rice_plant_height", "Oryza sativa (rice)", "水稻株高示例", "plant_height",
                      genotype, phenotype["Plant.height"], source),
        write_dataset("demo_rice_flowering_arkansas", "Oryza sativa (rice)", "水稻开花期示例", "flowering_arkansas",
                      genotype, phenotype["Flowering.time.at.Arkansas"], source),
    ]


def prepare_wheat() -> list:
    source = SOURCES["wheat"]
    wheat = rdata.read_rda(download(source))
    x = wheat["wheat.X"]
    y = wheat["wheat.Y"]
    sample_ids = [str(value) for value in y.coords["dim_0"].values]
    raw_markers = [str(value) for value in x.coords["dim_1"].values]
    marker_names = [f"0_{index + 1}_{marker.replace('.', '_')}" for index, marker in enumerate(raw_markers)]
    genotype = pd.DataFrame(x.values.astype(np.int8), index=sample_ids, columns=marker_names)
    return [
        write_dataset("demo_wheat_yield_env1", "Triticum aestivum (wheat)", "小麦环境1产量示例", "grain_yield_env1",
                      genotype, pd.Series(y.values[:, 0], index=sample_ids), source),
        write_dataset("demo_wheat_yield_env2", "Triticum aestivum (wheat)", "小麦环境2产量示例", "grain_yield_env2",
                      genotype, pd.Series(y.values[:, 1], index=sample_ids), source),
    ]


def main() -> None:
    catalog = prepare_rice() + prepare_wheat()
    (DEMO_ROOT / "catalog.json").write_text(
        json.dumps({"datasets": catalog}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"datasets": catalog}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

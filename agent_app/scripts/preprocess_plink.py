"""Convert an uploaded PLINK 1 binary archive into a bounded MENET matrix.

Run this on an LSF CPU worker. The original archive and phenotype views stay intact.
"""

import argparse
import csv
import json
import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


MAX_MARKERS = 8192
CANDIDATE_MARKERS = 8192
MAX_MISSING_RATE = 0.1
MIN_MAF = 0.01
LOOKUP = np.array([[-1, np.nan, 0, 1][(byte >> shift) & 3]
                   for byte in range(256) for shift in (0, 2, 4, 6)], dtype=np.float32).reshape(256, 4)


def write_status(path, status, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat(), **values}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def archive_members(archive):
    by_suffix = {}
    for member in archive.getmembers():
        if not member.isfile():
            continue
        suffix = Path(member.name).suffix.lower()
        if suffix in {".bed", ".bim", ".fam"}:
            if suffix in by_suffix:
                raise ValueError(f"压缩包包含多个 {suffix} 文件")
            by_suffix[suffix] = member
    if set(by_suffix) != {".bed", ".bim", ".fam"}:
        raise ValueError("PLINK 压缩包必须恰好包含一组 .bed、.bim、.fam")
    if len({Path(member.name).stem for member in by_suffix.values()}) != 1:
        raise ValueError(".bed、.bim、.fam 的文件前缀必须一致")
    return by_suffix


def convert(dataset_dir, status_path):
    source_dir = dataset_dir / "genotype" / "source"
    archives = list(source_dir.glob("*.tar.gz"))
    if len(archives) != 1:
        raise ValueError("应有且仅有一个 PLINK .tar.gz 文件")
    archive_path = archives[0]
    output_path = dataset_dir / "genotype" / "genotype.csv"
    report_path = dataset_dir / "genotype" / "preprocessing_report.json"
    if output_path.exists() and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        write_status(status_path, "completed", report={key: value for key, value in report.items() if key != "marker_indices_zero_based"})
        return
    with tempfile.TemporaryDirectory(prefix="menet_plink_", dir=source_dir) as scratch:
        scratch_dir = Path(scratch)
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive_members(archive)
            for suffix in (".fam", ".bim"):
                with archive.extractfile(members[suffix]) as source, (scratch_dir / f"input{suffix}").open("wb") as target:
                    shutil.copyfileobj(source, target, 1024 * 1024)
            fam_rows = [line.split() for line in (scratch_dir / "input.fam").read_text().splitlines() if line.strip()]
            sample_ids = [row[1] for row in fam_rows if len(row) >= 2]
            if len(sample_ids) != len(fam_rows) or len(sample_ids) < 3 or len(set(sample_ids)) != len(sample_ids):
                raise ValueError(".fam 样本 ID 缺失、重复或样本数量不足")
            phenotype_source = next((dataset_dir / "phenotype" / "source").iterdir())
            with phenotype_source.open("r", encoding="utf-8-sig", newline="") as handle:
                first = handle.readline()
                handle.seek(0)
                phenotype_ids = {row[0] for row in csv.reader(handle, delimiter="\t" if "\t" in first else ",") if row}
            phenotype_ids.discard(first.split("\t" if "\t" in first else ",")[0].strip())
            matched = len(set(sample_ids) & phenotype_ids)
            if matched < 3:
                raise ValueError(f"PLINK 与表型仅匹配 {matched} 个样本，请检查 ID 列")
            marker_count = sum(1 for line in (scratch_dir / "input.bim").open("r") if line.strip())
            if marker_count < 128:
                raise ValueError(f".bim 只有 {marker_count} 个标记，MENET 至少需要 128 个")
            candidate_count = min(CANDIDATE_MARKERS, marker_count)
            candidates = set(np.linspace(0, marker_count - 1, candidate_count, dtype=np.int64).tolist())
            marker_names = {}
            with (scratch_dir / "input.bim").open("r") as handle:
                for index, line in enumerate(handle):
                    if index not in candidates:
                        continue
                    fields = line.split()
                    if len(fields) < 6:
                        raise ValueError(f".bim 第 {index + 1} 行格式错误")
                    chrom = fields[0].removeprefix("chr")
                    marker_names[index] = f"{chrom}_{fields[3]}_{fields[1]}"
            bytes_per_marker = (len(sample_ids) + 3) // 4
            bed_member = members[".bed"]
            expected_size = 3 + bytes_per_marker * marker_count
            if bed_member.size != expected_size:
                raise ValueError(f".bed 大小与 .bim/.fam 不匹配：{bed_member.size} != {expected_size}")
            write_status(status_path, "running", phase="extracting", samples=len(sample_ids), source_markers=marker_count, matched_samples=matched)
            bed_path = scratch_dir / "input.bed"
            with archive.extractfile(bed_member) as source, bed_path.open("wb") as target:
                shutil.copyfileobj(source, target, 8 * 1024 * 1024)
        matrix_path = scratch_dir / "selected.float32"
        matrix = np.memmap(matrix_path, dtype=np.float32, mode="w+", shape=(len(sample_ids), MAX_MARKERS))
        retained = []
        imputed = 0
        with bed_path.open("rb") as handle:
            if handle.read(3) != b"\x6c\x1b\x01":
                raise ValueError("仅支持 SNP-major PLINK 1 .bed 文件")
            for position, index in enumerate(sorted(candidates)):
                handle.seek(3 + index * bytes_per_marker)
                raw = np.frombuffer(handle.read(bytes_per_marker), dtype=np.uint8)
                if len(raw) != bytes_per_marker:
                    raise ValueError(".bed 文件提前结束")
                values = LOOKUP[raw].reshape(-1)[:len(sample_ids)]
                missing = np.isnan(values)
                if missing.mean() > MAX_MISSING_RATE:
                    continue
                observed = values[~missing]
                if not len(observed):
                    continue
                maf = min(float((observed + 1).mean() / 2), float((1 - observed).mean() / 2))
                if maf < MIN_MAF:
                    continue
                if missing.any():
                    values = values.copy()
                    values[missing] = float(np.rint(np.median(observed)))
                    imputed += int(missing.sum())
                matrix[:, len(retained)] = values
                retained.append((index, marker_names[index]))
                if position % 2000 == 0:
                    write_status(status_path, "running", phase="filtering", candidates_checked=position + 1, retained_markers=len(retained), source_markers=marker_count)
        if len(retained) < 128:
            raise ValueError(f"质量筛选后只保留 {len(retained)} 个标记，无法用于 MENET")
        names = [name for _, name in retained]
        if len(set(names)) != len(names):
            raise ValueError("筛选后出现重复标记名，请检查 .bim")
        temporary_output = output_path.with_suffix(".csv.tmp")
        write_status(status_path, "running", phase="writing", retained_markers=len(retained), source_markers=marker_count)
        with temporary_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", *names])
            for index, sample_id in enumerate(sample_ids):
                writer.writerow([sample_id, *matrix[index, :len(retained)].astype(np.int8).tolist()])
        report = {
            "format": "PLINK 1 SNP-major", "source_archive": archive_path.name,
            "sample_count": len(sample_ids), "matched_phenotype_samples": matched,
            "source_markers": marker_count, "candidate_markers": candidate_count,
            "retained_markers": len(retained), "max_markers": MAX_MARKERS,
            "selection": "evenly spaced across .bim order, then MAF >= 0.01 and missing rate <= 0.1",
            "missing_values_imputed": imputed, "imputation": "per-marker rounded median",
            "marker_indices_zero_based": [index for index, _ in retained],
            "warning": "This is a bounded marker subset, not a whole-genome conversion; training results depend on this selection.",
        }
        temporary_report = report_path.with_suffix(".json.tmp")
        temporary_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary_output, output_path)
        os.replace(temporary_report, report_path)
        matrix.flush()
        del matrix
    write_status(status_path, "completed", report={key: value for key, value in report.items() if key != "marker_indices_zero_based"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.resolve()
    status_path = dataset_dir / "genotype" / "preprocessing_status.json"
    write_status(status_path, "running", phase="starting")
    try:
        convert(dataset_dir, status_path)
    except Exception as exc:
        write_status(status_path, "failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()

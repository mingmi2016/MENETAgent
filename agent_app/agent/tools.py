"""Allow-listed, deterministic tools exposed to the MENET Agent."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os

import numpy as np
import pandas as pd

from .models import MenetTask, TaskIntent


CORE_ROOT = Path(
    os.environ.get(
        "MENET_CORE_ROOT",
        Path(__file__).resolve().parents[2] / "MENET",
    )
).resolve()


class MenetTools:
    """Computation boundary used by the workflow orchestrator.

    Training methods remain explicit placeholders until the legacy scripts are
    wrapped as resumable workers. The data tools are safe to run repeatedly.
    """

    def validate_dataset(self, task: MenetTask) -> Dict[str, Any]:
        root = Path(task.dataset_dir)
        genotype_path = root / "genotype" / "genotype.csv"
        phenotype_path = root / "phenotype" / f"{task.trait}.csv"
        errors: List[str] = []
        warnings: List[str] = []

        missing = [str(path) for path in (genotype_path, phenotype_path) if not path.is_file()]
        if missing:
            errors.extend(f"缺少必要文件: {path}" for path in missing)
            return self._result(False, "validation_failed", errors, warnings)

        try:
            genotype = pd.read_csv(genotype_path, index_col=0, comment="*")
            phenotype = pd.read_csv(phenotype_path, index_col=0, comment="*")
        except Exception as exc:
            return self._result(False, "validation_failed", [f"读取数据失败: {exc}"], warnings)

        if genotype.empty or phenotype.empty:
            errors.append("基因型和表型数据都不能为空")
        if genotype.shape[1] < 128:
            message = f"当前性状编码器网络至少需要约 128 个 SNP，检测到 {genotype.shape[1]} 个"
            if task.intent == TaskIntent.INSPECT_DATA:
                warnings.append(message)
            else:
                errors.append(message)
        if genotype.index.has_duplicates:
            errors.append("基因型文件包含重复样本 ID")
        if phenotype.index.has_duplicates:
            errors.append("表型文件包含重复样本 ID")
        if phenotype.shape[1] != 1:
            warnings.append(f"表型文件包含 {phenotype.shape[1]} 列，当前将使用第一列作为目标性状")

        common_ids = genotype.index.intersection(phenotype.index)
        only_genotype = genotype.index.difference(phenotype.index)
        only_phenotype = phenotype.index.difference(genotype.index)
        if len(common_ids) == 0:
            errors.append("基因型和表型没有匹配的样本 ID")
        if len(only_genotype):
            warnings.append(f"有 {len(only_genotype)} 个样本只存在于基因型文件")
        if len(only_phenotype):
            warnings.append(f"有 {len(only_phenotype)} 个样本只存在于表型文件")

        genotype_numeric = genotype.apply(pd.to_numeric, errors="coerce")
        phenotype_numeric = phenotype.iloc[:, 0].apply(pd.to_numeric, errors="coerce")
        genotype_missing = int(genotype_numeric.isna().sum().sum())
        phenotype_missing = int(phenotype_numeric.isna().sum())
        if phenotype_missing:
            errors.append(f"表型目标列包含 {phenotype_missing} 个非数值或缺失值")
        elif phenotype_numeric.nunique(dropna=True) <= 1:
            errors.append("表型目标列没有变异，无法训练预测模型")
        if genotype_missing:
            warnings.append(f"基因型包含 {genotype_missing} 个非数值或缺失值")

        duplicate_markers = int(genotype.columns.duplicated().sum())
        if duplicate_markers:
            errors.append(f"基因型包含 {duplicate_markers} 个重复 SNP 列名")
        malformed_markers = [str(name) for name in genotype.columns if "_" not in str(name)]
        if malformed_markers:
            warnings.append(f"有 {len(malformed_markers)} 个 SNP 列名不包含 '_'，染色体窗口模式可能无法使用")

        values = genotype_numeric.to_numpy()
        finite_values = values[np.isfinite(values)]
        if finite_values.size and not np.isin(finite_values, [-1, 0, 1]).all():
            warnings.append("检测到不是 -1/0/1 的基因型编码，请确认编码方式与模型配置一致")
        duplicate_samples = int(genotype_numeric.duplicated().sum())
        if duplicate_samples:
            warnings.append(f"检测到 {duplicate_samples} 个基因型完全相同的样本，请确认是否为重复材料")

        total_genotype_values = genotype.shape[0] * genotype.shape[1]
        missing_rate = genotype_missing / total_genotype_values if total_genotype_values else 0.0
        if task.split_strategy == "existing" and task.intent == TaskIntent.TRAIN_MODEL:
            split_dir = root / "split"
            for name in ("train_index.txt", "valid_index.txt", "test_index.txt"):
                path = split_dir / name
                if not path.is_file():
                    errors.append(f"缺少必要文件: {path}")
            split_paths = [split_dir / name for name in ("train_index.txt", "valid_index.txt", "test_index.txt")]
            if all(path.is_file() for path in split_paths):
                groups = [{line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()} for path in split_paths]
                overlap = (groups[0] & groups[1]) | (groups[0] & groups[2]) | (groups[1] & groups[2])
                if overlap:
                    errors.append(f"训练、验证和测试划分存在 {len(overlap)} 个重复样本，可能造成数据泄漏")

        data = {
            "dataset_dir": str(root),
            "trait": task.trait,
            "sample_count_genotype": int(len(genotype)),
            "sample_count_phenotype": int(len(phenotype)),
            "matched_sample_count": int(len(common_ids)),
            "snp_count": int(genotype.shape[1]),
            "genotype_missing_count": genotype_missing,
            "genotype_missing_rate": round(missing_rate, 8),
            "phenotype_missing_count": phenotype_missing,
            "duplicate_marker_count": duplicate_markers,
            "duplicate_genotype_sample_count": duplicate_samples,
        }
        return self._result(not errors, "validated" if not errors else "validation_failed", errors, warnings, data)

    def create_split(self, task: MenetTask) -> Dict[str, Any]:
        if task.split_strategy == "existing":
            return self._result(True, "skipped", [], [])
        if task.split_strategy != "random":
            return self._result(False, "not_implemented", ["当前仅支持 existing 和 random 数据划分"], [])

        root = Path(task.dataset_dir)
        phenotype_path = root / "phenotype" / f"{task.trait}.csv"
        if not phenotype_path.is_file():
            return self._result(False, "failed", [f"缺少表型文件: {phenotype_path}"], [])
        phenotype = pd.read_csv(phenotype_path, index_col=0, comment="*")
        ids = phenotype.index.astype(str).tolist()
        if len(ids) < 3:
            return self._result(False, "failed", ["自动划分至少需要 3 个样本"], [])

        rng = np.random.default_rng(int(task.metadata.get("split_seed", 42)))
        shuffled = np.asarray(ids, dtype=object)
        rng.shuffle(shuffled)
        train_count = max(1, int(round(len(ids) * task.train_ratio)))
        valid_count = max(1, int(round(len(ids) * task.valid_ratio)))
        if train_count + valid_count >= len(ids):
            valid_count = 1
            train_count = len(ids) - 2
        test_count = len(ids) - train_count - valid_count
        if test_count < 1:
            return self._result(False, "failed", ["数据量不足以生成非空的三份数据集"], [])

        working_root = Path(task.output_dir) / "input"
        working_root.mkdir(parents=True, exist_ok=True)
        for folder in ("genotype", "phenotype"):
            link = working_root / folder
            if not link.exists():
                link.symlink_to((root / folder).resolve(), target_is_directory=True)
        task.metadata["working_dataset_dir"] = str(working_root.resolve())
        split_dir = working_root / "split"
        split_dir.mkdir(parents=True, exist_ok=True)
        groups = {
            "train_index.txt": shuffled[:train_count],
            "valid_index.txt": shuffled[train_count:train_count + valid_count],
            "test_index.txt": shuffled[train_count + valid_count:],
        }
        for filename, group in groups.items():
            (split_dir / filename).write_text("\n".join(group.tolist()) + "\n", encoding="utf-8")

        data = {
            "seed": int(task.metadata.get("split_seed", 42)),
            "train_count": train_count,
            "valid_count": valid_count,
            "test_count": test_count,
            "files": [str(split_dir / filename) for filename in groups],
        }
        return self._result(True, "split_created", [], [], data)

    def train_trait_encoder(self, task: MenetTask) -> Dict[str, Any]:
        try:
            import torch
            from network.contrastive_learning import TraitSpecificEncoderForRepGeno
            from utils.dataset import create_triplet_dataloader
            from utils.loss import TripletLoss
            from utils.train import train_trait_specific_encoder
            from utils.relatedness import calculate_genetic_relatedness
            from utils.utils import get_phen_snp, set_seed

            config = self._load_config("configs/contrastive_learning.json", task)
            data, data_train, data_val, _ = get_phen_snp(config, task.trait)
            if len(data_train) < 2 or len(data_val) < 2:
                return self._result(False, "failed", ["训练性状特异编码器至少需要训练集和验证集各 2 个样本"], [])
            set_seed(int(task.metadata.get("seed", 42)))
            config["batch_size"] = min(int(config["batch_size"]), len(data_train))
            train_batch_size = config["batch_size"]
            drop_singleton = len(data_train) % config["batch_size"] == 1
            train_loader = create_triplet_dataloader(
                config, data_train, flag=config["flag"], shuffle=True, drop_last=drop_singleton
            )
            config["batch_size"] = min(int(config["batch_size"]), len(data_val))
            validation_batch_size = config["batch_size"]
            val_loader = create_triplet_dataloader(config, data_val, flag=config["flag"])
            self._write_training_snapshot(task, "trait_encoder", {
                **config,
                "train_batch_size": train_batch_size,
                "validation_batch_size": validation_batch_size,
            })
            model = TraitSpecificEncoderForRepGeno(
                snp_size=data_train.shape[-1] - 1,
                stride=config["stride"],
                out_dim=config["out_dim"],
            )
            criterion = TripletLoss(margin=config["margin"], flag=bool(config["flag"]))
            train_trait_specific_encoder(config, model, train_loader, val_loader, criterion)
            calculate_genetic_relatedness(model, data, config)
            relatedness = Path(config["gr_path"]) / "genetic_relatedness.pt"
            if not relatedness.is_file():
                return self._result(False, "failed", ["编码器训练完成，但未生成遗传相关性文件"], [])
            return self._result(True, "completed", [], [], {
                "sample_count": int(len(data)),
                "artifacts": [str(Path(config["model_path"]) / "trait_specific_encoder.pt"), str(relatedness)],
                "device": config["device"],
            })
        except Exception as exc:
            return self._result(False, "failed", [f"性状特异编码器训练失败: {exc}"], [])

    def build_relatedness(self, task: MenetTask) -> Dict[str, Any]:
        relatedness = Path(task.output_dir) / "genetic_relatedness.pt"
        if relatedness.is_file():
            return self._result(True, "completed", [], [], {"artifact": str(relatedness)})
        return self._not_implemented("遗传相关性文件不存在；请先完成性状特异编码器训练")

    def train_menet(self, task: MenetTask) -> Dict[str, Any]:
        try:
            from utils.dataset import create_dual_scale_dataloader, prepare_tensors
            from utils.loss import L1Loss
            from utils.train import train_menet
            from utils.utils import get_phen_gr, get_phen_snp, set_seed, windows_flag
            from network.menet import MeNet

            config = self._load_config("configs/MeNet.json", task)
            phen_snp, snp_train, snp_val, snp_test = get_phen_snp(config, task.trait)
            phen_gr, gr_train, gr_val, gr_test = get_phen_gr(config, task.trait)
            if len(snp_train) < 2 or len(snp_val) < 2 or len(snp_test) < 1:
                return self._result(False, "failed", ["MENET 至少需要非空的训练、验证和测试数据集"], [])
            set_seed(int(task.metadata.get("seed", 42)))
            config["batch_size"] = min(int(config["batch_size"]), len(snp_train))
            drop_singleton = len(snp_train) % config["batch_size"] == 1
            self._write_training_snapshot(task, "menet", config)
            train_loader = create_dual_scale_dataloader(
                snp_train, gr_train, config, shuffle=True, drop_last=drop_singleton
            )
            val_loader = create_dual_scale_dataloader(snp_val, gr_val, config)
            test_loader = create_dual_scale_dataloader(snp_test, gr_test, config)
            tensor_for_ig = prepare_tensors(phen_snp, phen_gr)
            windows = windows_flag(config, phen_snp)
            model = MeNet(phen_snp.shape[-1] - 1, phen_gr.shape[-1] - 1, config["param"], windows=windows)
            criterion = L1Loss()
            metrics = train_menet(config, model, train_loader, val_loader, test_loader, criterion, tensor_for_ig, windows=windows)
            data = metrics or {"artifacts": []}
            if task.explain_snp:
                data.setdefault("artifacts", []).append(self._write_snp_importance(
                    task, model, phen_snp, phen_gr, windows, config["device"]
                ))
            return self._result(True, "completed", [], [], data)
        except Exception as exc:
            return self._result(False, "failed", [f"MENET 训练失败: {exc}"], [])

    def predict_trait(self, task: MenetTask) -> Dict[str, Any]:
        try:
            import torch
            from network.menet import MeNet
            from utils.dataset import prepare_tensors
            from utils.utils import windows_flag

            config = self._load_config("configs/MeNet.json", task)
            source_dir = Path(task.source_output_dir or task.output_dir)
            model_path = source_dir / "menet_model.pt"
            relatedness_path = source_dir / "genetic_relatedness.pt"
            if not model_path.is_file():
                return self._result(False, "failed", [f"找不到 MENET 模型: {model_path}"], [])
            if not relatedness_path.is_file():
                return self._result(False, "failed", [f"找不到遗传相关性矩阵: {relatedness_path}"], [])

            prediction_path = task.prediction_genotype_path or task.metadata.get("prediction_genotype_path")
            genotype_path = Path(prediction_path) if prediction_path else Path(task.dataset_dir) / "genotype" / "genotype.csv"
            genotype = pd.read_csv(genotype_path, index_col=0, comment="*")
            training_genotype = pd.read_csv(Path(task.dataset_dir) / "genotype" / "genotype.csv", index_col=0, comment="*")
            # Normalize sample IDs because CSV parsing may infer numeric IDs while
            # the persisted relatedness matrix stores the same IDs as strings.
            genotype.index = genotype.index.astype(str)
            training_genotype.index = training_genotype.index.astype(str)
            if list(genotype.columns) != list(training_genotype.columns):
                return self._result(False, "failed", ["新基因型文件的 SNP 名称或顺序与训练数据不一致"], [])
            if genotype.index.has_duplicates:
                return self._result(False, "failed", ["新基因型文件包含重复样本 ID"], [])
            numeric_genotype = genotype.apply(pd.to_numeric, errors="coerce")
            if numeric_genotype.isna().any().any():
                return self._result(False, "failed", ["新基因型文件包含缺失或非数值编码"], [])
            relatedness = torch.load(relatedness_path, map_location=config["device"], weights_only=False)
            if not isinstance(relatedness, pd.DataFrame):
                relatedness = pd.DataFrame(relatedness)
            relatedness.index = relatedness.index.astype(str)
            relatedness.columns = relatedness.columns.astype(str)
            if prediction_path:
                from network.contrastive_learning import TraitSpecificEncoderForRepGeno
                encoder_config = self._load_config("configs/contrastive_learning.json", task)
                encoder_path = source_dir / "trait_specific_encoder.pt"
                if not encoder_path.is_file():
                    return self._result(False, "failed", [f"找不到性状特异编码器: {encoder_path}"], [])
                encoder = TraitSpecificEncoderForRepGeno(
                    snp_size=training_genotype.shape[1],
                    stride=encoder_config["stride"],
                    out_dim=encoder_config["out_dim"],
                )
                encoder.load_state_dict(torch.load(encoder_path, map_location=config["device"], weights_only=True))
                encoder.to(config["device"]).eval()
                train_tensor = torch.tensor(training_genotype.values, dtype=torch.float32).reshape(len(training_genotype), 1, -1).to(config["device"])
                new_tensor = torch.tensor(numeric_genotype.values, dtype=torch.float32).reshape(len(numeric_genotype), 1, -1).to(config["device"])
                with torch.no_grad():
                    train_rep = encoder.forward_once(train_tensor)
                    new_rep = encoder.forward_once(new_tensor)
                train_rep = train_rep / train_rep.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                new_rep = new_rep / new_rep.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                distances = torch.cdist(new_rep, train_rep, p=2)
                scale = max(float(torch.cdist(train_rep, train_rep, p=2).max().item()), 1e-12)
                cross = (1 - distances / scale).clamp(min=-1, max=1).cpu().numpy()
                common = genotype.index
                relatedness = pd.DataFrame(cross, index=common, columns=training_genotype.index)
            else:
                common = genotype.index.intersection(relatedness.index)
                if len(common) != len(genotype):
                    return self._result(False, "failed", ["预测样本无法在遗传相关性矩阵中全部匹配"], [])
                genotype = genotype.loc[common]
                relatedness = relatedness.loc[common, common]
            phen_snp = genotype.copy()
            phen_snp.insert(0, task.trait, 0.0)
            phen_gr = relatedness.copy()
            phen_gr.insert(0, task.trait, 0.0)
            windows = windows_flag(config, phen_snp)
            model = MeNet(phen_snp.shape[-1] - 1, phen_gr.shape[-1] - 1, config["param"], windows=windows)
            model.load_state_dict(torch.load(model_path, map_location=config["device"], weights_only=True))
            model.to(config["device"]).eval()
            snp_tensor, gr_tensor, _ = prepare_tensors(phen_snp, phen_gr)
            with torch.no_grad():
                menet_predictions = model(snp_tensor.to(config["device"]), gr_tensor.to(config["device"]))
            menet_predictions = menet_predictions.flatten().cpu().numpy()
            metrics = json.loads((source_dir / "metrics.json").read_text(encoding="utf-8"))
            selection = task.metadata.get("prediction_selection") or self._select_prediction_method(metrics)
            if selection.get("primary_method") not in {"menet", "genomic_ridge", "random_forest", "xgboost"}:
                selection = self._select_prediction_method(metrics)
            if selection.get("primary_method") != "menet":
                selected_metrics = metrics.get("baselines", {}).get(selection["primary_method"], {})
                if not isinstance(selected_metrics, dict) or "test_r2" not in selected_metrics:
                    selection = self._select_prediction_method(metrics)
            primary_method = selection["primary_method"]
            output = {"sample_id": common.astype(str)}
            if primary_method == "menet":
                output["recommended_menet_prediction"] = menet_predictions
            else:
                baseline_path = source_dir / f"{primary_method}_model.joblib"
                if not baseline_path.is_file():
                    self._rebuild_baseline_model(task, source_dir, primary_method, baseline_path)
                import joblib
                baseline_model = joblib.load(baseline_path)
                features = genotype.loc[common].apply(pd.to_numeric, errors="raise").to_numpy()
                output[f"recommended_{primary_method}_prediction"] = baseline_model.predict(features)
                output["reference_menet_prediction"] = menet_predictions
            output_path = Path(task.output_dir) / "predictions.csv"
            pd.DataFrame(output).to_csv(output_path, index=False)
            return self._result(True, "completed", [], [], {
                "sample_count": len(common),
                "input_type": "new_genotype" if prediction_path else "current_dataset",
                "prediction_selection": selection,
                "artifacts": [str(output_path)],
            })
        except Exception as exc:
            return self._result(False, "failed", [f"表型预测失败: {exc}"], [])

    @staticmethod
    def _select_prediction_method(metrics: Dict[str, Any], threshold: float = 0.05) -> Dict[str, Any]:
        menet_r2 = metrics.get("test_r2")
        candidates = {"menet": float(menet_r2)} if isinstance(menet_r2, (int, float)) else {}
        for method, item in (metrics.get("baselines") or {}).items():
            if isinstance(item, dict) and isinstance(item.get("test_r2"), (int, float)):
                candidates[method] = float(item["test_r2"])
        if "menet" not in candidates:
            return {"primary_method": "menet", "reason": "missing_comparable_metrics", "candidates": candidates,
                    "minimum_r2_gain": threshold}
        best_method, best_r2 = max(candidates.items(), key=lambda item: item[1])
        gain = best_r2 - candidates["menet"]
        primary_method = best_method if best_method != "menet" and gain > threshold else "menet"
        return {
            "primary_method": primary_method,
            "primary_test_r2": candidates[primary_method],
            "menet_test_r2": candidates["menet"],
            "gain_over_menet": candidates[primary_method] - candidates["menet"],
            "minimum_r2_gain": threshold,
            "candidates": candidates,
            "reason": "baseline_significantly_better" if primary_method != "menet" else "menet_or_no_clear_winner",
        }

    @staticmethod
    def _rebuild_baseline_model(task: MenetTask, source_dir: Path, method: str, target_path: Path) -> None:
        import joblib
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge

        training_root = source_dir / "input"
        if not (training_root / "split" / "train_index.txt").is_file():
            training_root = Path(task.dataset_dir)
        genotype = pd.read_csv(training_root / "genotype" / "genotype.csv", index_col=0, comment="*")
        phenotype = pd.read_csv(training_root / "phenotype" / f"{task.trait}.csv", index_col=0, comment="*")
        train_ids = [line.strip() for line in (training_root / "split" / "train_index.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
        train_ids = [sample_id for sample_id in train_ids if sample_id in genotype.index and sample_id in phenotype.index]
        train_x = genotype.loc[train_ids].apply(pd.to_numeric, errors="raise").to_numpy()
        train_y = phenotype.loc[train_ids].iloc[:, 0].apply(pd.to_numeric, errors="raise").to_numpy()
        snapshot_path = source_dir / "training_config.json"
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8")) if snapshot_path.is_file() else {}
        seed = int(snapshot.get("training_seed", 42))
        if method == "genomic_ridge":
            model = Ridge(alpha=1.0, solver="lsqr", max_iter=1000)
        elif method == "random_forest":
            model = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1, max_features="sqrt")
        elif method == "xgboost":
            from xgboost import XGBRegressor
            model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8,
                                 colsample_bytree=0.8, objective="reg:squarederror", random_state=seed,
                                 n_jobs=-1, tree_method="hist")
        else:
            raise ValueError(f"不支持的预测方法: {method}")
        model.fit(train_x, train_y)
        joblib.dump(model, target_path)

    def evaluate_model(self, task: MenetTask) -> Dict[str, Any]:
        metrics_path = Path(task.source_output_dir or task.output_dir) / "metrics.json"
        if not metrics_path.is_file():
            return self._result(False, "failed", [f"找不到模型评估结果: {metrics_path}"], [])
        try:
            return self._result(True, "completed", [], [], json.loads(metrics_path.read_text(encoding="utf-8")))
        except Exception as exc:
            return self._result(False, "failed", [f"读取模型评估结果失败: {exc}"], [])

    def explain_model(self, task: MenetTask) -> Dict[str, Any]:
        history_path = Path(task.source_output_dir or task.output_dir) / "training_history.json"
        if not history_path.is_file():
            return self._result(False, "failed", [f"找不到训练历史: {history_path}"], [])
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
            latest = history[-1] if history else {}
            return self._result(True, "completed", [], [], {
                "ve_contribution": latest.get("ig_ve"),
                "repgeno_contribution": latest.get("ig_repgeno"),
                "artifacts": [str(history_path)],
            })
        except Exception as exc:
            return self._result(False, "failed", [f"读取模型解释结果失败: {exc}"], [])

    @staticmethod
    def _write_training_snapshot(task: MenetTask, component: str, config: Dict[str, Any]) -> str:
        path = Path(task.output_dir) / "training_config.json"
        try:
            snapshot = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            snapshot = {}
        snapshot.update({
            "schema_version": 1,
            "training_mode": task.metadata.get("training_mode", "recommended"),
            "trait": task.trait,
            "device_requested": task.device,
            "split_strategy": task.split_strategy,
            "split_ratios": {"train": task.train_ratio, "validation": task.valid_ratio, "test": task.test_ratio},
            "split_seed": int(task.metadata.get("split_seed", 42)),
            "training_seed": int(task.metadata.get("seed", 42)),
            "explain_snp": task.explain_snp,
        })
        snapshot.setdefault("components", {})[component] = json.loads(json.dumps(config))
        path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    @staticmethod
    def _load_config(filename: str, task: MenetTask) -> Dict[str, Any]:
        with (CORE_ROOT / filename).open(encoding="utf-8") as handle:
            config = json.load(handle)
        output = Path(task.output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        config["root_path"] = str(Path(task.metadata.get("working_dataset_dir", task.dataset_dir)).resolve())
        config["model_path"] = str(output)
        config["gr_path"] = str(output)
        config["phen_name"] = task.trait
        config["device"] = MenetTools._resolve_device(task.device)
        config["windows"] = int(task.metadata.get("windows", 0))
        if task.epochs is not None:
            config["epoch"] = task.epochs
        return config

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("请求使用 CUDA，但当前环境没有可用 GPU")
        return device

    @staticmethod
    def _write_snp_importance(task: MenetTask, model, phen_snp, phen_gr, windows, device: str) -> str:
        import torch

        model_path = Path(task.output_dir) / "menet_model.pt"
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        snp_tensor, gr_tensor, _ = __import__("utils.dataset", fromlist=["prepare_tensors"]).prepare_tensors(phen_snp, phen_gr)
        snp_tensor = snp_tensor.to(device).requires_grad_(True)
        gr_tensor = gr_tensor.to(device)
        prediction = model(snp_tensor, gr_tensor).sum()
        gradient = torch.autograd.grad(prediction, snp_tensor)[0]
        importance = (gradient * snp_tensor).abs().mean(dim=(0, 1)).detach().cpu().numpy()
        output = Path(task.output_dir) / "snp_importance.csv"
        pd.DataFrame({"snp": list(phen_snp.columns[1:]), "importance": importance}).sort_values(
            "importance", ascending=False
        ).to_csv(output, index=False)
        return str(output)

    @staticmethod
    def _result(success: bool, status: str, errors: List[str], warnings: List[str], data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "success": success,
            "status": status,
            "data": data or {},
            "errors": errors,
            "warnings": warnings,
        }

    @classmethod
    def _not_implemented(cls, message: str) -> Dict[str, Any]:
        return cls._result(False, "not_implemented", [message], [])

import torch
import copy
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import json
from pathlib import Path
from utils.ig import ig_analysis


def _write_progress(config, phase, current_epoch, total_epochs, metrics):
    output = Path(config.get("model_path", "save"))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "progress.json"
    temporary = output / "progress.json.tmp"
    payload = {
        "phase": phase,
        "current_epoch": current_epoch,
        "total_epochs": total_epochs,
        "percent": round(current_epoch / total_epochs * 100, 1),
        "metrics": metrics,
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    temporary.write_text(json.dumps(payload, ensure_ascii=False, default=float), encoding="utf-8")
    temporary.replace(path)


def _raise_if_cancelled(config):
    if (Path(config.get("model_path", "save")) / "cancel.requested").is_file():
        raise InterruptedError("任务已由用户取消")

def setup_training_env(config, model, criterion):
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 0.01))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)
    return device, optimizer, scheduler

def forward_triplet_loss_step(model, criterion, batch, device):
    batch = [x.to(device) for x in batch]
    anchor, anchor_phen, pos, pos_phen, neg, neg_phen = batch
    anchor_out, pos_out, neg_out = model(anchor, pos, neg)
    return criterion(anchor_out, anchor_phen, pos_out, pos_phen, neg_out, neg_phen)

def train_trait_specific_encoder_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    for batch_train in train_loader:
        optimizer.zero_grad()
        loss = forward_triplet_loss_step(model, criterion, batch_train, device)
        loss.backward()
        optimizer.step()

def evaluate_trait_specific_encoder(model, dataloader, criterion, device):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch_val in dataloader:
            loss = forward_triplet_loss_step(model, criterion, batch_val, device)
            val_loss += loss.item()
        val_loss /= len(dataloader)
    return val_loss


def train_trait_specific_encoder(config, model, train_loader, val_loader, criterion):
    device, optimizer, scheduler = setup_training_env(config, model, criterion)
    best_val_loss=float('inf')
    best_model = copy.deepcopy(model)
    for epoch in range(config['epoch']):
        _raise_if_cancelled(config)
        train_trait_specific_encoder_one_epoch(model, train_loader, optimizer, criterion, device)
        train_loss = evaluate_trait_specific_encoder(model, train_loader, criterion, device)
        val_loss = evaluate_trait_specific_encoder(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(f"[Epoch {epoch + 1:03d}] "
              f"Train Loss: {train_loss:.4f} "
              f"Val Loss: {val_loss:.4f} ")
        _write_progress(config, "training_encoder", epoch + 1, config["epoch"], {
            "train_loss": train_loss, "val_loss": val_loss,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO] Trait-specific encoder trained successfully. "
          f"Saved at: {config['model_path']}/trait_specific_encoder.pt")
    torch.save(best_model.state_dict(), f"{config['model_path']}/trait_specific_encoder.pt")
    return best_model


def train_menet_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    for x1, x2, y in dataloader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x1, x2)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()


def evaluate_menet(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    y_preds, y_trues = [], []

    with torch.no_grad():
        for x1, x2, y in dataloader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            y_pred = model(x1, x2)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item() * x1.size(0)
            y_preds.extend(y_pred.detach().cpu().numpy())
            y_trues.extend(y.detach().cpu().numpy())
    y_preds = np.array(y_preds).flatten()
    y_trues = np.array(y_trues).flatten()
    avg_loss = total_loss / len(dataloader.dataset)
    r2 = r2_score(y_trues, y_preds)
    return avg_loss, r2, y_preds, y_trues


def _baseline_arrays(train_loader, test_loader):
    train_x, _, train_y = train_loader.dataset.tensors
    test_x, _, test_y = test_loader.dataset.tensors
    return (
        train_x.detach().cpu().numpy().reshape(len(train_x), -1),
        test_x.detach().cpu().numpy().reshape(len(test_x), -1),
        train_y.detach().cpu().numpy().reshape(-1),
        test_y.detach().cpu().numpy().reshape(-1),
    )


def _baseline_metrics(method, predictions, truths, train_count, test_count, **params):
    return {
        "method": method,
        **params,
        "test_r2": float(r2_score(truths, predictions)),
        "test_mae": float(mean_absolute_error(truths, predictions)),
        "train_sample_count": int(train_count),
        "test_sample_count": int(test_count),
    }


def evaluate_ridge_baseline(train_loader, test_loader, alpha=1.0):
    """Fit a deterministic genomic ridge baseline on the same split as MENET."""
    train_x, test_x, train_y, test_y = _baseline_arrays(train_loader, test_loader)
    model = Ridge(alpha=alpha, solver="lsqr", max_iter=1000)
    model.fit(train_x, train_y)
    return _baseline_metrics("genomic_ridge", model.predict(test_x), test_y, len(train_y), len(test_y), alpha=float(alpha))


def evaluate_random_forest_baseline(train_loader, test_loader, random_state=42):
    """Fit a bounded random forest baseline on the same split as MENET."""
    train_x, test_x, train_y, test_y = _baseline_arrays(train_loader, test_loader)
    model = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1, max_features="sqrt")
    model.fit(train_x, train_y)
    return _baseline_metrics("random_forest", model.predict(test_x), test_y, len(train_y), len(test_y), n_estimators=300, random_state=random_state)


def evaluate_xgboost_baseline(train_loader, test_loader, random_state=42):
    """Fit XGBoost when the optional dependency is installed."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return {"method": "xgboost", "status": "unavailable", "error": "未安装 xgboost"}
    train_x, test_x, train_y, test_y = _baseline_arrays(train_loader, test_loader)
    model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03, subsample=0.8,
                         colsample_bytree=0.8, objective="reg:squarederror", random_state=random_state,
                         n_jobs=-1, tree_method="hist")
    model.fit(train_x, train_y, verbose=False)
    return _baseline_metrics("xgboost", model.predict(test_x), test_y, len(train_y), len(test_y),
                             n_estimators=300, max_depth=4, learning_rate=0.03, random_state=random_state)


def evaluate_baselines(train_loader, test_loader, random_state=42):
    return {
        "genomic_ridge": evaluate_ridge_baseline(train_loader, test_loader),
        "random_forest": evaluate_random_forest_baseline(train_loader, test_loader, random_state),
        "xgboost": evaluate_xgboost_baseline(train_loader, test_loader, random_state),
    }

def train_menet(config, model, train_loader, val_loader, test_loader, criterion, tensor_for_ig, windows=None):
    device, optimizer, scheduler = setup_training_env(config, model, criterion)
    best_val_r2=float('-inf')
    best_model = copy.deepcopy(model)
    history = []
    for epoch in range(config['epoch']):
        _raise_if_cancelled(config)
        train_menet_one_epoch(model, train_loader, optimizer, criterion, device)
        train_loss, train_r2, _, _ = evaluate_menet(model, train_loader, criterion, device)
        val_loss, val_r2, _, _ = evaluate_menet(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        ig_ve, ig_repgeno = ig_analysis(model, tensor_for_ig[0], tensor_for_ig[1], device, windows)
        print(f"train_loss = {train_loss:.4f}, train_r2 = {train_r2:.4f}, "
              f"val_loss = {val_loss:.4f}, val_r2 = {val_r2:.4f}, "
              f"ig_VE = {ig_ve:.4f}, ig_RepGeno={ig_repgeno:.4f}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_r2": train_r2,
                        "val_loss": val_loss, "val_r2": val_r2, "ig_ve": ig_ve,
                        "ig_repgeno": ig_repgeno})
        _write_progress(config, "training_menet", epoch + 1, config["epoch"], history[-1])

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model = copy.deepcopy(model)
    test_loss, test_r2, _, _ = evaluate_menet(best_model, test_loader, criterion, device)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [INFO] Best model achieved R² = {test_r2:.4f} on the test set.")
    output_dir = Path(config.get("model_path", "save"))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "menet_model.pt"
    history_path = output_dir / "training_history.json"
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "test_predictions.csv"
    torch.save(best_model.state_dict(), model_path)
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    _, _, predictions, truths = evaluate_menet(best_model, test_loader, criterion, device)
    np.savetxt(predictions_path, np.column_stack((truths, predictions)), delimiter=",",
               header="true,predicted", comments="")
    try:
        baselines = evaluate_baselines(train_loader, test_loader)
    except Exception as exc:
        baselines = {"status": "failed", "error": str(exc)}
    ridge = baselines.get("genomic_ridge", {}) if isinstance(baselines, dict) else {}
    metrics = {"test_loss": float(test_loss), "test_r2": float(test_r2),
               "baseline": ridge,
               "baselines": baselines,
               "r2_gain_vs_baseline": (
                   float(test_r2 - ridge["test_r2"]) if "test_r2" in ridge else None
               ),
               "device": str(device), "epochs": int(config["epoch"]),
               "artifacts": [str(model_path), str(history_path), str(metrics_path), str(predictions_path)]}
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return metrics

if __name__ == '__main__':
    pass

import torch
import copy
from datetime import datetime
from sklearn.metrics import r2_score
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
    metrics = {"test_loss": float(test_loss), "test_r2": float(test_r2),
               "device": str(device), "epochs": int(config["epoch"]),
               "artifacts": [str(model_path), str(history_path), str(metrics_path), str(predictions_path)]}
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return metrics

if __name__ == '__main__':
    pass

import os
import time
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from src.data import get_dataloaders, DataConfig
from src.model import SmallCNN
from src.utils import set_seed, ensure_dir

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_optimizer(name: str, params, lr: float, wd: float):
    if name.lower() == "adam":
        return Adam(params, lr=lr, weight_decay=wd)
    if name.lower() == "sgd":
        return SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    return Adam(params, lr=lr, weight_decay=wd)

def build_scheduler(name: str, optimizer, epochs: int):
    if name.lower() == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    return None

def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()

def train_one_epoch(model, loader: DataLoader, optimizer, scaler, device, criterion):
    model.train()
    loss_sum = 0.0
    acc_sum = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bsz = x.size(0)
        loss_sum += loss.item() * bsz
        acc_sum += accuracy(logits.detach(), y) * bsz
        n += bsz
    return loss_sum / n, acc_sum / n

@torch.no_grad()
def validate(model, loader: DataLoader, device, criterion):
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        bsz = x.size(0)
        loss_sum += loss.item() * bsz
        acc_sum += accuracy(logits, y) * bsz
        n += bsz
    return loss_sum / n, acc_sum / n

def main():
    cfg = load_cfg("configs/default.yaml")
    set_seed(int(cfg.get("seed", 1337)))

    dcfg = DataConfig(
        root=cfg["data"]["root"],
        split=cfg["data"]["split"],
        subset=cfg["data"]["subset"],
        batch_size=int(cfg["data"]["batch_size"]),
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        drop_last=bool(cfg["data"]["drop_last"]),
        train_val_split=float(cfg["data"]["train_val_split"]),
    )
    train_loader, val_loader, num_classes, classes = get_dataloaders(dcfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNN(num_classes=num_classes).to(device)

    epochs = int(cfg["train"]["epochs"])
    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"]["weight_decay"])
    optim_name = str(cfg["train"]["optimizer"])
    scheduler_name = str(cfg["train"]["scheduler"])
    label_smoothing = float(cfg["train"].get("label_smoothing", 0.0))

    optimizer = build_optimizer(optim_name, model.parameters(), lr, wd)
    scheduler = build_scheduler(scheduler_name, optimizer, epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    out_dir = os.path.join("data", "models")
    ensure_dir(out_dir)
    metrics_path = os.path.join("data", "metrics.csv")
    best_path = os.path.join(out_dir, "best.pt")
    last_path = os.path.join(out_dir, "last.pt")

    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr\n")

    best_val = -1.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion)
        val_loss, val_acc = validate(model, val_loader, device, criterion)
        if scheduler is not None:
            scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},{lr_now:.8f}\n")
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc, "classes": classes}, last_path)
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc, "classes": classes}, best_path)
        dt = time.time() - t0
        print(f"epoch {epoch}/{epochs} train_loss {train_loss:.4f} train_acc {train_acc:.4f} val_loss {val_loss:.4f} val_acc"
              f" {val_acc:.4f} lr {lr_now:.6f} time {dt:.1f}s")

        test = ("test czy dziala zwijanie "
                "wierszy podczas pisanie ")

if __name__ == "__main__":
    main()

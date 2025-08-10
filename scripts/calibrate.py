import os
import json
import yaml
import torch
import torch.nn as nn
from src.data import get_dataloaders, DataConfig
from src.model import SmallCNN
from src.utils import set_seed, ensure_dir

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@torch.no_grad()
def eval_nll(model, loader, device, T):
    model.eval()
    nll = 0.0
    n = 0
    crit = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x) / T
        nll += crit(logits, y).item()
        n += x.size(0)
    return nll / n

def main():
    cfg = load_cfg("configs/default.yaml")
    set_seed(int(cfg.get("seed", 1337)))
    dcfg = DataConfig(
        root=cfg["data"]["root"],
        split=cfg["data"]["split"],
        subset=cfg["data"]["subset"],
        batch_size=512,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=False,
        drop_last=False,
        train_val_split=float(cfg["data"]["train_val_split"]),
    )
    _, val_loader, _, _ = get_dataloaders(dcfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join("data", "models", "best.pt")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        sd = state["model"]
        classes = state.get("classes", [str(i) for i in range(10)])
    else:
        sd = state
        classes = [str(i) for i in range(10)]
    model = SmallCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(sd, strict=True)
    T = torch.tensor([1.0], device=device, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")
    crit = nn.CrossEntropyLoss(reduction="mean")

    def closure():
        opt.zero_grad()
        loss_sum = 0.0
        m = 0
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x) / T
            loss = crit(logits, y)
            loss_sum += loss
            m += 1
        loss_avg = loss_sum / max(1, m)
        loss_avg.backward()
        return loss_avg

    opt.step(closure)
    T_value = float(T.detach().clamp(min=1e-3).item())
    out_dir = os.path.join("data", "models")
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "calib.json"), "w", encoding="utf-8") as f:
        json.dump({"temperature": T_value}, f)
    nll_before = eval_nll(model, val_loader, device, 1.0)
    nll_after = eval_nll(model, val_loader, device, T_value)
    print(f"temperature: {T_value:.4f}")
    print(f"nll_before: {nll_before:.6f}")
    print(f"nll_after: {nll_after:.6f}")

if __name__ == "__main__":
    main()

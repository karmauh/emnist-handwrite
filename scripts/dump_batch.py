import os
from torchvision.utils import save_image
from src.data import get_dataloaders, DataConfig

def inv_norm_batch(x):
    mean = 0.1307
    std = 0.3081
    return x * std + mean

cfg = DataConfig(root="data", split="balanced", subset="digits", batch_size=64, num_workers=0, pin_memory=False, drop_last=False, train_val_split=0.9)
train_loader, _, _, _ = get_dataloaders(cfg)
x, y = next(iter(train_loader))
x = inv_norm_batch(x[:64])
os.makedirs("data/debug", exist_ok=True)
save_image(x, "data/debug/batch_grid.png", nrow=8, padding=2)
print("saved: data/debug/batch_grid.png")

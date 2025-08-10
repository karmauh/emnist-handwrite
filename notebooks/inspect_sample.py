import torch
from torchvision.utils import make_grid
from torchvision import transforms
import matplotlib.pyplot as plt
from src.data import get_dataloaders, DataConfig

def inv_norm(x):
    mean = 0.1307
    std = 0.3081
    return x * std + mean

cfg = DataConfig(root="data", split="balanced", subset="digits", batch_size=32, num_workers=0, pin_memory=False, drop_last=False, train_val_split=0.9)
train_loader, val_loader, n_cls, classes = get_dataloaders(cfg)
x, y = next(iter(train_loader))
x_show = inv_norm(x[:16])
grid = make_grid(x_show, nrow=8, padding=2)
plt.figure(figsize=(6,6))
plt.axis("off")
plt.imshow(grid.squeeze().numpy().transpose(1,2,0), cmap="gray")
plt.title(str([int(t.item()) for t in y[:16]]))
plt.show()

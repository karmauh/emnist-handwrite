import collections
from torch.utils.data import DataLoader
from src.data import get_dataloaders, DataConfig

cfg = DataConfig(root="data", split="balanced", subset="digits", batch_size=512, num_workers=0, pin_memory=False, drop_last=False, train_val_split=0.9)
train_loader, val_loader, n_cls, classes = get_dataloaders(cfg)

# x, y = next(iter(train_loader))
# print(x.shape, x.min().item(), x.max().item())
# print(y.tolist())

def count(loader):
    c = collections.Counter()
    for _, y in loader:
        for t in y.tolist():
            c[t] += 1
    return c

ct_train = count(train_loader)
ct_val = count(val_loader)
print("train:", dict(sorted(ct_train.items())))
print("val:", dict(sorted(ct_val.items())))
print("classes:", classes)

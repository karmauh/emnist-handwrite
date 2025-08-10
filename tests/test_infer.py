from types import SimpleNamespace
from src.data import get_dataloaders, DataConfig

def test_dataloaders_build():
    cfg = DataConfig(root="data", split="balanced", subset="digits", batch_size=32, num_workers=0, pin_memory=False, drop_last=False, train_val_split=0.9)
    train_loader, val_loader, n_cls, classes = get_dataloaders(cfg)
    x, y = next(iter(train_loader))
    assert x.shape[1:] == (1, 28, 28)
    assert n_cls in (10, 47)
    assert len(classes) in (10, 47)

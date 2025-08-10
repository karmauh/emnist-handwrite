from dataclasses import dataclass
from typing import Tuple, List, Optional
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

@dataclass
class DataConfig:
    root: str
    split: str
    subset: str
    batch_size: int
    num_workers: int
    pin_memory: bool
    drop_last: bool
    train_val_split: float

def _emnist_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    mean = (0.1307,)
    std = (0.3081,)
    train_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, test_tf

def _filter_digits(dataset: datasets.EMNIST) -> Subset:
    targets = dataset.targets

    if isinstance(targets, list):
        targets = torch.tensor(targets, dtype=torch.long)
    idx = torch.where(targets < 10)[0]
    return Subset(dataset, idx.tolist())

def _num_classes(subset: str) -> int:
    return 10 if subset == "digits" else 47

def get_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader, int, List[str]]:
    train_tf, test_tf = _emnist_transforms()
    train_set = datasets.EMNIST(root=cfg.root, split=cfg.split, train=True, download=True, transform=train_tf)
    test_set = datasets.EMNIST(root=cfg.root, split=cfg.split, train=False, download=True, transform=test_tf)
    classes = list(train_set.classes) if hasattr(train_set, "classes") else []

    if cfg.subset == "digits":
        train_set = _filter_digits(train_set)
        test_set = _filter_digits(test_set)
        classes = [str(i) for i in range(10)]
    n_total = len(train_set) if isinstance(train_set, Subset) else len(train_set.data)
    n_train = int(n_total * cfg.train_val_split)
    n_val = n_total - n_train

    if isinstance(train_set, Subset):
        base = train_set
        generator = torch.Generator().manual_seed(1337)
        train_subset, val_subset = random_split(base, [n_train, n_val], generator=generator)
    else:
        generator = torch.Generator().manual_seed(1337)
        train_subset, val_subset = random_split(train_set, [n_train, n_val], generator=generator)
    train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=cfg.drop_last)
    val_loader = DataLoader(val_subset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, drop_last=False)
    return train_loader, val_loader, _num_classes(cfg.subset), classes

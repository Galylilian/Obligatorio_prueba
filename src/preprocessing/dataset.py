import os

from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets

from src.preprocessing.eda_stats import BINARY_CLASS_NAMES
from src.preprocessing.transforms import get_eval_transforms, get_train_transforms
from src.settings.config import settings


def is_valid_file(path):
    return path.lower().endswith((".jpg", ".jpeg", ".png"))


def _split_dir(split: str) -> str:
    return os.path.join(settings.data_dir, split)


def get_dataloaders(batch_size: int | None = None):
    """Crea dataloaders para train, valid y test."""
    batch_size = batch_size or settings.batch_size

    train_dir = _split_dir("train")
    valid_dir = _split_dir("valid")
    test_dir = _split_dir("test")

    train_data = datasets.ImageFolder(
        train_dir,
        transform=get_train_transforms(),
        is_valid_file=is_valid_file,
    )
    valid_data = datasets.ImageFolder(
        valid_dir,
        transform=get_eval_transforms(),
        is_valid_file=is_valid_file,
    )
    test_data = datasets.ImageFolder(
        test_dir,
        transform=get_eval_transforms(),
        is_valid_file=is_valid_file,
    )

    sample_weights = [
        1.0 / train_data.targets.count(target)
        for target in train_data.targets
    ]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=sampler,
    )
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader, train_data.classes

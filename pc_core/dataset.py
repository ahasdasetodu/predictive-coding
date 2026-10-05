import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

DATA_DIR = "./data"
_MEAN = 0.1307
_STD = 0.3081


def _flat_split(train: bool) -> TensorDataset:
    """Flatten and normalize one MNIST split in one pass.

    Same values as ToTensor + Normalize + flatten, applied once to the
    whole split. MNIST fits in memory; a per-image Python transform was
    most of the loader time.
    """
    raw = datasets.MNIST(root=DATA_DIR, train=train, download=True)
    images = raw.data.float().div_(255.0)
    images.sub_(_MEAN).div_(_STD)
    images = images.flatten(1)
    labels = raw.targets.long()
    return TensorDataset(images, labels)


def load_dataset(
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Return normalized, flattened MNIST train and test loaders."""
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        _flat_split(True),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        _flat_split(False),
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=pin,
    )
    return train_loader, test_loader

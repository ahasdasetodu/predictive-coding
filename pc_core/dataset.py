from torch import flatten
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = "./data"
_MNIST_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: flatten(x))
    ]
)

def load_dataset(batch_size=32,shuffle=True,drop_last=True) :

    train_data= datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=_MNIST_TRANSFORM
    )
    test_data = datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform= _MNIST_TRANSFORM
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=0,
    )
    return train_loader, test_loader



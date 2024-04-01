import torch
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

degrees = [0, 270]

def get_mnist(data_path: str = "./data"):
    """Download MNIST and apply minimal transformation."""

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.RandomRotation(degrees)]
    )

    trainset = CIFAR10(data_path, train=True, download=True, transform=transform)
    testset = CIFAR10(data_path, train=False, download=True, transform=transform)

    return trainset, testset


def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
    """Download MNIST and generate IID partitions."""

    # download MNIST in case it's not already in the system
    trainset, testset = get_mnist()

    # split trainset into `num_partitions` trainsets (one per client)
    num_images = len(trainset) // num_partitions

    # a list of partition lenghts (all partitions are of equal size)
    partition_len = [num_images] * num_partitions

    trainsets = random_split(
        trainset, partition_len, torch.Generator().manual_seed(2023)
    )

    # create dataloaders with train+val support
    trainloaders = []
    valloaders = []
    # for each train set, let's put aside some training examples for validation
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)
        )

        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
        )

    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader

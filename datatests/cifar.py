import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

degrees = [0, 270]

def load_data(user_size, devices_per_user):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.RandomRotation(degrees)]
    )
    trainset = CIFAR10("./data", train=True, download=True, transform=transform)
    testset = CIFAR10("./data", train=False, download=True, transform=transform)

    trainloaders = []
    testloaders = []

    start = 0
    devices_count = user_size * devices_per_user
    dataset_size = len(trainset) // devices_count

    for i in range(1, devices_count+1):
        indices = list(range(start, start+dataset_size))
        start = start + dataset_size
        subset = Subset(trainset, indices)
        trainloader = DataLoader(subset, batch_size=32, shuffle=True)
        trainloaders.append(trainloader)

    start = 0
    dataset_size = len(testset) // devices_count

    for i in range(1, devices_count+1):
        indices = list(range(start, start+dataset_size))
        start = start + dataset_size
        subset = Subset(testset, indices)
        testloader = DataLoader(subset, batch_size=32, shuffle=True)
        testloaders.append(testloader)

    dataset = []

    for i in range(1, devices_count + 1):
        dataset.append({
            'user_id': i % user_size,
            'train': trainloaders[i-1],
            'test': testloaders[i-1]
        })

    return dataset

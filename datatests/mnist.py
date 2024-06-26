import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST

degrees = [0, 270]

def load_data(user_size, devices_per_user):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,)),
         transforms.RandomRotation(degrees)]
    )
    trainset = MNIST("./data", train=True, download=True, transform=transform)
    testset = MNIST("./data", train=False, download=True, transform=transform)

    trainloaders = []
    testloaders = []

    start = 0
    dataset_size = len(trainset) // (user_size * devices_per_user)

    for i in range(1, user_size*devices_per_user+1):
        indices = list(range(start, start+dataset_size))
        start = start + dataset_size
        subset = Subset(trainset, indices)
        trainloader = DataLoader(subset, batch_size=32, shuffle=True)
        trainloaders.append(trainloader)

    start = 0
    dataset_size = len(testset) // (user_size * devices_per_user)

    for i in range(1, user_size*devices_per_user+1):
        indices = list(range(start, start+dataset_size))
        start = start + dataset_size
        subset = Subset(testset, indices)
        testloader = DataLoader(subset, batch_size=32, shuffle=True)
        testloaders.append(testloader)

    dataset = []

    for i in range(1, user_size*devices_per_user + 1):
        dataset.append({
            'user_id': i % user_size,
            'train': trainloaders[i-1],
            'test': testloaders[i-1]
        })

    return dataset

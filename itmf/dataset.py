import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def setup_dataset(dataset_name, batch_size, data_dir="./data"):
    if dataset_name in ["MNIST", "F-MNIST"]:
        input_dim = 28 * 28
        num_classes = 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        
        if dataset_name == "MNIST":
            train_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
            test_set = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        else:
            train_set = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
            test_set = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    elif dataset_name in ["CIFAR10", "CIFAR100"]:
        input_dim = 3 * 32 * 32
        num_classes = 10 if dataset_name == "CIFAR10" else 100
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        
        if dataset_name == "CIFAR10":
            train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
            test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        else:
            train_set = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
            test_set = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)

    return train_loader, test_loader, input_dim, num_classes
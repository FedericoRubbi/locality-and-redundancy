import shutil
import urllib.request
import zipfile
from pathlib import Path

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

TINY_IMAGENET_MEAN = (0.4802, 0.4481, 0.3975)
TINY_IMAGENET_STD = (0.2770, 0.2691, 0.2821)

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


class MultiViewTransformCIFAR:
    def __init__(self, size=32):
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class MultiViewTransformMNIST:
    def __init__(self, size=32):
        self.transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomRotation(15),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class MultiViewTransformTinyImageNet:
    def __init__(self, size=64):
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=TINY_IMAGENET_MEAN, std=TINY_IMAGENET_STD),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class EvaluationTransformTinyImageNet:
    def __init__(self, size=64):
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=TINY_IMAGENET_MEAN, std=TINY_IMAGENET_STD),
        ])

    def __call__(self, x):
        return self.transform(x)


class EvaluationTransformCIFAR:
    def __init__(self, size=32):
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ])

    def __call__(self, x):
        return self.transform(x)


class EvaluationTransformMNIST:
    def __init__(self, size=32):
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD),
        ])

    def __call__(self, x):
        return self.transform(x)


def get_cifar10_ablation_dataloaders(batch_size, num_workers=4, data_dir='./data'):
    train_transform = MultiViewTransformCIFAR(size=32)
    eval_transform = EvaluationTransformCIFAR(size=32)
    
    train_ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_ds = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=eval_transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader, 3 * 32 * 32, 10


def get_mnist_ablation_dataloaders(batch_size, num_workers=4, data_dir='./data'):
    train_transform = MultiViewTransformMNIST(size=28)
    eval_transform = EvaluationTransformMNIST(size=28)
    
    train_ds = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
    test_ds = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=eval_transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader, 1 * 28 * 28, 10


def prepare_tiny_imagenet(data_dir='./data'):
    root = Path(data_dir) / 'tiny-imagenet-200'

    if not root.exists():
        zip_path = Path(data_dir) / 'tiny-imagenet-200.zip'
        if not zip_path.exists():
            print(f"Downloading Tiny-ImageNet-200 to {zip_path} (~240MB)...")
            urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path)
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_dir)

    val_images_dir = root / 'val' / 'images'
    if val_images_dir.exists():
        print("Restructuring Tiny-ImageNet validation split into per-class folders...")
        annotations = {}
        with open(root / 'val' / 'val_annotations.txt', 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                annotations[parts[0]] = parts[1]

        for filename, wnid in annotations.items():
            class_dir = root / 'val' / wnid
            class_dir.mkdir(exist_ok=True)
            src = val_images_dir / filename
            if src.exists():
                shutil.move(str(src), str(class_dir / filename))

        shutil.rmtree(val_images_dir)

    return root


def get_tiny_imagenet_ablation_dataloaders(batch_size, num_workers=4, data_dir='./data'):
    root = prepare_tiny_imagenet(data_dir)

    train_transform = MultiViewTransformTinyImageNet(size=64)
    eval_transform = EvaluationTransformTinyImageNet(size=64)

    train_ds = torchvision.datasets.ImageFolder(root / 'train', transform=train_transform)
    test_ds = torchvision.datasets.ImageFolder(root / 'val', transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              drop_last=True, pin_memory=True, persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True, persistent_workers=num_workers > 0)

    return train_loader, test_loader, 3 * 64 * 64, 200


def get_ablation_dataloaders(dataset_name, batch_size, num_workers=4, data_dir='./data'):
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        return get_cifar10_ablation_dataloaders(batch_size, num_workers=num_workers, data_dir=data_dir)
    if dataset_name == 'mnist':
        return get_mnist_ablation_dataloaders(batch_size, num_workers=num_workers, data_dir=data_dir)
    if dataset_name in ('tiny_imagenet', 'tiny-imagenet', 'tinyimagenet'):
        return get_tiny_imagenet_ablation_dataloaders(batch_size, num_workers=num_workers, data_dir=data_dir)
    raise ValueError(f'Unsupported dataset: {dataset_name}')


def get_probe_train_loader(dataset_name, batch_size, num_workers=4, data_dir='./data'):
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        transform = EvaluationTransformCIFAR(size=32)
        ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    elif dataset_name == 'mnist':
        transform = EvaluationTransformMNIST(size=28)
        ds = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    elif dataset_name in ('tiny_imagenet', 'tiny-imagenet', 'tinyimagenet'):
        root = prepare_tiny_imagenet(data_dir)
        transform = EvaluationTransformTinyImageNet(size=64)
        ds = torchvision.datasets.ImageFolder(root / 'train', transform=transform)
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}')

    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=True, persistent_workers=num_workers > 0)
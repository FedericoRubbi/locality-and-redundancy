import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def get_cifar100_dataloaders(batch_size=128, num_workers=4, data_dir='./data', distributed=False):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    trainset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
    
    if distributed:
        train_sampler = DistributedSampler(trainset)
        trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, 
                                 num_workers=num_workers, pin_memory=True)
    else:
        train_sampler = None
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                                 num_workers=num_workers, pin_memory=True)
    
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    return trainloader, testloader, train_sampler

def get_imagenet100_dataloaders(batch_size=128, num_workers=4, data_dir='./data/imagenet100', distributed=False):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    trainset = datasets.ImageFolder(train_dir, transform=transform_train)
    valset = datasets.ImageFolder(val_dir, transform=transform_test)
    
    if distributed:
        train_sampler = DistributedSampler(trainset)
        trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, 
                                 num_workers=num_workers, pin_memory=True)
    else:
        train_sampler = None
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                                 num_workers=num_workers, pin_memory=True)
        
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    return trainloader, valloader, train_sampler
import torch
from torch.utils.data import DataLoader, random_split, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def create_cifar10_dataloaders(valid_split, batch_size=128):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
     ])

    dataset = datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)
    
    train_size = int((1 - valid_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                        shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=4)

    # Hard-coded classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, valid_loader, test_loader, classes
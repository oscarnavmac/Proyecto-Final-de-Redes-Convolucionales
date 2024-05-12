import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def create_cifar10_dataloaders(batch_size=4):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
     ])

    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    # Hard-coded classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes
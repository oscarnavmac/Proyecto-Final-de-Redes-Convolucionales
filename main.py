import torch
from data_setup import create_cifar10_dataloaders
from student_model import LightNN
from train import train, train_KD
from eval import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently running on {device}")

train_loader, test_loader, classes = create_cifar10_dataloaders(128)

torch.manual_seed(42)
student = LightNN(num_classes=10).to(device)
train(student, train_loader, epochs=10, learning_rate=0.001, device=device)
test_accuracy_deep = test(student, test_loader, device)
import torch
from data_setup import create_cifar10_dataloaders
from student_model import LightNN
from teacher_model import DeepNN
from train import train, train_KD
from eval import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_loader, test_loader, classes = create_cifar10_dataloaders()

torch.manual_seed(42)
nn_deep = DeepNN(num_classes=10).to(device)
train(nn_deep, train_loader, epochs=10, learning_rate=0.001, device=device)
test_accuracy_deep = test(nn_deep, test_loader, device)
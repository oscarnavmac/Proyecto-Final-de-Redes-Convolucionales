import torch
from data_setup import create_cifar10_dataloaders
from student_model import LightNN
from train import train, train_KD
from eval import test
from model import TinyVGG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently running on {device}")

train_loader, test_loader, classes = create_cifar10_dataloaders(batch_size=128)

teacher = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
total_params_teacher = "{:,}".format(sum(p.numel() for p in teacher.parameters()))
print(f"vgg11 parameters: {total_params_teacher}")
test_accuracy_deep = test(teacher, test_loader, device)

torch.manual_seed(42)
student = LightNN(num_classes=10).to(device)
total_params_student = "{:,}".format(sum(p.numel() for p in student.parameters()))
print(f"student parameters: {total_params_student}")
train_KD(teacher=teacher, student=student, train_loader=train_loader, epochs=10,
         learning_rate=0.001, T=2, alpha=0.75, device=device)
test_accuracy_deep = test(student, test_loader, device)

torch.manual_seed(42)
student = TinyVGG(num_classes=10).to(device)
total_params_student = "{:,}".format(sum(p.numel() for p in student.parameters()))
print(f"student parameters: {total_params_student}")
train_KD(teacher=teacher, student=student, train_loader=train_loader, epochs=10,
         learning_rate=0.001, T=2, alpha=0.75, device=device)
test_accuracy_deep = test(student, test_loader, device)
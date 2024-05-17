import torch
import matplotlib.pyplot as plt
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
print(f"vgg16 parameters: {total_params_teacher}")
test_accuracy_deep = test(teacher, test_loader, device)

torch.manual_seed(42)
tiny = TinyVGG(num_classes=10).to(device)
total_params_tiny = "{:,}".format(sum(p.numel() for p in tiny.parameters()))
print(f"tiny model parameters: {total_params_tiny}")
train_losses = train(tiny, train_loader, epochs=40, learning_rate=0.001, device=device)
test_accuracy_deep = test(tiny, test_loader, device)

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
#plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

torch.manual_seed(42)
student = TinyVGG(num_classes=10).to(device)
print(f"student parameters: {total_params_tiny}")
kd_losses = train_KD(teacher=teacher, student=student, train_loader=train_loader, epochs=40,
         learning_rate=0.001, T=5, alpha=0.9, device=device)
test_accuracy_deep = test(student, test_loader, device)

plt.figure(figsize=(10, 5))
plt.plot(kd_losses, label='Training Loss')
#plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
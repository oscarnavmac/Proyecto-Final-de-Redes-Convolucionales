import torch
import matplotlib.pyplot as plt
from data_setup import create_cifar10_dataloaders
from student_model import LightNN
from train import train, train_KD
from eval import test
from model import TinyVGG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Currently running on {device}")

train_loader, valid_loader, test_loader, classes = create_cifar10_dataloaders(valid_split=0.1,
                                                                              batch_size=256)

teacher = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
total_params_teacher = "{:,}".format(sum(p.numel() for p in teacher.parameters()))
print(f"vgg16 parameters: {total_params_teacher}")
test_accuracy_deep = test(teacher, test_loader, device)

torch.manual_seed(42)
tiny = TinyVGG(num_classes=10).to(device)
total_params_tiny = "{:,}".format(sum(p.numel() for p in tiny.parameters()))
print(f"tiny model parameters: {total_params_tiny}")
train_losses = train(tiny, train_loader, valid_loader, epochs=30, learning_rate=0.1, device=device)
test_accuracy_normal = test(tiny, test_loader, device)

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
#plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss of Vanilla Model')
plt.legend()
plt.savefig('train_loss.png')

torch.manual_seed(42)
student = TinyVGG(num_classes=10).to(device)
print(f"student parameters: {total_params_tiny}")
kd_losses = train_KD(teacher, student, train_loader, valid_loader, epochs=30, 
                     learning_rate=0.1, T=2, alpha=0.5, device=device)
test_accuracy_distill = test(student, test_loader, device)

plt.figure(figsize=(10, 5))
plt.plot(kd_losses, label='Training Loss')
#plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss of Distilled Model')
plt.legend()
plt.savefig('kd_loss.png')
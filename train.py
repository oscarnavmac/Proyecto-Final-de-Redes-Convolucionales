import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from eval import validate

def train(model, train_loader, valid_loader, epochs, learning_rate, device):
    train_losses = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # inputs: Una coleccion de imagenes con batch_size tamano de lote
            # labels: Un vector de dimensionalidad batch_size con enteros denotando la clase para cada imagen
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        valid_accuracy = validate(model, valid_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss},  Valid Acc: {valid_accuracy:.2f}%")

    return train_losses

        
def train_KD(teacher, student, train_loader, valid_loader, epochs, learning_rate, 
             Temperature, alpha, device):
    train_losses = []
    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()
    student.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(inputs)

            student_logits = student(inputs)
            
            distill_loss = CELoss(F.softmax(student_logits / Temperature, dim=-1),
                              F.softmax(teacher_logits / Temperature, dim=-1))

            student_loss = CELoss(student_logits, labels)

            loss = alpha*student_loss + (1. - alpha)*distill_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        valid_accuracy = validate(model, valid_loader, CELoss, device)
        print(f"{epoch+1}/{epochs}, Loss: {epoch_loss}, Valid Acc: {valid_accuracy:.2f}%")

    return train_losses
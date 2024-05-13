import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train(model, train_loader, epochs, learning_rate, device):
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

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

        
def train_KD(teacher, student, train_loader, epochs, learning_rate, T, alpha, device):
    
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

            loss_fct = nn.KLDivLoss(reduction="batchmean")
            loss_kd = T**2 * loss_fct(
                    F.log_softmax(student_logits / T, dim=-1),
                    F.softmax(teacher_logits / T, dim=-1))

            loss_ce = CELoss(student_logits, labels)

            loss = alpha * loss_ce + (1. - alpha) * loss_kd

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"{epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
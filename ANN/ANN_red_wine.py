# red wine only consolidated script

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class WineQualityNet(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(WineQualityNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(hidden_layers)):
            self.layers.append(nn.Linear(input_size if i == 0 else hidden_layers[i-1], hidden_layers[i]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_layers[-1], output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def prepare_tensors(X_train, X_val, X_test, y_train, y_val, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    return X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor

def split_data(X, y, test_size=0.2, val_size=0.1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size_adjusted, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_dataloader(X_tensor, y_tensor, batch_size=64):
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=100):
    train_losses, val_losses, val_accuracies = [], [], []
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        # Calculate and store average losses and accuracy
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)

        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')

    return train_losses, val_losses, val_accuracies

# Evaluation Function
def evaluate_model(model, test_loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Main Script
# Load red wine data
red_wine = pd.read_csv('Data/winequality-red_clean.csv')

# Prepare data
X_train, X_val, X_test, y_train, y_val, y_test = split_data(red_wine.drop('quality', axis=1), red_wine['quality'])
X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor = prepare_tensors(X_train, X_val, X_test, y_train, y_val, y_test)

# DataLoaders
batch_size = 64 
train_loader = create_dataloader(X_train_tensor, y_train_tensor, batch_size=batch_size)
val_loader = create_dataloader(X_val_tensor, y_val_tensor, batch_size=batch_size)
test_loader = create_dataloader(X_test_tensor, y_test_tensor, batch_size=batch_size)

# Model, Loss, and Optimizer
input_size = X_train.shape[1]
output_size = 10
hidden_layers = [64, 32]
model = WineQualityNet(input_size, hidden_layers, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Training Loop with Scheduler
train_losses, val_losses, val_accuracies = train_model(model, criterion, optimizer, train_loader, val_loader, epochs=50)

# Evaluate the model
print("Evaluating the model")
accuracy = evaluate_model(model, test_loader)
print(f'Test Accuracy: {accuracy}%')


# Plotting
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()

print("Saving the model")
torch.save(model.state_dict(), 'wine_quality_model_red_wine.pth')

def evaluate_model_by_class_and_save_predictions(model, test_loader, num_classes=10):
    model.eval()
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.tolist())
            all_labels.extend(y_batch.tolist())
            c = (predicted == y_batch).squeeze()
            for i in range(y_batch.size(0)):
                label = y_batch[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    class_accuracies = [(100 * class_correct[i] / class_total[i]) if class_total[i] > 0 else 0 for i in range(num_classes)]
    return class_accuracies, all_predictions, all_labels

class_accuracies, all_predictions, all_labels = evaluate_model_by_class_and_save_predictions(model, test_loader)

predictions_df = pd.DataFrame({
    'Predictions': all_predictions,
    'True Labels': all_labels
})

# Save to CSV
predictions_df.to_csv('data/wine_quality_predictions_red.csv', index=False)

predictions_df.head()

# Plotting class-wise accuracies
plt.figure(figsize=(10, 6))
plt.bar(range(10), class_accuracies)
plt.xlabel('Wine Quality Class')
plt.ylabel('Accuracy (%)')
plt.title('Prediction Accuracy by Class for Red Wine')
plt.xticks(range(10))
plt.show()
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transforms
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# Load MNIST dataset
train_data = datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
test_data = datasets.MNIST(root="./data", train=False, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)


# Simple Linear MLP model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.model(x)


model = SimpleNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Track losses
train_losses = []
test_losses = []
test_accuracies = []

# Training loop
epochs = 25
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Evaluation
    model.eval()
    total_test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_test_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)

    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    acc = 100.0 * correct / total
    test_accuracies.append(acc)
    print(f"Test Accuracy: {acc:.2f}%")

    print(
        f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Accuracy: {acc}%"
    )


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# Loss subplot
ax1.plot(train_losses, label="Train Loss")
ax1.plot(test_losses, label="Test Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Train vs Test Loss")
ax1.legend()
ax1.grid(True)

# Accuracy subplot
ax2.plot(test_accuracies, label="Test Accuracy", color="orange", linestyle="--")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Test Accuracy")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("mnist_loss_plot.png")
plt.show()

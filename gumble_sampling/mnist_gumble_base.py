import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from gumble_sampling_layer import GumbelSamplingLayer
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_data = datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
test_data = datasets.MNIST(root="./data", train=False, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)


# Model using GumbelSamplingLayer
class GumbelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = GumbelSamplingLayer(784, 512, 256)
        self.relu = nn.ReLU()
        self.fc2 = GumbelSamplingLayer(256, 256, 128)
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.output(x)


model = GumbelNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_losses = []
test_accuracies = []
test_losses = []
# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    total_test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_test_loss += loss.item()
            # Calculate accuracy
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    acc = 100.0 * correct / total
    test_accuracies.append(acc)
    print(
        f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Accuracy: {acc}%"
    )

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# First subplot: Losses
ax1.plot(train_losses, label="Train Loss")
ax1.plot(test_losses, label="Test Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Train vs Test Loss on MNIST")
ax1.legend()
ax1.grid(True)

# Second subplot: Accuracy
ax2.plot(test_accuracies, label="Test Accuracy", linestyle="--", color="orange")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Test Accuracy on MNIST")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("mnist_loss_plot_gumble.png")

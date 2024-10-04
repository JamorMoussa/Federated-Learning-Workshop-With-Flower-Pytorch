import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fedtorch.models import MNISTModel  # Assuming the model class is in the models.py
from fedtorch.utils.utils import federated_averaging  # Assuming the fed avg func is in another file

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple criterion and optimizer
criterion = nn.CrossEntropyLoss()

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize multiple models (simulating multiple clients)
client_models = [MNISTModel().to(device) for _ in range(3)]

# Optimizers for each client model
optimizers = [optim.SGD(model.parameters(), lr=0.01) for model in client_models]

# Train each client model
for i, model in enumerate(client_models):
    print(f"Training model {i + 1}")
    model.train_model(train_loader, criterion, optimizers[i], device=device, epochs=2)

# Perform federated averaging
aggregated_weights = federated_averaging(client_models)

# Initialize a global model and load the averaged weights
global_model = MNISTModel().to(device)
global_model.load_state_dict(aggregated_weights)

# Evaluate the global model on test data (optional)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define a test function to evaluate the global model
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# Evaluate the global model
evaluate_model(global_model, test_loader, device)

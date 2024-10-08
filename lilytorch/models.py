import torch
import torch.nn as nn

class MNISTModel(nn.Module):

    def __init__(self):
        super(MNISTModel, self).__init__()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128), 
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, inputs: torch.Tensor):

        return self.fc(inputs)

    def train_model(self, train_loader, criterion, optimizer, device="cpu", epochs=10):
        self.train()  # Set the model to training mode
        self.to(device)  # Send the model to the appropriate device

        for epoch in range(epochs):
            running_loss = 0
            correct = 0
            total = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.view(inputs.size(0), -1)  # Flatten the images

                optimizer.zero_grad()  # Reset the gradients

                outputs = self(inputs)  # Forward pass
                loss = criterion(outputs, targets)  # Compute loss

                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            epoch_loss = running_loss / len(train_loader)
            accuracy = 100 * correct / total

            print("\n\n", "="* 7, "Training Mode", "="*24)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
            print("="* 44, "\n\n")

        # print("Training completed.")

    def evaluate_model(self, test_loader, criterion, device="cpu"):
        self.eval()  # Set the model to evaluation mode
        self.to(device)  # Send the model to the appropriate device

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():  # Disable gradient computation
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.view(inputs.size(0), -1)  # Flatten the images

                outputs = self(inputs)  # Forward pass
                loss = criterion(outputs, targets)  # Compute loss

                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        average_loss = total_loss / len(test_loader)
        accuracy = 100 * correct / total

        print("="* 7, "Eval Mode", "="*24)
        print(f"Test Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")
        print("="* 44, "\n\n")
        
        return average_loss, accuracy

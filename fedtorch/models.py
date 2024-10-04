import torch, torch.nn as nn


class MNISTModel(nn.Module):

    def __init__(self):
        super(MNISTModel, self).__init__()

        self.fc = nn.Sequential(
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
    def train_model(self,train_loader,criterion,optimizer,device="cpu",epochs=10):
        self.train()
        self.to(device)
        for epoch in range(epochs):
            running_loss=0
            correct=0
            total=0
            for inputs,targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.view(inputs.size(0), -1)
                optimizer.zero_grad()  # Reset the gradients
                 # Forward pass
                outputs = self(inputs)
                # Loss calculation
                loss = criterion(outputs, targets)

                # Backpropagation
                loss.backward()

                # Update weights
                optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            epoch_loss = running_loss / len(train_loader)
            accuracy = 100 * correct / total
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

        print("Training completed.")





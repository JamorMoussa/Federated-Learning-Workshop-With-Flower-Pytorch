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
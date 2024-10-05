import torch, torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import lilytorch as lt 



if __name__ == "__main__":

    model = lt.models.MNISTModel()

    dataset = lt.datasets.FederatedMnistDataset.load("./save/mnist-client-1.pt")

    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    
    opt = optim.Adam(model.parameters(), lr=0.01)


    model.train_model(
        train_loader= train_loader,
        criterion= loss_fn,
        optimizer= opt, 
        device= torch.device("cpu"),
        epochs= 10
    )


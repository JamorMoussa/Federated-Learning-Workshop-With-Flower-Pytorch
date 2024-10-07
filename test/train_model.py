import torch, torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import lilytorch as lt 



if __name__ == "__main__":

    model = lt.models.MNISTModel()

    dataset = lt.datasets.FederatedMnistDataset.load("./save/mnist-client-1.pt")

    test_set = lt.datasets.FederatedMnistDataset.load("./save/mnist-global-test-set.pt")


    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=32)

    loss_fn = nn.CrossEntropyLoss()
    
    opt = optim.Adam(model.parameters(), lr=0.01)


    model.train_model(
        train_loader= train_loader,
        criterion= loss_fn,
        optimizer= opt, 
        epochs= 1,
        device= torch.device("cpu"),
    )

    model.evaluate_model(
        test_loader= test_loader,
        criterion= loss_fn,
        device= torch.device("cpu")
    )

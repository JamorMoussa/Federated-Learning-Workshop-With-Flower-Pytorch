from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import ndarrays_to_parameters, Context
from flwr.server import ServerApp, ServerConfig
from flwr.server import ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation

from .models import MNISTModel
from .utils import set_weights, get_weights

import torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from typing import Dict



class FlowerClient(NumPyClient):
    def __init__(self, client_id: int,  trainset, testset):
        super(FlowerClient, self).__init__()

        self.net: MNISTModel = MNISTModel()

        self.client_id = client_id

        self.trainset = trainset
        self.testset = testset

        self.train_loader = DataLoader(dataset=self.trainset, batch_size=35, shuffle=True)
        self.test_loader = DataLoader(dataset=self.testset, batch_size=35)

        self.loss_fn = nn.CrossEntropyLoss()
        
        self.opt = optim.Adam(self.net.parameters(), lr=0.01)

    def get_parameters(self, config):
        return get_weights(self.net)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)

        self.net.train_model(
            train_loader= self.train_loader,
            criterion= self.loss_fn,
            optimizer= self.opt, 
            device= torch.device("cpu"),
            epochs= 1
        )

        return get_weights(self.net), len(self.trainset), {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        
        loss, accuracy = self.net.evaluate_model(
            test_loader=self.test_loader,
            criterion= self.loss_fn,
            device= torch.device("cpu")
        )
        return loss, len(self.test_loader), {"accuracy": accuracy}



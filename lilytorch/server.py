from .models import MNISTModel

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lilytorch as lt
import os.path as osp

import flwr as fl


def evaluate(server_round, parameters, config, net, test_loader):

    lt.utils.set_weights(net, parameters)

    net.evaluate_model(
        test_loader= test_loader,
        criterion= nn.CrossEntropyLoss(),
        device= torch.device("cpu")
    )


def start_server(save_dir: str, rounds: int): 

    test_set = lt.datasets.FederatedMnistDataset.load(
        filepath= osp.join(save_dir, "mnist-global-test-set.pt")
    )

    test_loader = DataLoader(
        dataset= test_set, batch_size=35
    )

    model = lt.models.MNISTModel()

    strategy = fl.server.strategy.FedAvg(
        min_available_clients=3,
        min_fit_clients=3,
        min_evaluate_clients=3,
        evaluate_fn= lambda server_round, parameters, config: evaluate(server_round, parameters, config, model, test_loader)
    )

    fl.server.start_server(
        server_address= "localhost:8080",
        config= fl.server.ServerConfig(num_rounds=rounds),
        strategy= strategy
    )

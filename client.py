import lilytorch as lt
from lilytorch.client import FlowerClient


import flwr as fl

import argparse
import os.path as osp



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description= "")

    parser.add_argument('--client_id', type= int, default=1)
    parser.add_argument('--data_dir', type= str, default="./save")

    args = parser.parse_args()

    train_set = lt.datasets.FederatedMnistDataset.load(
        osp.join(args.data_dir, f"mnist-client-{str(args.client_id)}.pt")
    )

    test_set = lt.datasets.FederatedMnistDataset.load(
        osp.join(args.data_dir, f"mnist-global-test-set.pt")
    )

    args = parser.parse_args()

    client = fl.client.start_client(
        server_address= "0.0.0.0:8080",
        client= FlowerClient(
            client_id= args.client_id,
            trainset= train_set, 
            testset= test_set
        ).to_client()
    )
    
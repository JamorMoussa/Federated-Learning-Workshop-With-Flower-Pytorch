import torch 
from torch.utils.data import Dataset

from torchvision import datasets
import torchvision.transforms as T

from pathlib import Path
import os.path as osp
import logging as log


class FederatedMnistDataset(Dataset):
        def __init__(self, data):
            self.data = data

        @staticmethod
        def load(dir: str):
            return torch.load(dir, weights_only= True)

        def save(self, save_dir: str):
            torch.save(self.data, f = save_dir)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]


def filter_train_dataset(trainset, exclude_digits: list[int]):
     
    return [(x, y) for x, y in trainset if y not in exclude_digits]


def create_federated_mnist_datasets(
    root_dir: str | Path, 
    save_dir: str | Path,
    exclude_digits_per_client: list[list[int]] = [[1, 3, 7], [2, 5, 8], [4, 6, 9]]
) -> tuple[FederatedMnistDataset]:

    log.basicConfig(level=log.DEBUG)
    
    root_dir = Path(root_dir)
    save_dir = Path(save_dir)

    if not root_dir.exists(): root_dir.mkdir()
    if not save_dir.exists(): raise FileNotFoundError("No such save directory")

    transform = T.Compose([
        T.ToTensor(), 
        T.Normalize((0.1307,), (0.3081,))
    ])

    if not Path(osp.join(root_dir, "MNIST")).exists():
         log.info("start downloading the mnist dataset.")

    trainset = datasets.MNIST(
        root=root_dir, download=True, train=True, transform=transform
    )

    testset = datasets.MNIST(
        root=root_dir, download=True, train=False, transform=transform
    )

    log.info("The donwloding is completed")

    log.info("saving federated mnist for client 1 ...")    

    FederatedMnistDataset(
         data= filter_train_dataset(
                trainset= trainset, exclude_digits= exclude_digits_per_client[0]
        )
    ).save(save_dir=osp.join(save_dir, "mnsit-client-1.pt") )

    log.info("saving federated mnist for client 2 ...")

    FederatedMnistDataset(
         data= filter_train_dataset(
              trainset= trainset, exclude_digits= exclude_digits_per_client[1]
        )
    ).save(save_dir=osp.join(save_dir, "mnsit-client-2.pt") )

    log.info("saving federated mnist for client 3 ...")

    FederatedMnistDataset(
         data= filter_train_dataset(
              trainset= trainset, exclude_digits= exclude_digits_per_client[1]
        )
    ).save(save_dir=osp.join(save_dir, "mnsit-client-3.pt"))


    log.info("saving test dataset ...")

    FederatedMnistDataset(
        data= (testset.data, testset.targets)
    ).save(save_dir=osp.join(save_dir, "mnsit-global-test-set.pt"))

    log.info("finished ...")






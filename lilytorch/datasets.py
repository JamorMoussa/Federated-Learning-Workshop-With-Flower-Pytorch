import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from torchvision import datasets
import torchvision.transforms as T

from pathlib import Path
import os.path as osp
import logging as log


class FederatedMnistDataset(Dataset):
    def __init__(self, data):
        self.data = data

    @staticmethod
    def load(filepath: str):
        return torch.load(filepath, weights_only=False)

    def save(self, save_dir: str):
        images, labels = zip(*self.data)
        images = torch.stack(images) 
        labels = torch.tensor(labels)

        dataset = TensorDataset(images, labels)
        torch.save(dataset, save_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def filter_dataset_by_class(dataset, exclude_digits: list[int]):
    return [(x, y) for x, y in dataset if y not in exclude_digits]


def create_federated_mnist_datasets(
    root_dir: str | Path,
    save_dir: str | Path,
    exclude_digits_per_client: list[list[int]] = [[1, 3, 7], [2, 5, 8], [4, 6, 9]]
) -> tuple[FederatedMnistDataset]:

    log.basicConfig(level=log.DEBUG)
    
    root_dir = Path(root_dir)
    save_dir = Path(save_dir)

    if not root_dir.exists(): root_dir.mkdir()
    if not save_dir.exists(): save_dir.mkdir()

    transform = T.Compose([
        T.ToTensor(), 
        T.Normalize((0.1307,), (0.3081,))
    ])

    if not Path(osp.join(root_dir, "MNIST")).exists():
        log.info("Start downloading the MNIST dataset.")

    trainset = datasets.MNIST(
        root=root_dir, download=True, train=True, transform=transform
    )

    testset = datasets.MNIST(
        root=root_dir, download=True, train=False, transform=transform
    )

    log.info("Downloading completed")

    for client_idx, exclude_digits in enumerate(exclude_digits_per_client, 1):
        log.info(f"Saving federated MNIST for client {client_idx} ...")
        FederatedMnistDataset(
            data=filter_dataset_by_class(
                dataset=trainset, exclude_digits=exclude_digits
            )
        ).save(save_dir=osp.join(save_dir, f"mnist-client-{client_idx}.pt"))

    log.info("Saving global test dataset ...")
    FederatedMnistDataset(
        data=list(zip(testset.data.type(torch.float32), testset.targets)) 
    ).save(save_dir=osp.join(save_dir, "mnist-global-test-set.pt"))

    log.info("Finished ...")



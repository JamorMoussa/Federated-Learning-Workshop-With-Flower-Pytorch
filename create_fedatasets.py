from fedtorch.datasets import create_federated_mnist_datasets

import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", type= str)
    parser.add_argument("--save_dir", type= str)

    args = parser.parse_args()


    create_federated_mnist_datasets(
        root_dir= args.root_dir, 
        save_dir= args.save_dir
    )




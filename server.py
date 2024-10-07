from lilytorch.server import start_server

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description= "")

    parser.add_argument('--data_dir', type= str, default="./save")
    parser.add_argument('--rounds', type= int, default= 5)

    args = parser.parse_args()

    start_server(
        save_dir= args.data_dir,
        rounds= args.rounds
    )

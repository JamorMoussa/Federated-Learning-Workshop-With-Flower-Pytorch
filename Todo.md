# Federated Learning Project with Flower and PyTorch - To-Do List

This project simulates federated learning using **Flower** as the framework for communication and orchestration between clients and server, and **PyTorch** for model training. The simulation will involve a server and multiple clients, each training locally on its own subset of the MNIST dataset.

---

## 01. Federated Training Process Setup

### 1.1 Utility Functions (`utils` module)

- [ ] **Define `set_weight()` function (for PyTorch models):**
    - This function will set the model's weights received from the server.
    - It will take a PyTorch model instance and a list of tensors as input and update the model's `state_dict()`.

    ```python
    def set_weight(model, weights):
        model.load_state_dict(weights)
    ```

- [ ] **Define `get_weight()` function (for PyTorch models):**
    - This function will retrieve the current weights from a PyTorch model.
    - It will return the model's weights using `state_dict()`.

    ```python
    def get_weight(model):
        return model.state_dict()
    ```

---

### 1.2 Client-Side Implementation (`client` module)

- [ ] **Create the `Client` class using Flower's `flwr.client.NumPyClient`:**
    - Each client will handle local training on its unique dataset (a subset of MNIST).
    - Flower provides the `NumPyClient` interface, which we will subclass to define our client behavior.
    - The class should implement the following methods:

      - **`get_parameters()`**:
        - Return the client's model weights (using `get_weight()`).

      - **`fit()`**:
        - Perform local training on the client's dataset using PyTorch.
        - Return the updated model parameters after training.

      - **`evaluate()`**:
        - Evaluate the model on the client's local data and return evaluation metrics (e.g., accuracy, loss).

    ```python
    import flwr as fl
    import torch

    class Client(fl.client.NumPyClient):
        def __init__(self, model, train_loader, test_loader, device):
            self.model = model
            self.train_loader = train_loader
            self.test_loader = test_loader
            self.device = device

        def get_parameters(self):
            return get_weight(self.model)

        def fit(self, parameters, config):
            set_weight(self.model, parameters)
            self.model.train()
            # Training loop for PyTorch goes here
            return get_weight(self.model), len(self.train_loader.dataset), {}

        def evaluate(self, parameters, config):
            set_weight(self.model, parameters)
            self.model.eval()
            # Evaluation loop for PyTorch goes here
            return 0.0, len(self.test_loader.dataset), {}  # Return loss, dataset size, other metrics
    ```

- [ ] **Prepare the dataset split for clients:**
    - Exclude certain digit classes from each client's MNIST dataset to simulate different local datasets.
    - Use PyTorch’s `DataLoader` to handle loading of training and test data.

---

### 1.3 Server-Side Implementation (`server` module)

- [ ] **Create the `Server` class using Flower's server components:**
    - The server will handle communication with the clients, coordinate training rounds, and aggregate the model weights.
    - Flower will manage these aspects, so the focus will be on configuring the server's behavior.

    ```python
    import flwr as fl

    # Define your aggregation strategy (e.g., FedAvg)
    strategy = fl.server.strategy.FedAvg()

    # Start the server
    def start_server():
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            strategy=strategy,
            config={"num_rounds": 5}
        )
    ```

- [ ] **Set up aggregation strategy:**
    - Use Flower's `FedAvg` strategy to average weights across clients.
    - Flower provides built-in aggregation strategies such as **Federated Averaging (FedAvg)**, or you can define custom ones if needed.

    ```python
    from flwr.server.strategy import FedAvg

    strategy = FedAvg(
        fraction_fit=0.1,  # Fraction of clients used for training
        min_fit_clients=2,  # Minimum number of clients required to be available
        min_eval_clients=2,  # Minimum number of clients required to evaluate
        min_available_clients=3,  # Total clients in the system
    )
    ```

---

## 02. Simulating Federated Learning with Docker

### 2.1 Setting up Clients with Docker

- [ ] **Prepare Dockerfiles for clients:**
    - Each client container should contain the local version of the MNIST dataset.
    - The Dockerfile should install PyTorch, Flower, and any other necessary dependencies.
    
    Example Dockerfile for a client:
    
    ```dockerfile
    FROM python:3.9-slim

    WORKDIR /app

    # Install dependencies
    RUN pip install torch torchvision flwr

    # Copy client code
    COPY . /app

    # Start client
    CMD ["python", "client.py"]
    ```

- [ ] **Client Startup Script (`client.py`):**
    - Each client should instantiate a Flower `Client` and connect to the server.
    
    ```python
    import flwr as fl
    from client_module import Client, load_data

    if __name__ == "__main__":
        model = ...  # Initialize your PyTorch model here
        train_loader, test_loader = load_data()
        client = Client(model, train_loader, test_loader, device='cpu')
        fl.client.start_numpy_client(server_address="server:8080", client=client)
    ```

---

### 2.2 Setting up the Server with Docker

- [ ] **Prepare Dockerfile for the server:**
    - The server will manage all incoming connections from the clients and orchestrate the training process.
    
    Example Dockerfile for the server:
    
    ```dockerfile
    FROM python:3.9-slim

    WORKDIR /app

    # Install dependencies
    RUN pip install flwr torch torchvision

    # Copy server code
    COPY . /app

    # Start server
    CMD ["python", "server.py"]
    ```

- [ ] **Server Startup Script (`server.py`):**
    - The server will start and wait for client connections.
    
    ```python
    from server_module import start_server

    if __name__ == "__main__":
        start_server()
    ```

---

### 2.3 Running the Simulation with Docker Compose

- [ ] **Create Docker Compose file to define server and client services:**
    - Define services for the server and three clients, each running in separate containers.
    
    Example `docker-compose.yml`:
    
    ```yaml
    version: '3'
    services:
      server:
        build: ./server
        ports:
          - "8080:8080"
      
      client1:
        build: ./client
        environment:
          - CLIENT_ID=1
      
      client2:
        build: ./client
        environment:
          - CLIENT_ID=2

      client3:
        build: ./client
        environment:
          - CLIENT_ID=3
    ```

- [ ] **Run the simulation:**
    - Use `docker-compose up` to start all services (server and clients).
    - Monitor the Flower server logs to observe the federated learning process, including client-server communication and aggregation of model weights.

---

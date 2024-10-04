# Federated Learning Project - To-Do List

A project to simulate federated learning with a distributed training framework. The system consists of a server and multiple clients, each of which trains locally on its own dataset.

---

## 01. Federated Training Process Setup

### 1.1 Utility Functions (`utils` module)

- [ ] **Define `set_weight()` function:**
    - This function will be responsible for setting model weights after receiving them from the server.
    - It should take a model instance and a list of weights as input and apply the weights to the model.
  
- [ ] **Define `get_weight()` function:**
    - This function will retrieve and return the current weights of a model.
    - It will be used by the client to send its locally trained model weights to the server for aggregation.

---

### 1.2 Client-Side Implementation (`client` module)

- [ ] **Create the `Client` class:**
    - The client will handle local training on a subset of the MNIST dataset. Each client will have its own version of the dataset, potentially excluding some digit classes.
    - Each client should implement the following methods:
      - **`fit(self)`**:
        - Perform local training on the client's dataset. It should update the local model's weights.
      - **`evaluate(self)`**:
        - Evaluate the trained model only on the client's local dataset and return the evaluation metrics (e.g., accuracy, loss).
      - **`send_weights(self)`**:
        - Retrieve and send the local model weights to the server after training using the `get_weight()` utility function.
      - **`receive_weights(self)`**:
        - Accept updated global weights from the server and apply them to the local model using `set_weight()`.

---

### 1.3 Server-Side Implementation (`server` module)

- [ ] **Create the `Server` class:**
    - The server coordinates training by collecting model weights from clients, aggregating them, and broadcasting the updated weights back to the clients.
    - The server should implement the following methods:
      - **`receive_weights(self)`**:
        - Collect weights from multiple clients after they have trained on their local data.
      - **`aggregate(self)`**:
        - Implement an aggregation strategy (e.g., averaging the weights from different clients).
        - Ensure this function handles cases where certain clients might have trained on datasets of different sizes.
      - **`send_weights(self)`**:
        - Send the updated global model weights back to all clients for the next round of training.

---

## 02. Simulating Federated Learning with Docker

After the implementation of the federated learning setup, the next step is to simulate the training using Docker to replicate real-world distributed training environments. The simulation will involve three clients, each running in its own container, and a server, also in a separate container.

---

### 2.1 Setting up Clients

- [ ] **Prepare Dockerfiles for clients:**
    - Each client container should have access to its own local dataset (a subset of MNIST).
    - The Dockerfile should install all necessary dependencies such as Python, TensorFlow, PyTorch, or any other libraries required for training.
    - Ensure that each client has a distinct dataset by excluding different digits from MNIST.

- [ ] **Configure the client startup script:**
    - The client container should start by training on its local data and then send the trained weights to the server.
    - The client should continue to participate in multiple rounds of training as dictated by the server.

---

### 2.2 Setting up the Server

- [ ] **Prepare Dockerfile for the server:**
    - The server container should handle incoming connections from multiple clients and coordinate the training process by receiving weights, aggregating them, and sending updated weights back.
    - Ensure the Dockerfile installs necessary dependencies such as the framework used for communication between the clients and server (e.g., gRPC, Flask, or another communication protocol).

- [ ] **Configure the server startup script:**
    - The server should listen for clients and initiate rounds of federated training.
    - After each round, the server should collect weights from all clients, aggregate them, and then broadcast the updated weights back to clients for further training.

---

### 2.3 Running the Simulation

- [ ] **Create Docker Compose file:**
    - Define services for the server and three clients.
    - Each client will run in its own container with a unique local dataset.
    - Use Docker networking to enable communication between the server and clients.

- [ ] **Run the simulation:**
    - Once all containers are built, use Docker Compose to start the server and clients.
    - Monitor the federated learning process over multiple rounds of training, ensuring that the server correctly aggregates model weights and improves performance.

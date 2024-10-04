import torch

def federated_averaging(models):
    """
    Aggregates the weights of different client models using federated averaging.
    
    :param models: List of client models (assumed to have the same architecture)
    :return: Aggregated model weights (FedAvg)
    """
    # Get the state dict (weights) of the first model
    aggregated_weights = models[0].state_dict()
    
    # Iterate through the remaining models and accumulate their weights
    for key in aggregated_weights.keys():
        for model in models[1:]:
            aggregated_weights[key] += model.state_dict()[key]
        
        # Compute the average of the weights
        aggregated_weights[key] = aggregated_weights[key] / len(models)
    
    return aggregated_weights

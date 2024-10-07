import torch, torch.nn as nn

from collections import OrderedDict

def set_weights(net: nn.Module, parameters):

    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in params_dict}
    )
    net.load_state_dict(state_dict, strict=True)

    return net 

def get_weights(net: nn.Module):
    ndarrays = [
        val.cpu().numpy() for _, val in net.state_dict().items()
    ]
    return ndarrays
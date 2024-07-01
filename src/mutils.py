'''
Orthogonal weight init
'''
import torch.nn as nn

def oinit(module, gain=nn.init.calculate_gain("relu")):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module

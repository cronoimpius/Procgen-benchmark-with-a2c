"""
Random agent implementation
"""
import torch 
import numpy as np

class RandomAgent():
    def __init__(self, act_space):
        self.act_space =act_space

    def act(self, obs):
        return torch.Tensor([np.random.randint(0,self.act_space.n)])

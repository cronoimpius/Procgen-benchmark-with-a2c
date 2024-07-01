'''
Model implementation based on DQN paper
'''
import torch.nn as nn 
from mutils import oinit
class Flatten(nn.Module):
    def forward(self, obs):
        return obs.view(obs.size(0), -1)

class Model(nn.Module):
    def __init__(self, in_ch, out_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=1024, out_features=out_dim), 
            nn.ReLU(),
        )
        self.apply(oinit)

    def forward(self, obs):
        return self.layers(obs)

import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info

# TODO: try train policy with resnet

class ResNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super.__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def __call__(self, x):
        return x + self.model(x)
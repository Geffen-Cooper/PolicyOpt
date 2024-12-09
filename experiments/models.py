import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
from ptflops import get_model_complexity_info


class EnergyPolicy(nn.Module):
    def __init__(self,in_dim, hidden_dim=32):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,1)
    

    def forward(self, x):
        # 101 --> 32
        x = F.relu(self.fc1(x))

        # 32 --> 1
        x = self.fc2(x)

        return F.sigmoid(x)

class SimpleNet(nn.Module):
    def __init__(self,in_channels,classes):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, 8, 5, padding=2)
        self.conv2 = nn.Conv1d(8, 8, 5, padding=2)
        self.fc1 = nn.Linear(64,classes)
    

    def forward(self, x):
        print(x.shape)
        # (3 x 8) --> (8 x 8)
        # print(x.shape)
        x = F.relu(self.conv1(x))

        # (8 x 8) --> (8 x 8)
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # (8 x 8) --> (64)
        x = x.view(x.shape[0],-1)

        # (64) --> (classes)
        x = self.fc1(x)

        return x
    


class DSADSNet(nn.Module):
    def __init__(self,in_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, 16, 5, padding=1)
        self.conv2 = nn.Conv1d(16, 16, 3, 3)
        self.conv3 = nn.Conv1d(16, 16, 3)
        self.conv4 = nn.Conv1d(16, 16, 3)
        self.conv5 = nn.Conv1d(16, 32, 12)
        # self.fc1 = nn.Linear(64,32)
        self.fc2 = nn.Linear(32,19)
    

    def forward(self, x):
        # print(x.shape)
        # (15 x 50) --> (32 x 48)
        x = F.relu(self.conv1(x))

        # (32 x 48) --> (32 x 16)
        # print(x.shape)
        x = F.relu(self.conv2(x))

        # (32 x 16) --> (32 x 14)
        # print(x.shape)
        x = F.relu(self.conv3(x))

        # (32 x 14) --> (32 x 12)
        # print(x.shape)
        x = F.relu(self.conv4(x))

        # (32 x 12) --> (64 x 1)
        # print(x.shape)
        x = F.relu(self.conv5(x))

        # (64 x 1) --> (64) --> (32)
        x = x.view(x.shape[0],-1)
        # x = F.relu(self.fc1(x))

        # (32) --> (19)
        # print(x.shape)
        x = self.fc2(x)

        return x
    
class DiscreteQNetwork(nn.Module):
    def __init__(self, N, in_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N)
        )

    def forward(self, x):
        return self.model(x)
    
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity) 
    
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


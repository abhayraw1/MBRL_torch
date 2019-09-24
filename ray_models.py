import pdb
import torch
from torch import nn
import torch.nn.functional as F

class DynamixForward(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Linear(6, 50),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(50, 50),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(50, 6),
            nn.Tanh()
        )

    def forward(self, state, action):
        x, s_t = state.clone().split([2, 4])
        ds_tp1 = self.model(torch.cat([s_t, action], dim=-1))
        return ds_tp1 + state.clone()

class ObservationModel(nn.Module):
    def
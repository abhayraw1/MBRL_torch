import pdb
import torch
from torch import nn
import torch.nn.functional as F

from ray.rllib.models.torch.misc import normc_initializer, SlimFC

class DynamixForward(nn.Module):
    def __init__(self, in_dim, op_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 50),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(50, 50),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(50, op_dim),
            nn.Tanh()
        )

    def forward(self, state, action):
        x, s_t = state.clone().split([2, 4])
        ds_tp1 = self.model(torch.cat([s_t, action], dim=-1))
        return ds_tp1 + state.clone()


class VanillaModel(nn.Module):
    def __init__(self,
                 in_dim,
                 op_dim,
                 hidden_units,
                 hidden_activation=nn.ReLU,
                 output_activation=nn.ReLU,
                 gaussian=False,
                 model_config=None):
        super().__init__()
        self.gaussian = gaussian
        if not isinstance(hidden_activation, list):
            hidden_activation = [hidden_activation]*len(hidden_units)
        layers = []
        last_layer_size = in_dim
        for i, (units, act) in enumerate(zip(hidden_units, hidden_activation)):
            layers.append(
                SlimFC(
                    in_size=last_layer_size,
                    out_size=units,
                    initializer=normc_initializer(1.0),
                    activation_fn=act))
            last_layer_size = units

        self.model = nn.Sequential(*layers)
        self.meanl = SlimFC(
            in_size=last_layer_size,
            out_size=op_dim,
            initializer=normc_initializer(0.01),
            activation_fn=output_activation
        )
        if self.gaussian:
            self.lstdl = SlimFC(
                in_size=last_layer_size,
                out_size=op_dim,
                initializer=normc_initializer(1.0)
            )
        self.op_dim = op_dim

    def forward(self, x, return_mode='SAMPLE', random=False):
        if random:
            m = torch.rand(x.size(0), self.op_dim)
            s = torch.rand(x.size(0), self.op_dim)
            return (m, s) if self.gaussian else m
        h = self.model(x)
        m = self.meanl(h)
        if self.gaussian:
            if return_mode == 'DETERMINISTIC':
                return m
            s = torch.clamp(self.lstdl(h), -13, 2)
            if return_mode == 'PARAMS':
                return m, s
            dist = torch.distributions.normal.Normal(m, torch.exp(s))
            if return_mode == 'DIST':
                return dist
            if return_mode == 'SAMPLE':
                return dist.sample()
        return m

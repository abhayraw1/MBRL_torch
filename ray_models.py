import pdb
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import utils
from ray.rllib.models.torch.misc import normc_initializer, SlimFC
from torch.distributions import Normal


OP_DIM = {
    Normal: lambda x: x,

}

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
                 log_std_op_dim=None,
                 model_config=None,
                 dist_class=None):
        super().__init__()
        self.gaussian = gaussian
        if not isinstance(hidden_activation, list):
            hidden_activation = [hidden_activation]*len(hidden_units)
        layers = []
        last_layer_size = in_dim
        self.dist_class = dist_class or Normal
        assert self.dist_class == Normal or log_std_op_dim is not None,\
        '# of o/p units needed if dist_class != Normal!'
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
                out_size=OP_DIM[self.dist_class](op_dim),
                initializer=normc_initializer(1.0)
            )
        self.op_dim = op_dim

    def print_grads(self):
        grads = {
            k: np.array([a.grad.min(), a.grad.max(), a.grad.mean(), a.grad.std()])
            for k, a in self.named_parameters()
        }
        print(utils.dict_as_table(grads))

    def forward(self, x, random=False):
        if random:
            m = torch.rand(x.size(0), self.op_dim)
            s = torch.rand(x.size(0), self.op_dim)
            return (m, s) if self.gaussian else m
        h = self.model(x)
        m = self.meanl(h)
        if self.gaussian:
            s = torch.clamp(self.lstdl(h), -20, 1)
            return self.dist_class(m, torch.exp(s))
        return m


class DynamixForward(nn.Module):
    def __init__(self, in_dim, out_dim, lr=1e-3):
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Linear(4, 50),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(50, 50),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(50, 3),
            # nn.Tanh()
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def train(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def forward(self, state, action):
        ds_tp1 = self.model(torch.cat([state, action], dim=-1))
        return ds_tp1 + state.clone()

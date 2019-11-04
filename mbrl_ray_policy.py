import os
import pdb
import gym
import time
import torch
import numpy as np

from threading import Lock

from torch import nn
from torch.optim import Adam
from torch.distributions import *

from ray.rllib.utils.annotations import override
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.policy.policy import Policy, LEARNER_STATS_KEY
from ray.rllib.models.torch.torch_action_dist import (TorchMultinomial,
                                                      TorchDiagGaussian)
from ray_models import *

class MBRLPolicy(TorchPolicy):
    """
    Model Predictive Policy for an Agent.
    """
    @staticmethod
    def size(x):
        from gym.spaces import Discrete, Box
        if isinstance(x, Discrete):
            return x.n
        elif isinstance(x, Box):
            return int(np.prod(x.shape))
        else:
            raise ValueError('This type of space is not supported')


    def __init__(self, observation_space, action_space, config=None):
        self.config = config or {}
        self.seq_len = self.config.get('seq_len', 20)
        self.lock = Lock()
        self.device = torch.device(self.config.get('device', 'cpu'))
        self.obj2trk = self.config.get('objects_to_track', 1)
        self.obs_space = self.observation_space = observation_space
        self.act_space = self.action_space = action_space
        self.obs_input = MBRLPolicy.size(self.obs_space)
        self.act_input = MBRLPolicy.size(self.act_space)
        self.obs_space_interval = self.obs_space.high - self.obs_space.low
        self.obs_space_interval = self.convert(self.obs_space_interval)
        self.discrete_ac_sp = isinstance(action_space, gym.spaces.Discrete)
        self._action_model = VanillaModel(
            self.obs_input, self.act_input, [50, 50], gaussian=True,
            output_activation=nn.Tanh
        ).to(self.device)
        self._prediction_model = VanillaModel(
            self.obs_input + self.act_input, self.obs_input, [50, 50],
            gaussian=True, output_activation=nn.Tanh
        ).to(self.device)
        self._actor_params = {
            f'A_{k}': v for k, v in
            self._action_model.named_parameters()
        }
        self._predictor_params = {
            f'P_{k}': v for k, v in
            self._prediction_model.named_parameters()
        }
        self._all_params = {}
        self._all_params.update(self._predictor_params)
        self._all_params.update(self._actor_params)
        self.action_dist_class = TorchDiagGaussian
        self._track_id = torch.arange(0, 1, 1/self.obj2trk).view(-1, 1)
        self.random_actions = False
        self._optimizer1 = Adam(self._action_model.parameters(), lr=0.0005)
        self._optimizer2 = Adam(self._prediction_model.parameters(), lr=0.01)
        self.discount = 0.99**torch.arange(self.seq_len-1, -1, -1)[None]
        self.discount = self.discount.transpose(1, 0).float()
        self.MSE_loss =  torch.nn.MSELoss()
        self.action_transforms = [
            TanhTransform(),
            AffineTransform(
                self.convert((self.act_space.high + self.act_space.low)/2),
                self.convert((self.act_space.high - self.act_space.low)/2)
            )
        ]
        self.prediction_transforms = [
            TanhTransform(),
            AffineTransform(
                self.convert((self.obs_space.high + self.obs_space.low)/2),
                self.convert((self.obs_space.high - self.obs_space.low)/2)
            )
        ]

    def convert(self, arr):
        tensor = torch.from_numpy(np.asarray(arr))
        if tensor.dtype == torch.double:
            tensor = tensor.float()
        return tensor.to(self.device)

    def trajectory_loss(self, xs, us, lp=None, means=None):
        # The timesteps are reversed for cumsum
        # We need the t=0 cost to be sum of c_t \in {0, 1,... self.seq_len}
        txs = torch.stack(xs[::-1])
        uss = torch.stack(us[::-1])
        c, s, d = txs.transpose(0, -1).transpose(-1, 1)
        th = torch.atan2(s, c)
        cost = torch.cumsum(th**2 + 0.1*d**2 + 0.001*uss**2, dim=0)
        if lp is not None:
            lps = torch.stack(lp[::-1]).squeeze()
            expected_cost = lps*cost
            return expected_cost.mean(dim=0).mean()
        else:
            return cost.mean(dim=0).mean()

    def eval_predictor(self, batch):
        x = self.convert(batch['obs'])
        u = self.convert(batch['actions']).view(-1, self.act_input)
        x_p1 = self.convert(batch['new_obs'])
        y_p1 = self._prediction_model(torch.cat([x, u], dim=-1))
        d_p1 = TransformedDistribution(y_p1, self.prediction_transforms)
        loss = -(d_p1.log_prob(x_p1).sum(dim=-1).mean())
        # pdb.set_trace()
        mse = 0 # loss.detach().item()
        mae = 0 # (y_p1 - x_p1).abs().sum(dim=-1).mean().detach().item()
        return loss, mse, mae

    def generate_trajectory(self,
                            x,
                            seq_len=None,
                            return_log_probs=False,
                            mode='T'):
        seq_len = seq_len or self.seq_len
        xs, us, lp = [], [], []
        for i in range(seq_len):
            u = self._action_model(x)
            if mode == 'E':
                mean = u.mean
                logstd = u.stddev.log()
            d = TransformedDistribution(u, self.action_transforms)
            a = mean if mode == 'E' else d.sample()
            x = self._prediction_model(torch.cat([x, a], dim=-1)).mean.detach()
            us.append(a.squeeze().clone())
            xs.append(x.squeeze().clone())
            if return_log_probs:
                if torch.any(torch.isnan(d.log_prob(a))):
                    pdb.set_trace()
                lp.append(d.log_prob(a))
        if return_log_probs:
            return xs, us, lp
        return xs, us

    def train_predictor(self, batch):
        with self.lock:
            self._optimizer2.zero_grad()
            loss, mse, mae = self.eval_predictor(batch)
            loss.backward()
            grad_process_info = self.extra_grad_process()
            self._optimizer2.step()
            return {'NLL': loss.detach().item(), 'MSE': mse, 'MAE': mae}

    def train_actor(self, batch, num_times=1):
        with self.lock:
            x = self.convert(batch[0])
            for _ in range(num_times):
                self._optimizer1.zero_grad()
                xs, us, lp = self.generate_trajectory(
                    x, seq_len=self.seq_len, return_log_probs=True
                )
                loss = self.trajectory_loss(xs, us, lp)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self._action_model.parameters(), 5, norm_type=2
                )
                self._optimizer1.step()
            return loss.detach().item()

    @override(TorchPolicy)
    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        return_states=False,
                        eval_mode='T',
                        **kwargs):
        with self.lock:
            with torch.no_grad():
                z = self.convert(obs_batch)
                xs, us = self.generate_trajectory(z, mode=eval_mode)
            local = torch.tensor(us, requires_grad=True)
            l_opt = Adam([local], lr=0.01)
            for _ in range(20):
                x = z.clone().detach().squeeze()
                xs, us, lp = [], [], []
                l_opt.zero_grad()
                for a in local.reshape(-1, 1):
                    x = self._prediction_model(torch.cat([x, a])).mean.detach()
                    us.append(a.squeeze().clone())
                    xs.append(x.squeeze().clone())
                self.trajectory_loss(xs, us).backward()
                l_opt.step()

            # pdb.set_trace()
            us = local.reshape(-1, 1)[0].detach()
            xs = self._prediction_model(
                    torch.cat([z.clone().detach().squeeze(), us])
                ).mean.detach().numpy()
            us = us.numpy()
        return ([us], [xs], {}) if return_states else ([us], [], {})

    def learn_on_batch(self, samples):
        print('implement your learning code here\n'*10)
        return {}


    def get_initial_state(self):
        return []

    def get_weights(self):
        with self.lock:
            return {
                'a_model': {
                    k: v.cpu() for k, v in
                    self._action_model.state_dict().items()
                },
                'p_model': {
                    k: v.cpu() for k, v in
                    self._prediction_model.state_dict().items()
                }
            }

    def set_weights(self, weights, action_model=True, prediction_model=True):
        assert action_model or prediction_model
        if action_model != prediction_model:
            m = 'action' if action_model else 'prediction'
            # print('** Setting weights for {} model only **'.format(m))
        with self.lock:
            if action_model:
                self._action_model.load_state_dict(weights['a_model'])
            if prediction_model:
                self._prediction_model.load_state_dict(weights['p_model'])

    def save_models(self, path):
        torch.save(self._prediction_model.state_dict(), path+'/predictor')
        torch.save(self._action_model.state_dict(), path+'/actor')

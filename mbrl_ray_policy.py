import os
import pdb
import gym
import time
import torch
import numpy as np

from torch import nn
from torch.optim import Adam
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.policy.policy import Policy, LEARNER_STATS_KEY
from ray_models import *
from threading import Lock
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_action_dist import (TorchCategorical,
                                                      TorchDiagGaussian)


class MBRLPolicy(TorchPolicy):
    """Model Predictive Policy for an Agent in multi-agent scenario.

    You might find it more convenient to extend TF/TorchPolicy instead
    for a real policy.
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
        self.seq_len = 20
        self.lock = Lock()
        self.device = torch.device("cpu")
        self.obj2trk = self.config.get('objects_to_track', 1)
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_input = MBRLPolicy.size(self.observation_space)
        self.act_input = MBRLPolicy.size(self.action_space)
        self.discrete_ac_sp = isinstance(action_space, gym.spaces.Discrete)
        self._action_model = VanillaModel(
            self.obs_input, self.act_input, [50, 50],
            output_activation=nn.Tanh, gaussian=True
        ).to(self.device)
        # 1 input for agent identification. O/P is Gaussian Params
        self._prediction_model = VanillaModel(
            self.obs_input + self.act_input, self.obs_input, [64, 64],
            output_activation=None, gaussian=True
        ).to(self.device)
        # self._action_model = VanillaModel(
        #     self.obs_input, self.act_input, [50, 50],
        #     output_activation=nn.Tanh, gaussian=True
        # ).to(self.device)
        # # 1 input for agent identification. O/P is Gaussian Params
        # self._prediction_model = VanillaModel(
        #     self.obs_input//self.obj2trk + self.act_input + 1, 6, [64, 64],
        #     output_activation=nn.Tanh, gaussian=True
        # ).to(self.device)
        self.action_dist_class = (
            TorchCategorical 
            if self.discrete_ac_sp
            else TorchDiagGaussian
        )
        self.action_dist_class = TorchDiagGaussian
        self._track_id = torch.arange(0, 1, 1/self.obj2trk).view(-1, 1)
        self.random_actions = False
        self._optimizer1 = Adam(self._action_model.parameters(), lr=0.0001)
        self._optimizer2 = Adam(self._prediction_model.parameters(), lr=0.001)

    def convert(self, arr):
        tensor = torch.from_numpy(np.asarray(arr))
        if tensor.dtype == torch.double:
            tensor = tensor.float()
        return tensor.to(self.device)


    def trajectory_loss(self, xs, us):
        txs = torch.stack(xs)[:, :, :6]
        #  or consider only last
        txs = txs[-1]
        pos = torch.chunk(txs, 3, dim=-1)
        dst = torch.stack([torch.norm(p, dim=-1) for p in pos], dim=-1)
        # if not (dst.detach().numpy() > 0).all() or True:
        #     pdb.set_trace()
        min_distance, _ = torch.min(dst, dim=-1)
        # loss = -torch.log(torch.sum(torch.exp(-dst), dim=-1)).mean()
        loss = min_distance.mean()
        return loss

    def train_predictor(self, batch):
        with self.lock:
            self._optimizer2.zero_grad()
            x = self.convert(batch['obs'])
            u = self.convert(batch['actions'])
            x_p1 = self.convert(batch['new_obs'])
            d_p1 = self._prediction_model(
                torch.cat([x, u], dim=-1), return_mode='DIST'
            )
            # pdb.set_trace()
            loss = -d_p1.log_prob(x_p1).sum(dim=-1).mean()
            loss.backward()
            grad_process_info = self.extra_grad_process()
            self._optimizer2.step()
            return loss.detach().item()


    def train_actor(self, batch, num_times=1):
        with self.lock:
            for _ in range(num_times):
                self._optimizer1.zero_grad()
                x = self.convert(batch[0])
                xs, us = self.generate_trajectory(x, seq_len=25)
                loss = self.trajectory_loss(xs, us)
                loss.backward()
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
                        **kwargs):
        with self.lock:
            with torch.no_grad():
                z = self.convert(obs_batch)
                # on trajectory optimization here ?
                xs, us = self.generate_trajectory(z, seq_len=1)
                us = [u.cpu().numpy() for u in us]
                xs = [u.cpu().numpy() for u in xs]
        return (us, xs, {}) if return_states else (us, [], {})

    def generate_trajectory(self, x, seq_len=None):
        seq_len = seq_len or self.seq_len
        xs, us = [], []
        for i in range(seq_len):
            u = self._action_model(x, return_mode='PARAMS', random=self.random_actions)
            d = self.action_dist_class(inputs=torch.cat(u, dim=-1), model=self._action_model)
            a = d.rsample()
            x = self._prediction_model(
                torch.cat([x, a], dim=-1), return_mode='DETERMINISTIC'
            )
            us.append(a.squeeze().clone())
            xs.append(x.squeeze().clone())
        return xs, us

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

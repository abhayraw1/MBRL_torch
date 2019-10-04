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
from ray.rllib.models.torch.torch_action_dist import (TorchMultinomial,
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
        self.seq_len = self.config.get('seq_len', 20)
        self.lock = Lock()
        self.device = torch.device("cpu")
        self.obj2trk = self.config.get('objects_to_track', 1)
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_input = MBRLPolicy.size(self.observation_space)
        self.act_input = MBRLPolicy.size(self.action_space)
        self.discrete_ac_sp = isinstance(action_space, gym.spaces.Discrete)
        self._action_model = VanillaModel(
            self.obs_input, self.act_input, [50, 50], gaussian=True
        ).to(self.device)
        # 1 input for agent identification. O/P is Gaussian Params
        self._prediction_model = VanillaModel(
            self.obs_input + self.act_input, self.obs_input, [64, 64],
            gaussian=True
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
        # self.action_dist_class = (
        #     TorchCategorical 
        #     if self.discrete_ac_sp
        #     else TorchDiagGaussian
        # )
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
        self._optimizer1 = Adam(self._action_model.parameters(), lr=0.00005)
        self._optimizer2 = Adam(self._prediction_model.parameters(), lr=0.01)
        self.discount = 0.99**torch.arange(self.seq_len-1, -1, -1)[None] #only consider last timestep
        self.discount = self.discount.transpose(1, 0).float()


    def convert(self, arr):
        tensor = torch.from_numpy(np.asarray(arr))
        if tensor.dtype == torch.double:
            tensor = tensor.float()
        return tensor.to(self.device)


    def trajectory_loss(self, xs, us, lp):
        txs = torch.stack(xs)
        uss = torch.stack(us)
        lps = torch.stack(lp)
        c, s, d = txs.transpose(0, -1).transpose(-1, 1)
        costs = []
        losses = []
        loss = 0
        th = torch.atan2(s, c)
        for i in range(self.seq_len):
            e = self.seq_len - i - 1
            cost = (th[e])**2# + 0.1*d[e]**2 + 0.001*uss[e]**2
            if i == 0:
                costs.insert(0, cost)
                continue
            costs.insert(0, cost + costs[0])
            losses.append(lp[e]*cost)
            loss -= lp[e]*cost
        # pdb.set_trace()
        # # pdb.set_trace()
        # cost = -(th**2 + 0.1*d**2 + 0.001*uss**2)*lps

        return loss.mean()/10
        # return cost.sum(dim=0).mean()

    # def trajectory_loss(self, xs, us, lp):
    #     txs = torch.stack(xs)
    #     lps = torch.stack(lp)
    #     #  or consider only last
    #     # txs = txs[-1]
    #     pos = torch.chunk(txs, 3, dim=-1)
    #     dst = torch.stack([torch.norm(p, dim=-1) for p in pos], dim=-1)
    #     # if not (dst.detach().numpy() > 0).all() or True:
    #     min_distance, _ = torch.min(dst, dim=-1)
    #     # print(min_distance.shape)
    #     # pdb.set_trace()
    #     loss = min_distance*(-lps*self.discount)
    #     # loss = min_distance.mean()
    #     # loss = 
    #     # loss = -torch.log(torch.sum(torch.exp(-dst), dim=-1)).mean()
    #     return loss.sum(dim=0).mean()

    def eval_predictor(self, batch):
        x = self.convert(batch['obs'])
        u = self.convert(batch['actions']).view(-1, self.act_input)
        x_p1 = self.convert(batch['new_obs'])
        d_p1 = x_p1 - x
        # pdb.set_trace()
        y_p1 = self._prediction_model(
            torch.cat([x, u], dim=-1), return_mode='DIST'
        )
        loss = -y_p1.log_prob(d_p1).sum(dim=-1).mean()
        mse = ((d_p1 - y_p1.mean)**2).sum(dim=-1).mean().detach().item()
        mae = ((d_p1 - y_p1.mean).abs()).sum(dim=-1).mean().detach().item()
        return loss, mse, mae

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
            with torch.no_grad():
                keys = ['obs', 'actions', 'reward', 'new_obs', 'done']
                datch = {k: self.convert(v) for k, v in zip(keys, batch)}
                pl = self.eval_predictor(datch)
            print('PRED LOSS ===--> ', pl)
            x = self.convert(batch[0])
            for _ in range(num_times):
                self._optimizer1.zero_grad()
                xs, us, lp = self.generate_trajectory(
                    x, seq_len=self.seq_len, return_log_probs=True
                )
                loss = self.trajectory_loss(xs, us, lp)
                pdb.set_trace()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self._action_model.parameters(), 5, norm_type=2
                )
                # self._action_model.print_grads()
                self._optimizer1.step()
                print(f'loss: {loss.detach().item()}')
            return loss.detach().item()
# make_dot(lp[2].sum(), self._all_params).render('lp2')

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
                # on trajectory optimization here ?
                xs, us = self.generate_trajectory(z, seq_len=1, mode=eval_mode)
                us = [u.cpu().numpy() for u in us]
                xs = [u.cpu().numpy() for u in xs]
        return (us, xs, {}) if return_states else (us, [], {})

    def generate_trajectory(self,
                            x,
                            seq_len=None,
                            return_log_probs=False,
                            mode='T'):
        seq_len = seq_len or self.seq_len
        xs, us, lp = [], [], []
        return_mode = 'DETERMINISTIC' if mode == 'E' else 'PARAMS'
        for i in range(seq_len):
            u = self._action_model(x, return_mode=return_mode)
            # pdb.set_trace()
            if return_mode == 'PARAMS':
                u = torch.cat(u, dim=-1)
            d = self.action_dist_class(inputs=u, model=self._action_model)
            a = d.sample().float().clamp(
                    self.action_space.low[0],
                    self.action_space.high[0]
                )
            x = x + self._prediction_model(
                torch.cat([x, a], dim=-1), return_mode='DETERMINISTIC'
            ).detach()
            # pdb.set_trace()
            us.append(a.squeeze().clone())
            xs.append(x.squeeze().clone())
            if return_log_probs:
                lp.append(d.logp(a))
        if return_log_probs:
            return xs, us, lp
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

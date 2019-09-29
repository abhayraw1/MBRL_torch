"""Example of using rollout worker classes directly to implement training.

Instead of using the built-in Trainer classes provided by RLlib, here we define
a custom Policy class and manually coordinate distributed sample
collection and policy optimization.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import gym
import time
import pathlib
import argparse

import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.evaluation import RolloutWorker, SampleBatch
from ray.rllib.optimizers.replay_buffer import ReplayBuffer
from mbrl_ray_policy import MBRLPolicy


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--num-iters", type=int, default=9000)
parser.add_argument("--num-workers", type=int, default=10)


def copy_local2remote(policy, workers, a_model=True, p_model=True):
    if isinstance(policy, dict):
        weights = ray.put({
            pid: v.get_weights() for pid, v in policy.items()
        })
    else:
        weights = ray.put({"default_policy": policy.get_weights()})
    for w in workers:
        w.set_weights.remote(
            weights, {'action_model': a_model, 'prediction_model': p_model}
        )


def get_make_env(scenario_name, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    from ray.rllib.env.multi_agent_env import MultiAgentEnv as MAE_RAY

    class ray_mae_cls(MultiAgentEnv, MAE_RAY):
        def __init__(self, *args, **kwargs):
            MultiAgentEnv.__init__(self, *args, **kwargs)
    
    def  make_env(c = None, return_ids=False, return_base=False):
        if benchmark:        
            base = ray_mae_cls(
                world,
                scenario.reset_world,
                scenario.reward,
                scenario.observation,
                scenario.benchmark_data
            )
        else:
            base = ray_mae_cls(
                world,
                scenario.reset_world,
                scenario.reward,
                scenario.observation
            )

        agent_ids = list(base.agents.keys())
        env = BaseEnv.to_base_env(base)
        rets = [env]
        if return_ids:
            rets.append(agent_ids)
        if return_base:
            rets.append(base)
        return rets if len(rets) > 1 else rets[0]
    return make_env


def training_workflow(config, reporter):
    # Setup policy and policy evaluation actors
    # env_maker = lambda c: BaseEnv.to_base_env(gym.make("Pendulum-v0"))
    env_maker = get_make_env('simple_spread')
    env, agent_ids, base = env_maker(None, return_ids=True, return_base=True)
    config['buffer_size'] = 100000
    config['objects_to_track'] = 1

    ############################ DEFINE POLICY STUFF ###########################
    policy = {}
    replay_buffers = {}

    for agent_id in agent_ids:
        obs = base.observation_space[agent_id]
        act = base.action_space[agent_id]
        policy[agent_id] = MBRLPolicy(obs, act, config)
        replay_buffers[agent_id] = ReplayBuffer(config['buffer_size'])

    policy_map = lambda agent_id: agent_id

    ############################## REMOTE WORKERS ##############################
    remote_workers = []
    for _ in range(config["num_workers"]):
        worker = RolloutWorker.as_remote().remote(
            env_maker,
            policy={
                agent_id: (MBRLPolicy,
                    base.observation_space[agent_id],
                    base.action_space[agent_id],
                    config)
                for agent_id in agent_ids
            },
            policy_mapping_fn=policy_map,
            episode_horizon=30,
            monitor_path='/home/aarg/Documents/mbrl_torch_g2g/monitor'
        )
        remote_workers.append(worker)

    copy_local2remote(policy, remote_workers)
    actor_loss = {}
    predictor_loss = {}
    for i in range(config["num_iters"]):
        # braodcast only the belief from the local policy to the remote one
        copy_local2remote(policy, remote_workers, a_model=False)

        # Gather a batch of samples
        T1 = SampleBatch.concat_samples(
            ray.get([w.sample.remote() for w in remote_workers])
        )

        # Train the belief network
        for agent_id, batch in T1.policy_batches.items():
            # Add samples to Replay Buffer
            for row in batch.rows():
                replay_buffers[agent_id].add(
                    row["obs"], row["actions"], row["rewards"],
                    row["new_obs"], row["dones"], weight=None
                )
            loss = policy[agent_id].train_predictor(batch)
            predictor_loss[agent_id] = loss

        # Train the actor network
        if (i + 1) % 15 == 0:
            actor_loss = {}
            for agent_id, rbuffer in replay_buffers.items():
                batch = rbuffer.sample(64)
                for _ in range(10):
                    loss = policy[agent_id].train_actor(batch)
                actor_loss[agent_id] = loss
            # Train the action Model (off-policy?)
            # broadcast the action model to the remote workers.
            copy_local2remote(policy, remote_workers, p_model=False)
        info = collect_metrics(remote_workers=remote_workers)
        # pdb.set_trace()
        info['predictor_loss'] = predictor_loss
        info['actor_loss'] = actor_loss
        if (i + 1) % 30 == 0:
            for agent_id, pi in policy.items():
                __id = time.strftime("%d%m%Y_%H%M", time.localtime())
                path = '/home/aarg/Documents/mbrl_torch_g2g/models/mbrl'
                path = f'{path}/{__id}/{agent_id}'
                pathlib.Path(path).mkdir(parents=True, exist_ok=True)
                pi.save_models(path)
        reporter(**info)


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    tune.run(
        training_workflow,
        resources_per_trial={
            "gpu": 1 if args.gpu else 0,
            "cpu": 1,
            "extra_cpu": args.num_workers,
        },
        config={
            "num_workers": args.num_workers,
            "num_iters": args.num_iters,
        },
    )
    pdb.set_trace()


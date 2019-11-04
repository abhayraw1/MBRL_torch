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
import logging
import argparse

import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.evaluation import RolloutWorker, SampleBatch
from ray.rllib.optimizers.replay_buffer import ReplayBuffer
from mbrl_ray_policy import MBRLPolicy


from gym.wrappers.monitoring.video_recorder import VideoRecorder

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--num-iters", type=int, default=9000)
parser.add_argument("--num-workers", type=int, default=10)
parser.add_argument("--num-eval-eps", type=int, default=5)
parser.add_argument("--eval-horizon", type=int, default=45)


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
    from gym.wrappers import Monitor
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    from ray.rllib.env.multi_agent_env import MultiAgentEnv as MAE_RAY

    class ray_mae_cls(MultiAgentEnv, MAE_RAY):
        def __init__(self, *args, **kwargs):
            MultiAgentEnv.__init__(self, *args, **kwargs)
    
    def  make_env(c=None, return_ids=False, return_base=False):
        if benchmark:        
            base = ray_mae_cls(
                world,
                scenario.reset_world,
                scenario.reward,
                scenario.observation,
                scenario.benchmark_data,
            )
        else:
            base = ray_mae_cls(
                world,
                scenario.reset_world,
                scenario.reward,
                scenario.observation,
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


def get_video_rec_callbacks(vr):
    def on_episode_start(data):
        if not vr.enabled:
            vr.enabled = True

    def on_episode_step(data):
        vr.capture_frame()

    def on_episode_end(data):
        pass

    def on_sample_end(data):
        vr.close()

    callbacks = {
        "on_episode_start": on_episode_start,
        "on_episode_step": on_episode_step,
        "on_episode_end": on_episode_end,
        "on_sample_end": on_sample_end,
    }
    return callbacks


def training_workflow(config, reporter):
    env_maker = get_make_env('simple_spread')
    env, agent_ids, base = env_maker(return_ids=True, return_base=True)
    config['buffer_size'] = 100000
    config['objects_to_track'] = 1
    config['seq_len'] = 25
    FLAG_MAKE_FRESH_DIR = True
    last_iter_saved = 0

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
            episode_horizon=30
        )
        remote_workers.append(worker)

    ################################ EVAL WORKER ###############################
    eval_worker = RolloutWorker(
        lambda x: base,
        policy={
            agent_id: (MBRLPolicy,
                base.observation_space[agent_id],
                base.action_space[agent_id],
                config)
            for agent_id in agent_ids
        },
        policy_mapping_fn=policy_map,
        episode_horizon=config['eval_horizon'],
        batch_steps=config['eval_horizon']*config['num_eval_eps'],
    )
    pdb.set_trace()
    eval_worker.policy_map = policy
    ########################### VIDEO RECORDER SETUP ###########################
    video_recorder = VideoRecorder(
        env=eval_worker.sampler.base_env.get_unwrapped()[0],
        path='/home/aarg/Documents/mbrl_torch_g2g/monitor/vid.mp4',
    )
    recorder_callbacks = get_video_rec_callbacks(video_recorder)
    eval_worker.callbacks.update(recorder_callbacks)
    eval_worker.sampler.callbacks.update(recorder_callbacks)
    video_dir = pathlib.Path(video_recorder.path).parent

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
        if (i - last_iter_saved)/30 > 0:
            last_iter_saved = i
            for agent_id, pi in policy.items():
                date = time.strftime("%d%m%Y", time.localtime())
                __id = time.strftime("%H%M", time.localtime())
                path = '/home/aarg/Documents/mbrl_torch_g2g/models/mbrl'
                c = 1
                while os.path.exists(f'{path}/{date}') and FLAG_MAKE_FRESH_DIR:
                    date = time.strftime("%d%m%Y", time.localtime()) + f'_{c}'
                    c += 1
                FLAG_MAKE_FRESH_DIR = False
                path = f'{path}/{date}/{__id}/{agent_id}'
                pathlib.Path(path).mkdir(parents=True, exist_ok=True)
                pi.save_models(path)
        if (i + 1) % 100 == 0:        
            video_recorder.path = str(video_dir/f'EvalRollout_{i+1}.mp4')
            eval_worker.sample()
        reporter(**info)


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(logging_level=logging.INFO)

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
            'num_eval_eps': args.num_eval_eps,
            'eval_horizon': args.eval_horizon
        },
        verbose=2,
        loggers=None,
    )
    pdb.set_trace()


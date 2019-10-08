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
import numpy as np

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
parser.add_argument("--eval-horizon", type=int, default=100)

np.set_printoptions(suppress=True, linewidth=300, precision=4,
                    formatter={'float_kind':'{:10.6f}'.format})


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
    env_maker = lambda c: gym.make('Pendulum-v0')
    base = env_maker(None)
    config['buffer_size'] = 10000
    config['seq_len'] = 25
    FLAG_MAKE_FRESH_DIR = True
    last_iter_saved = 0
    last_iter_train = 0

    ############################ DEFINE POLICY STUFF ###########################
    obs = base.observation_space
    act = base.action_space
    policy = MBRLPolicy(obs, act, config)
    replay_buffer = ReplayBuffer(config['buffer_size'])

    ############################## REMOTE WORKERS ##############################
    remote_workers = []
    for _ in range(config["num_workers"]):
        worker = RolloutWorker.as_remote().remote(
            env_maker,
            policy=MBRLPolicy,
            episode_horizon=100,
        )
        remote_workers.append(worker)

    ################################ EVAL WORKER ###############################
    eval_worker = RolloutWorker(
        lambda x: base,
        policy=MBRLPolicy,
        episode_horizon=config['eval_horizon'],
        batch_steps=config['eval_horizon']*config['num_eval_eps'],
        eval_mode=True,
    )
    eval_worker.policy_map['default_policy'] = policy
    ########################### VIDEO RECORDER SETUP ###########################
    video_recorder = VideoRecorder(
        env=eval_worker.sampler.base_env.get_unwrapped()[0],
        path='/home/aarg/Documents/mbrl_torch_g2g/monitor/pendulum/vid.mp4',
    )
    recorder_callbacks = get_video_rec_callbacks(video_recorder)
    eval_worker.callbacks.update(recorder_callbacks)
    eval_worker.sampler.callbacks.update(recorder_callbacks)
    video_dir = pathlib.Path(video_recorder.path).parent

    copy_local2remote(policy, remote_workers)

    actor_loss = 'None'
    for i in range(config["num_iters"]):
        # braodcast only the belief from the local policy to the remote one
        copy_local2remote(policy, remote_workers, a_model=False)

        # Gather a batch of samples
        batch = SampleBatch.concat_samples(
            ray.get([w.sample.remote() for w in remote_workers])
        )

        # pdb.set_trace()
        # Train the belief network
        # Add samples to Replay Buffer
        actions = []
        for row in batch.rows():
            replay_buffer.add(
                row["obs"], row["actions"], row["rewards"],
                row["new_obs"], row["dones"], weight=None
            )
            actions.append(row['actions'])
        for _ in range(10):
            batch = replay_buffer.sample(128, return_dict=True)
            predictor_loss = policy.train_predictor(batch)

        # Train the actor network
        if (i - last_iter_train)//15 > 0 and predictor_loss['MAE'] < 0.05:
            last_iter_train = i
            # Train the action Model (off-policy?)
            for _ in range(10):
                batch = replay_buffer.sample(128)
                loss = policy.train_actor(batch, 30)
                # print('-'*100)
            actor_loss = loss
            # broadcast the action model to the remote workers.
            copy_local2remote(policy, remote_workers, p_model=False)

        info = collect_metrics(remote_workers=remote_workers)
        info['actions'] = {
            'mean': np.mean(actions),
            'std': np.std(actions)
        }
        info['predictor_loss'] = predictor_loss
        info['actor_loss'] = actor_loss
        # if (i - last_iter_saved)/30 > 0:
        #     last_iter_saved = i
        #     for agent_id, pi in policy.items():
        #         date = time.strftime("%d%m%Y", time.localtime())
        #         __id = time.strftime("%H%M", time.localtime())
        #         path = '/home/aarg/Documents/mbrl_torch_g2g/models/mbrl'
        #         c = 1
        #         while os.path.exists(f'{path}/{date}') and FLAG_MAKE_FRESH_DIR:
        #             date = time.strftime("%d%m%Y", time.localtime()) + f'_{c}'
        #             c += 1
        #         FLAG_MAKE_FRESH_DIR = False
        #         path = f'{path}/{date}/{__id}/{agent_id}'
        #         pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        #         pi.save_models(path)
        if (i + 1) % 100 == 0:        
            video_recorder.path = str(video_dir/f'PendulumEvalRollout_{i+1}.mp4')
            eval_worker.sample()
            # pdb.set_trace()
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
            'eval_horizon': args.eval_horizon,
            'device': 'cuda' if args.gpu else 'cpu'
        },
        verbose=2,
        loggers=None,
    )
    pdb.set_trace()


import gym
from numpy import *
from gym import Spaces


class HolonomicEnv(gym.Env):
    def __init__(self, config=None):
        self.dt = 1e-2
        self.num_iter = 50
        self.max_episode_steps = 25
        self._max_episode_steps = 25

        self.action_low = np.array([0.0, -np.pi/4])
        self.action_high = np.array([0.3, np.pi/4])
        self.action_space = Box(self.action_low, self.action_high, dtype="f")
        self.observation_space = Box(low=-1, high=1, shape=(5,), dtype="f")

        # so that the goals are within the range of performing actions
        self.d_clip = self.action_high[0]*self.num_iter*self.dt*1.35


        self.limits = np.array([1, 1, np.pi])
        self.thresh = np.array([0.05, 0.05, 0.1])
        self.agent = Agent(0)
        if config is not None:
            self.__dict__.update(config)

        # if self.seed is not None:
        #     np.random.seed(self.seed)

        self.goal = None
        if not self.her:
            self.dMax = self.action_high[0]*self.dt*self.num_iter
            self.dRange = 2*self.dMax
        self.viewer = None
        self.spec = EnvSpec("Go2Goal-v0", max_episode_steps=self._max_episode_steps)
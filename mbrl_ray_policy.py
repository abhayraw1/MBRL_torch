from ray.rllib.policy.torch_policy import TorchPolicy
from ray_models import DynamixForward
class MBRLPolicy(TorchPolicy):
    """Model Predictive Policy for an Agent in multi-agent scenario.

    You might find it more convenient to extend TF/TorchPolicy instead
    for a real policy.
    """

    def __init__(self, observation_space, action_space, config):
        self.observation_space = observation_space
        self.action_space = action_space
        self.lock = Lock()
        self.device = config['device'] or (torch.device("cuda")
           if bool(os.environ.get("CUDA_VISIBLE_DEVICES", None))
           else torch.device("cpu"))
        self._action_model = MBRL.to(self.device)
        self._prediction_model = MBRL.to(self.device)
        self._dynamix_model = loss
        self._optimizer = self.optimizer()
        self._action_dist_class = action_distribution_class

    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        # return random actions
        return [self.action_space.sample() for _ in obs_batch], [], {}

    def learn_on_batch(self, samples):
        # implement your learning code here
        return {}

    def update_some_value(self, w):
        # can also call other methods on policies
        self.w = w

    def get_weights(self):
        return {"w": self.w}

    def set_weights(self, weights):
        self.w = weights["w"]
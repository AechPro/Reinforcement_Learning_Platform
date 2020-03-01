from Agents import Agent
import torch
from torch.distributions import Categorical
import numpy as np
class VanillaPolicyGradientsAgent(Agent):
    def __init__(self, config):
        super().__init__(config)
        self.noise_power = config["policy"]["action_noise_std"]

    def prepare_policy_input(self, obs_buffer, buffer_shape):
        return np.reshape(obs_buffer, buffer_shape)

    def get_action(self, policy, state, episode_data, obs_stats=None):
        obs = torch.as_tensor(state, dtype=torch.float32)
        action = policy.get_action(obs)
        return action

    def cleanup(self):
        pass
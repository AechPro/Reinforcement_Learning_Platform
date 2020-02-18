from Agents import Agent
import torch
from torch.distributions import Categorical
import numpy as np
class VanillaPolicyGradientsAgent(Agent):
    def __init__(self, config):
        super().__init__(config)

    def prepare_policy_input(self, obs_buffer, buffer_shape):
        return np.reshape(obs_buffer, buffer_shape)

    def get_action(self, policy, state, episode_data, obs_stats=None):
        obs = torch.as_tensor(state, dtype=torch.float32)
        policy_output = policy.model(obs)
        categorical_output = Categorical(probs=policy_output)
        action = categorical_output.sample()

        return action.item()

    def cleanup(self):
        pass
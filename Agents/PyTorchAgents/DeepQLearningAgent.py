from Agents import Agent
import torch
from torch.distributions import Categorical
import numpy as np

class BasicPolicyGradientsAgent(Agent):
    def __init__(self, config):
        super().__init__(config)
        self.eps_end = 0.05
        self.eps_start = 0.9
        self.iteration = 0
        self.eps_decay = 50


    def get_action(self, policy, state, obs_stats=None):
        num_actions = np.prod(policy.output_shape)
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1. * self.iteration / self.eps_decay)

        if self.cfg["rng"].uniform(0, 1) < epsilon:
            best_action = self.cfg["rng"].randint(0, num_actions)
        else:
            best_action = 0
            best_score = -np.inf
            for action in num_actions:
                input_state = np.concatenate((state, (action,)))
                input_batch = np.reshape(input_state, (1, len(input_state)))
                policy_output = policy.get_action(input_batch)
                action_quality = policy_output.max().item()

                if action_quality > best_score:
                    best_action = action
                    best_score = action_quality

        self.iteration += 1
        return best_action

    def cleanup(self):
        pass
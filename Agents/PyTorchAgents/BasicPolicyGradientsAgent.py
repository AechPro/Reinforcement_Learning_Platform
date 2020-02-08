from Agents import Agent
import torch
from torch.distributions import Categorical
class BasicPolicyGradientsAgent(Agent):
    def __init__(self, config):
        super().__init__(config)

    def get_action(self, policy, state, obs_stats=None):
        obs = torch.as_tensor(state, dtype=torch.float32)
        policy_output = policy.forward(obs)
        categorical_output = Categorical(policy_output)
        action = categorical_output.sample().item()
        return action

    def cleanup(self):
        pass
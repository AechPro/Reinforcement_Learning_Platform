from Agents import AgentFactory
from Environments import EnvironmentFactory
from Policies import PolicyFactory

import torch
from torch.distributions import Categorical

class Optimizer(object):
    def __init__(self, config):
        self.cfg = config
        self.agent = None
        self.env = None
        self.policy = None
        self.optimizer = None

    def configure(self):
        self.agent = AgentFactory.get_from_config(self.cfg)
        self.env = EnvironmentFactory.get_from_config(self.cfg)
        self.policy = PolicyFactory.get_from_config(self.cfg)
        self.optimizer = torch.optim.Adam(self.policy.model.parameters(), lr=1e-2)

    def reconfigure(self):
        self.cleanup()
        self.configure()

    def step(self):
        episode_data = self.agent.run_training_episode(self.policy, self.env)

        obs = torch.as_tensor(episode_data.observations, dtype=torch.float32)
        acts = torch.as_tensor(episode_data.actions, dtype=torch.int32)
        rewards = torch.as_tensor(episode_data.rewards, dtype=torch.float32)

        self.optimizer.zero_grad()
        output = self.policy.activate(obs)
        log_probs = Categorical(output).log_prob(acts)
        loss = -(log_probs * rewards)
        loss.backward()
        self.optimizer.step()

        print(loss)

    def is_done(self):
        return False

    def set_base_dir(self, base_dir):
        pass

    def cleanup(self):
        if self.agent is not None:
            self.agent.cleanup()
        if self.env is not None:
            self.env.cleanup()
        if self.policy is not None:
            self.policy.cleanup()

        del self.agent
        del self.env
        del self.policy
        del self.optimizer
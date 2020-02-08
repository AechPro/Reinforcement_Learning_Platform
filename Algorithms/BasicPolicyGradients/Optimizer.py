from Agents import AgentFactory
from Environments import EnvironmentFactory
from Policies import PolicyFactory, PolicyActionParsers

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

        policy = PolicyFactory.get_from_config(self.cfg)
        action_parser = PolicyActionParsers.linear_parse
        self.policy = policy(self.env.observation_shape, self.env.action_shape, action_parser, self.cfg)
        self.policy.build_model(self.cfg["policy"])

        self.optimizer = torch.optim.Adam(self.policy.model.parameters(), lr=1e-2)

    def reconfigure(self):
        self.cleanup()
        self.configure()

    def step(self):
        batch_size = 32
        self.policy.model.train()
        with torch.no_grad():
            rewards = []
            observations = []
            actions = []

            for i in range(batch_size):
                episode_data = self.agent.run_training_episode(self.policy, self.env)
                episode_data.compute_future_rewards(0.99)

                episode_reward = sum(episode_data.rewards)
                for j in range(len(episode_data.rewards)):
                    actions.append(episode_data.actions[j])
                    rewards.append(episode_reward)
                    observations.append(episode_data.observations[j])

        obs = torch.as_tensor(observations, dtype=torch.float32)
        acts = torch.as_tensor(actions, dtype=torch.int32)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)

        self.optimizer.zero_grad()
        output = self.policy.model(obs)
        log_probs = Categorical(logits=output).log_prob(acts)
        loss = -(log_probs * rewards).mean()
        loss.backward()
        self.optimizer.step()


        policy_reward = 0
        for i in range(10):
            policy_reward += self.agent.run_benchmark_episode(self.policy, self.env)
        policy_reward/=10

        print(loss.item()," | ", policy_reward)

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
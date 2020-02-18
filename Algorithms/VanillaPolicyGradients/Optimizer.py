from Agents import AgentFactory
from Environments import EnvironmentFactory
from Policies import PolicyFactory, PolicyActionParsers
from Util import ExperienceReplay
import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as f


class Optimizer(object):
    def __init__(self, config):
        self.cfg = config
        self.agent = None
        self.env = None
        self.value_network = None
        self.value_optimizer = None
        self.policy = None
        self.policy_optimizer = None
        self.experience_replay = None

        self.batch_size = 300
        self.gamma = self.cfg["policy_optimizer"]["gamma"]
        self.lmbda = 0.97
        self.epoch = 0

    def configure(self):
        self.agent = AgentFactory.get_from_config(self.cfg)
        self.env = EnvironmentFactory.get_from_config(self.cfg)

        models, action_parsers = PolicyFactory.get_from_config(self.cfg)

        value_input_shape = self.env.get_policy_input_shape()
        policy_input_shape = self.env.get_policy_input_shape()
        policy_output_shape = self.env.get_policy_output_shape()

        self.value_network = models["value_estimator"](value_input_shape, 1, action_parsers["value_estimator"], self.cfg, "value_estimator")
        self.value_network.build_model(self.cfg["value_estimator"])
        self.value_optimizer = torch.optim.RMSprop(self.value_network.model.parameters())

        self.policy = models["policy"](policy_input_shape, policy_output_shape, action_parsers["policy"], self.cfg, "policy")
        self.policy.build_model(self.cfg["policy"])
        self.policy_optimizer = torch.optim.Adam(self.policy.model.parameters(), lr=1e-2)

        self.experience_replay = ExperienceReplay(self.cfg)

    def reconfigure(self):
        self.cleanup()
        self.configure()

    def step(self):
        self.experience_replay.clear()

        with torch.no_grad():
            for i in range(self.batch_size+1):
                episode_data = self.agent.run_training_episode(self.policy, self.env)
                self.experience_replay.register_episode(episode_data)

        self.update_policy()
        for i in range(100):
            loss = self.update_value_estimator()

        policy_reward = 0
        for i in range(10):
            policy_reward += self.agent.run_benchmark_episode(self.policy, self.env)
        policy_reward /= 10

        print(self.epoch, "|", policy_reward)
        self.epoch += 1

    def compute_advantages(self, rewards, observations):
        residuals = []
        advantages = []
        value_estimations = self.value_network.model(observations).view(rewards.shape)

        for i in range(len(rewards)-1):
            res = rewards[i] + self.gamma*value_estimations[i+1] - value_estimations[i]
            residuals.append(res)
        residuals.append(rewards[-1])

        for i in range(len(residuals)):
            adv = 0
            for j in range(0, len(residuals)-i):
                coeff = np.power(self.lmbda*self.gamma, j)
                adv += coeff * residuals[i+j]
            advantages.append(adv)

        return torch.as_tensor(advantages, dtype=torch.float32)


    def update_policy(self):
        batch = self.experience_replay.get_batch(self.batch_size, as_columns=True)

        observations = torch.as_tensor(batch[ExperienceReplay.OBSERVATION_IDX], dtype=torch.float32)
        rewards = torch.as_tensor(batch[ExperienceReplay.FUTURE_REWARD_IDX], dtype=torch.float32)
        acts = torch.as_tensor(batch[ExperienceReplay.ACTION_IDX], dtype=torch.int32)

        advantages = self.compute_advantages(rewards, observations)

        self.policy_optimizer.zero_grad()
        output = self.policy.model(observations)
        log_probs = Categorical(probs=output).log_prob(acts)
        loss = -(log_probs * advantages).mean()
        loss.backward()
        self.policy_optimizer.step()

    def update_value_estimator(self):
        batch = self.experience_replay.get_batch(self.batch_size, as_columns=True)

        observations = torch.as_tensor(batch[ExperienceReplay.OBSERVATION_IDX], dtype=torch.float32)
        rewards = torch.as_tensor(batch[ExperienceReplay.FUTURE_REWARD_IDX], dtype=torch.float32)
        value_estimations = self.value_network.model(observations).view(rewards.shape)

        self.value_optimizer.zero_grad()
        loss = f.smooth_l1_loss(value_estimations, rewards)
        loss.backward()
        self.value_optimizer.step()

        return loss.item()

    def is_done(self):
        return False

    def set_base_dir(self, base_dir):
        pass

    def cleanup(self):
        if self.agent is not None:
            self.agent.cleanup()
        if self.env is not None:
            self.env.cleanup()
        if self.value_network is not None:
            self.value_network.cleanup()

        del self.agent
        del self.env
        del self.value_network
        del self.value_optimizer
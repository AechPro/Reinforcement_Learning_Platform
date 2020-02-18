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

        self.batch_size = 1
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

        with torch.no_grad():
            rewards = []
            observations = []
            actions = []
            values = []
            advantages = []

            for i in range(self.batch_size):
                episode_data = self.agent.run_training_episode(self.policy, self.env)
                episode_data.compute_future_rewards(self.gamma)

                actions += episode_data.actions
                rewards += episode_data.future_rewards
                observations += episode_data.observations
                values += self.value_network.model(torch.as_tensor(observations, dtype=torch.float32))

                episode_data.compute_td_residuals(values, self.gamma)
                episode_data.compute_general_advantage_estimation(self.gamma, self.lmbda)
                advantages += episode_data.advantages

            obs = torch.as_tensor(observations, dtype=torch.float32)
            acts = torch.as_tensor(actions, dtype=torch.int32)

            rewards = torch.as_tensor(rewards, dtype=torch.float32)
            advantages = torch.as_tensor(advantages, dtype=torch.float32)

        self.update_policy(obs,acts,advantages)
        loss1 = self.update_value_estimator(obs, rewards)
        for i in range(200):
            loss = self.update_value_estimator(obs,rewards)
        loss2 = self.update_value_estimator(obs, rewards)
        print("loss difference:",loss1-loss2)

        policy_reward = 0
        for i in range(10):
            policy_reward += self.agent.run_benchmark_episode(self.policy, self.env)
        policy_reward /= 10

        print(self.epoch, "|", policy_reward)
        self.epoch += 1

    def update_policy(self, observations, acts, advantages):

        print("update policy")
        self.policy_optimizer.zero_grad()
        output = self.policy.model(observations)
        log_probs = Categorical(probs=output).log_prob(acts)
        loss = -(log_probs * advantages).mean()
        loss.backward()
        self.policy_optimizer.step()

    def update_value_estimator(self, observations, rewards):
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
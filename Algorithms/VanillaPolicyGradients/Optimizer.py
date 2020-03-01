from Agents import AgentFactory
from Environments import EnvironmentFactory
from Policies import PolicyFactory, PolicyActionParsers
from Util import ExperienceReplay
import numpy as np
import torch
from torch.distributions import Categorical
from torch.distributions.normal import Normal
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

        self.batch_size = 32
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
        self.value_optimizer = torch.optim.Adam(self.value_network.model.parameters(), lr=1e-3)

        self.policy = models["policy"](policy_input_shape, policy_output_shape, action_parsers["policy"], self.cfg, "policy")
        self.policy.build_model(self.cfg["policy"])
        self.policy_optimizer = torch.optim.Adam(self.policy.model.parameters(), lr=0.05)

        self.experience_replay = ExperienceReplay(self.cfg)


    def reconfigure(self):
        self.cleanup()
        self.configure()

    def step(self):
        with torch.no_grad():
            for i in range(self.batch_size):
                episode_data = self.agent.run_training_episode(self.policy, self.env)
                episode_data.compute_future_rewards(self.gamma)

                values = self.value_network.model(torch.as_tensor(episode_data.observations, dtype=torch.float32))
                episode_data.compute_td_residuals(values, self.gamma)
                episode_data.compute_general_advantage_estimation(self.gamma, self.lmbda)

            self.experience_replay.register_episode(episode_data, compute_future_returns=False)

        self.update_policy()

        loss1 = self.update_value_estimator()
        while True:
            for i in range(10):
                loss = self.update_value_estimator()
            loss2 = self.update_value_estimator()
            print("loss difference:",loss1-loss2)
            if loss1-loss2>0:
                break

        policy_reward = 0
        for i in range(10):
            policy_reward += self.agent.run_benchmark_episode(self.policy, self.env)
        policy_reward /= 10

        print(self.epoch, "|", policy_reward)
        self.epoch += 1

    def update_policy(self):
        batch = self.experience_replay.get_random_batch(500, as_columns=True)

        observations = torch.as_tensor(batch[ExperienceReplay.OBSERVATION_IDX], dtype=torch.float32)
        advantages = torch.as_tensor(batch[ExperienceReplay.ADVANTAGE_IDX], dtype=torch.float32)
        actions = torch.as_tensor(batch[ExperienceReplay.ACTION_IDX], dtype=torch.int32)

        advantages = (advantages - advantages.mean())/advantages.std()

        self.policy_optimizer.zero_grad()

        policy_output = Categorical(probs=self.policy.model(observations))
        log_probs = policy_output.log_prob(actions)
        loss = (log_probs * advantages).mean()
        loss = -loss
        loss.backward()
        self.policy_optimizer.step()

    def update_value_estimator(self):
        batch = self.experience_replay.get_random_batch(500, as_columns=True)
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
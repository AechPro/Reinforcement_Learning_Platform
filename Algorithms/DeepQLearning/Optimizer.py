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
        self.quality_network = None
        self.optimizer = None
        self.experience_replay = None
        self.target_quality_network = None

    def configure(self):
        self.agent = AgentFactory.get_from_config(self.cfg)
        self.env = EnvironmentFactory.get_from_config(self.cfg)

        policy = PolicyFactory.get_from_config(self.cfg)
        action_parser = PolicyActionParsers.linear_parse
        self.quality_network = policy(self.env.observation_shape, 1, action_parser, self.cfg)
        self.quality_network.build_model(self.cfg["policy"])

        policy = PolicyFactory.get_from_config(self.cfg)
        action_parser = PolicyActionParsers.linear_parse
        self.target_quality_network = policy(self.env.observation_shape, 1, action_parser, self.cfg)
        self.target_quality_network.build_model(self.cfg["policy"])

        self.optimizer = torch.optim.Adadelta(self.quality_network.model.parameters(), lr=1e-2)

        self.experience_replay = ExperienceReplay(self.cfg)
        self.experience_replay.init(self.env, self.agent)

    def reconfigure(self):
        self.cleanup()
        self.configure()

    def step(self):
        batch_size = 128
        self.quality_network.model.train()
        with torch.no_grad():
            for i in range(batch_size):
                episode_data = self.agent.run_training_episode(self.quality_network, self.env)
                self.experience_replay.register_episode(episode_data)

        policy_reward = 0
        for i in range(10):
            policy_reward += self.agent.run_benchmark_episode(self.quality_network, self.env)
        policy_reward/=10

        print(policy_reward)

    def update(self):
        batch_size = 256
        batch = self.experience_replay.get_batch(self.cfg["rng"], batch_size, as_columns = True)

        observations = torch.as_tensor(batch[ExperienceReplay.OBSERVATION_IDX])
        actions = torch.as_tensor(batch[ExperienceReplay.ACTION_IDX])
        rewards = torch.as_tensor(batch[ExperienceReplay.REWARD_IDX])
        dones = torch.as_tensor(batch[ExperienceReplay.DONE_IDX])
        
        num_actions = int(np.prod(self.env.action_space))

        y = []
        for i in range(len(observations)):
            highest_quality = None
            for action in range(num_actions):
                input_obs = np.concatenate((observations[i], (action,)))
                target_output = self.target_quality_network.model(input_obs)

                if highest_quality is None or target_output.item() > highest_quality:
                    highest_quality = target_output.item()

                if dones[i] == 0:
                    y_hat = rewards[i] + self.cfg["policy_optimizer"]["gamma"] * highest_quality
                else:
                    y_hat = rewards[i]

                y.append(y_hat)
        y = torch.as_tensor(y)

        q_network_input = torch.cat([observations, actions], dim=1)
        q_network_output = self.quality_network.model(q_network_input)

        self.optimizer.zero_grad()
        loss = f.smooth_l1_loss(q_network_output, y)
        loss.backward()
        self.optimizer.step()

    def is_done(self):
        return False

    def set_base_dir(self, base_dir):
        pass

    def cleanup(self):
        if self.agent is not None:
            self.agent.cleanup()
        if self.env is not None:
            self.env.cleanup()
        if self.quality_network is not None:
            self.quality_network.cleanup()
        if self.target_quality_network is not None:
            self.target_quality_network.cleanup()

        del self.agent
        del self.env
        del self.quality_network
        del self.target_quality_network
        del self.optimizer
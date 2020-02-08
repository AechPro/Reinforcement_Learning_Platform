import numpy as np
class EpisodeData(object):
    def __init__(self):
        self.actions = []
        self.observations = []
        self.next_observations = []
        self.timesteps = 0
        self.rewards = []
        self.dones = []
        self.future_rewards = []

    def register_data(self, environment_step_data):
        policy_input, action, next_policy_input, reward, done = environment_step_data
        if done:
            numerical_done = 1
        else:
            numerical_done = 0

        self.actions.append(action)
        self.observations.append(policy_input)
        self.next_observations.append(next_policy_input)
        self.rewards.append(reward)
        self.dones.append(numerical_done)

    def compute_future_rewards(self, gamma):
        self.future_rewards = []
        for i in range(len(self.rewards)):
            reward = 0
            for j in range(i, len(self.rewards)):
                reward += self.rewards[i]*np.power(gamma, j)
            self.future_rewards.append(reward)

    def cleanup(self):
        del self.actions
        del self.observations
        del self.next_observations
        del self.rewards
        del self.dones
        del self.future_rewards

        self.__init__()
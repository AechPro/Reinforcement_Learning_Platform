from Util import ExperienceReplay
import numpy as np
class EpisodeData(object):
    def __init__(self):
        self.timesteps = 0
        self.actions = []
        self.observations = []
        self.next_observations = []
        self.rewards = []
        self.dones = []
        self.future_rewards = []
        self.td_residuals = []
        self.advantages = []
        self.action_log_probs = []

    def serialize_to_timesteps(self):
        timesteps = []
        for i in range(len(self.observations)):
            timestep = [None for _ in range(ExperienceReplay.TIMESTEP_SIZE)]
            self._add_entry_to_timestep(timestep, ExperienceReplay.REWARD_IDX, self.rewards, i)
            self._add_entry_to_timestep(timestep, ExperienceReplay.OBSERVATION_IDX, self.observations, i)
            self._add_entry_to_timestep(timestep, ExperienceReplay.NEXT_OBSERVATION_IDX, self.next_observations, i)
            self._add_entry_to_timestep(timestep, ExperienceReplay.ACTION_IDX, self.actions, i)
            self._add_entry_to_timestep(timestep, ExperienceReplay.ACTION_LOG_PROB_IDX, self.action_log_probs, i)
            self._add_entry_to_timestep(timestep, ExperienceReplay.ADVANTAGE_IDX, self.advantages, i)
            self._add_entry_to_timestep(timestep, ExperienceReplay.TD_RESIDUAL_IDX, self.td_residuals, i)
            self._add_entry_to_timestep(timestep, ExperienceReplay.FUTURE_REWARD_IDX, self.future_rewards, i)
            self._add_entry_to_timestep(timestep, ExperienceReplay.DONE_IDX, self.dones, i)
            timesteps.append(tuple(timestep))
        return timesteps

    def register_environment_step(self, environment_step_data):
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
    
    def compute_td_residuals(self, value_estimations, gamma):
        residuals = []
        rewards = self.rewards

        for i in range(len(rewards)-1):
            res = rewards[i] + gamma*value_estimations[i+1] - value_estimations[i]
            residuals.append(res)
        residuals.append(rewards[-1])

        self.td_residuals = residuals

    def compute_general_advantage_estimation(self, gamma, lmbda):
        residuals = self.td_residuals
        advantages = []
        for i in range(len(residuals)):
            adv = 0
            for j in range(0, len(residuals)-i):
                coeff = np.power(lmbda*gamma, j)
                adv += coeff * residuals[i+j]
            advantages.append(adv)
        self.advantages = advantages

    def _add_entry_to_timestep(self, ts, idx, arr, i):
        if arr is not None and i < len(arr):
            ts[idx] = arr[i]
        else:
            ts[idx] = None

    def cleanup(self):
        del self.actions
        del self.observations
        del self.next_observations
        del self.rewards
        del self.dones
        del self.future_rewards
        del self.td_residuals
        del self.advantages
        del self.action_log_probs

        self.__init__()
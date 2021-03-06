from Agents import Agent
import numpy as np

class DeepQLearningAgent(Agent):
    def __init__(self, config):
        super().__init__(config)
        self.eps_end = 0.05
        self.eps_start = 0.9
        self.iteration = 0
        self.eps_decay = 50

    def prepare_policy_input(self, obs_buffer, buffer_shape):
        return obs_buffer

    def get_action(self, policy, state, episode_data, obs_stats=None):
        num_actions = int(np.prod(policy.output_shape))
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1. * self.iteration / self.eps_decay)

        if self.cfg["rng"].uniform(0, 1) < epsilon:
            best_action = self.cfg["rng"].randint(0, num_actions)
        else:
            best_action = 0
            best_score = -np.inf
            for action in range(2):
                if type(state) == list:
                    input_state = np.concatenate( (state[0], (action,)) )
                    input_state = input_state.reshape((1, len(state[0]) + 1))
                else:
                    input_state = np.concatenate((state[0,:], (action,)), axis=0)

                input_state = input_state.astype(np.float32)
                policy_output = policy.get_action(input_state)
                action_quality = policy_output.max().item()

                if action_quality > best_score:
                    best_action = action
                    best_score = action_quality

        return best_action

    def cleanup(self):
        self.iteration = 0
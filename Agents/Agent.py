import numpy as np
import torch

class Agent(object):
    def __init__(self, config):
        super().__init__()
        self.cfg = config

    def run_training_episode(self, policy, env, obs_stats = None):
        initial_obs = env.reset()
        done = False
        state = env.get_random_state()
        episode_results = []

        reward = 0
        timesteps = 1

        buffer_shape = [self.cfg["policy"]["observation_buffer_length"]]
        for entry in policy.input_shape:
            buffer_shape.append(entry)

        obs_buffer = [initial_obs.copy() for _ in range(buffer_shape[0])]

        while not env.needs_reset:
            policy_input = np.reshape(obs_buffer, buffer_shape)

            action = self.get_action(policy_input)
            obs, rew = env.step(action)

            reward += rew + self.cfg["rng"].choice(self.cfg["policy_optimizer"]["reward_jiggle"])

            #attach_obs_to_buffer(obs_buffer, obs, buffer_shape[0])

            obs_buffer.append(obs)
            if len(obs_buffer) >= buffer_shape[0]:
                old = obs_buffer.pop(0)
                del old

            timesteps += 1

        env.close()
        del obs_buffer
        return reward, timesteps

        while not done:


            # epsilon greedy
            epsilon = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1. * self.iteration / self.eps_decay)
            if np.random.uniform(0, 1) < epsilon:
                best_action = env.get_random_action()
            else:
                with torch.no_grad():
                    best_action = 0
                    best_score = -np.inf
                    for action in num_actions:
                        input_state = np.concatenate((state, (action,)))
                        input_batch = np.reshape(input_state, (1, len(input_state)))
                        policy_output = policy.get_action(input_batch)
                        action_quality = policy_output.max().item()

                        if action_quality > best_score:
                            best_action = action
                            best_score = action_quality

            next_state, reward, done, _ = env.step(best_action)
            episode_results.append((state, (best_action,), next_state, reward, done))
            state = next_state

        return episode_results

    def run_benchmark_episode(self, policy, env):
        env.reset()
        done = False
        state = env.get_random_state()
        total_reward = 0
        num_actions = range(np.prod(env.output_space))

        while not done:
            with torch.no_grad():
                best_action = 0
                best_score = -np.inf
                for action in num_actions:
                    input_state = np.concatenate((state, (action,)))
                    input_batch = np.reshape(input_state, (1, len(input_state)))
                    policy_output = policy.get_action(input_batch)
                    action_quality = policy_output.max().item()

                    if action_quality > best_score:
                        best_action = action
                        best_score = action_quality

            state, reward, done, _ = env.step(best_action)
            total_reward += reward
            env.env.render()
        #env.env.close()

        return total_reward

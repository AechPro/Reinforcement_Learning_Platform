import numpy as np
import time
from Agents import EpisodeData

class Agent(object):
    def __init__(self, config):
        super().__init__()
        self.cfg = config

    def run_training_episode(self, policy, env, obs_stats = None):
        initial_obs = env.reset()
        episode_data = EpisodeData()

        buffer_shape = [self.cfg["policy"]["observation_buffer_length"]]
        for entry in policy.input_shape:
            buffer_shape.append(entry)

        obs_buffer = [initial_obs.copy() for _ in range(buffer_shape[0])]
        policy_input = policy_input = np.reshape(obs_buffer, buffer_shape)

        while not env.needs_reset:
            action = self.get_action(policy_input, obs_stats=obs_stats)
            obs, reward = env.step(action)

            _attach_obs_to_buffer(obs, obs_buffer, buffer_shape[0])

            next_policy_input = np.reshape(obs_buffer, buffer_shape)
            episode_data.register((policy_input.copy(), (action,), next_policy_input.copy(), reward, env.needs_reset))
            policy_input = next_policy_input

            episode_data.timesteps+=1

        env.close()
        del obs_buffer
        return episode_data

    def run_benchmark_episode(self, policy, env, obs_stats = None, render = False, render_frame_delay = None):
        initial_obs = env.reset()

        benchmark_reward = 0

        buffer_shape = [self.cfg["policy"]["observation_buffer_length"]]
        for entry in policy.input_shape:
            buffer_shape.append(entry)

        obs_buffer = [initial_obs.copy() for _ in range(buffer_shape[0])]
        policy_input = policy_input = np.reshape(obs_buffer, buffer_shape)

        while not env.needs_reset:
            action = self.get_action(policy_input, obs_stats=obs_stats)
            obs, reward = env.step(action)

            _attach_obs_to_buffer(obs, obs_buffer, buffer_shape[0])
            policy_input = np.reshape(obs_buffer, buffer_shape)

            benchmark_reward += reward

            if render:
                env.render()
                if render_frame_delay is not None:
                    assert type(render_frame_delay) in (int, float), "ATTEMPTED TO RENDER A BENCHMARK EPISODE WITH INVALID" \
                                                                     "FRAME DELAY PERIOD {}".format(render_frame_delay)
                    time.sleep(render_frame_delay)

        env.close()
        del obs_buffer
        return benchmark_reward

    def get_action(self, state, obs_stats=None):
        raise NotImplementedError

def _attach_obs_to_buffer(obs, obs_buffer, max_buffer_len):
    obs_buffer.append(obs)
    if len(obs_buffer) >= max_buffer_len:
        old = obs_buffer.pop(0)
        del old

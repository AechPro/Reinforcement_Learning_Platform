import gym
import numpy as np

from Environments import Environment

class GymEnvironment(Environment):
    def __init__(self, env_id=None, env=None):
        super().__init__()
        if env_id is not None:
            self._env = gym.make(env_id)
        else:
            self._env = env

        self.observation_shape = np.shape(self._env.observation_space)
        try:
            self.action_shape = self._env.action_space.n
        except:
            self.action_shape = np.shape(self._env.action_space)

    def step(self, action):
        assert self._env is not None, "TRIED TO STEP WITH NO ENVIRONMENT!"

        obs, reward, done, debug = self._env.step(action)

        del self.current_observation
        self.current_observation = obs
        self.needs_reset = done

        return obs, reward

    def reset(self):
        assert self._env is not None, "TRIED TO RESET WITH NO ENVIRONMENT!"
        del self.current_observation
        self.current_observation = None
        self.needs_reset = False
        return self._env.reset()

    def get_random_obs(self):
        obs, _ = self.step(self.get_random_action())
        return obs

    def get_random_action(self):
        assert self._env is not None, "TRIED TO GET ACTION WITH NO ENVIRONMENT!"
        return self._env.action_space.sample()

    def get_obs(self):
        return self.current_observation

    def render(self):
        assert self._env is not None, "TRIED TO RENDER WITH NO ENVIRONMENT!"
        self._env.render()

    def close(self):
        assert self._env is not None, "TRIED TO CLOSE WITH NO ENVIRONMENT!"
        self._env.close()

    def seed(self, seed):
        assert self._env is not None, "TRIED TO SEED WITH NO ENVIRONMENT!"
        self._env.seed(seed)
        self._env.action_space.seed(seed)

    def cleanup(self):
        self.close()
        del self._env
        del self.current_observation
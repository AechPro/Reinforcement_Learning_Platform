import bsuite
import numpy as np
class ActionSpace(object):
    def __init__(self, rng, action_spec):
        self.rng = rng
        self.n = action_spec.num_values
        self.min = action_spec.minimum
        self.max = action_spec.maximum

    def sample(self):
        #TODO: ensure all BSuite environments are actually discrete. It looks like they are.
        return self.rng.randint(self.min, self.max+1)

    def seed(self, seed):
        del self.rng
        self.rng = np.random.RandomState(seed)

class BSuiteWrapper(object):
    def __init__(self, env_id):
        self.id = env_id
        self.env = bsuite.load_from_id(env_id)
        self.action_space = ActionSpace(np.random.RandomState(0), self.env.action_spec())

        shape = self.env.observation_spec().shape
        if shape[0] < 2:
            self.observation_space = np.zeros(shape=shape[1:])
        else:
            self.observation_space = np.zeros(shape=shape)

    def step(self, action):
        timestep = self.env.step(action)

        obs = timestep.observation.reshape(self.observation_space.shape)
        reward = timestep.reward
        done = timestep.last()
        debug = None
        return obs, reward, done, debug

    def seed(self, seed):
        self.env._rng.seed(seed)

    def reset(self):
        self.env.reset()

    def close(self):
        self.env.close()

    def render(self):
        return

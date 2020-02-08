import numpy as np

class Environment(object):
    def __init__(self):
        self.observation_shape = None
        self.action_shape = None
        self.needs_reset = True
        self.current_observation = None
    
    def get_random_obs(self):
        raise NotImplementedError

    def get_random_action(self):
        raise NotImplementedError
    
    def get_obs(self):
        raise NotImplementedError
    
    def step(self, action):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError
    
    def render(self):
        raise NotImplementedError
    
    def close(self):
        raise NotImplementedError

    def seed(self, seed):
        raise NotImplementedError

    def cleanup(self):
        raise NotImplementedError
    
    def test_seeding(self):
        self.seed(0)
        action = self.get_random_action()

        self.reset()
        first_obs_set = []
        while not self.needs_reset:
            obs, _ = self.step(action)
            first_obs_set.append(obs)

        self.seed(0)
        self.reset()

        second_obs_set = []
        while not self.needs_reset:
            obs, _ = self.step(action)
            second_obs_set.append(obs)


        for obs_1, obs_2 in zip(first_obs_set, second_obs_set):
            for arg1, arg2 in zip(obs_1, obs_2):
                if arg1 != arg2:
                    print("!!!ENVIRONMENT COULD NOT BE SEEDED!!!")
                    del first_obs_set
                    del second_obs_set
                    self.close()
                    return

        del first_obs_set
        del second_obs_set
        self.close()
        print("ENVIRONMENT PASSSED SEEDING TEST!")


    def get_policy_input_shape(self):
        return self.observation_shape

    def get_policy_output_shape(self):
        return self.action_shape

    def get_random_batch(self, batch_size):
        assert self.observation_shape is not None, "ENVIRONMENT OBSERVATION SHAPE IS NONE!"

        batch = []
        for i in range(batch_size):
            if self.needs_reset:
                self.reset()

            frame = self.get_random_obs()
            batch.append(frame)
        if type(self.observation_shape) not in (list, np.array, tuple):
            obs_shape = [self.observation_shape, ]
        else:
            obs_shape = self.observation_shape

        batch_shape = [batch_size]
        for entry in obs_shape:
            batch_shape.append(entry)

        reshaped_batch = np.reshape(batch, batch_shape)

        del batch
        del batch_shape
        del obs_shape

        return reshaped_batch
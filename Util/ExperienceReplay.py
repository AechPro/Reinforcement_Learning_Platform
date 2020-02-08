DONE_IDX = 0
ACTION_IDX = 1
REWARD_IDX = 2
OBSERVATION_IDX = 3
FUTURE_REWARD_IDX = 4
NEXT_OBSERVATION_IDX = 5

class ExperienceReplay(object):
    def __init__(self, size):
        self.size = size
        self.memory = []

    def register_data(self, episode_data):
        episode_data.compute_future_rewards()
        for i in range(len(episode_data.rewards)):
            timestep = (
                episode_data.dones[i],
                episode_data.actions[i],
                episode_data.rewards[i],
                episode_data.observations[i],
                episode_data.future_rewards[i],
                episode_data.next_observations[i]
            )
            self.add_timestep(timestep)

    def get_random(self, rng):
        idx = rng.randint(0, len(self.memory))
        return self.memory[idx]

    def get_random_batch(self, rng, batch_size):
        batch = [self.get_random(rng) for _ in range(batch_size)]
        return batch

    def add_timestep(self, timestep):
        self.memory.append(timestep)

        if len(self.memory) > self.size:
            _ = self.memory.pop(0)
            del _
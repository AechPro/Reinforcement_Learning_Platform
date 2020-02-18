class ExperienceReplay(object):
    DONE_IDX = 0
    ACTION_IDX = 1
    REWARD_IDX = 2
    OBSERVATION_IDX = 3
    FUTURE_REWARD_IDX = 4
    NEXT_OBSERVATION_IDX = 5
    def __init__(self, config):
        self.cfg = config
        self.size = config["experience_replay"]["size"]
        self.memory = []

    def init(self, env, agent):
        from Policies import RandomPolicy
        policy = RandomPolicy(env.observation_shape, env.action_shape, None, self.cfg)

        while len(self.memory) < self.cfg["experience_replay"]["initial_frames"]:
            episode_data = agent.run_training_episode(policy, env)
            self.register_episode(episode_data)


    def register_episode(self, episode_data):
        episode_data.compute_future_rewards(self.cfg["policy_optimizer"]["gamma"])
        for i in range(len(episode_data.rewards)):
            timestep = (
                episode_data.dones[i],
                episode_data.actions[i],
                episode_data.rewards[i],
                episode_data.observations[i],
                episode_data.future_rewards[i],
                episode_data.next_observations[i]
            )
            self.register_timestep(timestep)

    def register_timestep(self, timestep):
        self.memory.append(timestep)

        if len(self.memory) > self.size:
            _ = self.memory.pop(0)
            del _

    def get(self, idx):
        if len(self.memory) > idx >= 0:
            return self.memory[idx]
        return None

    def get_batch(self, batch_size, as_columns=False):
        batch = [self.get(i) for i in range(batch_size)]

        if as_columns:
            dones = []
            actions = []
            rewards = []
            observations = []
            next_observations = []
            future_rewards = []

            for entry in batch:
                dones.append(entry[ExperienceReplay.DONE_IDX])
                actions.append(entry[ExperienceReplay.ACTION_IDX])
                rewards.append(entry[ExperienceReplay.REWARD_IDX])
                observations.append(entry[ExperienceReplay.OBSERVATION_IDX])
                next_observations.append(entry[ExperienceReplay.NEXT_OBSERVATION_IDX])
                future_rewards.append(entry[ExperienceReplay.FUTURE_REWARD_IDX])

            batch = (dones, actions, rewards, observations, future_rewards, next_observations)
            return batch

    def get_random(self):
        idx = self.cfg["rng"].randint(0, len(self.memory)-1)
        return self.get(idx)

    def get_random_batch(self, batch_size, as_columns = False):
        batch = [self.get_random() for _ in range(batch_size)]

        if as_columns:
            dones = []
            actions = []
            rewards = []
            observations = []
            next_observations = []
            future_rewards = []

            for entry in batch:
                dones.append(entry[ExperienceReplay.DONE_IDX])
                actions.append(entry[ExperienceReplay.ACTION_IDX])
                rewards.append(entry[ExperienceReplay.REWARD_IDX])
                observations.append(entry[ExperienceReplay.OBSERVATION_IDX])
                next_observations.append(entry[ExperienceReplay.NEXT_OBSERVATION_IDX])
                future_rewards.append(entry[ExperienceReplay.FUTURE_REWARD_IDX])

            batch = (dones, actions, rewards, observations, future_rewards, next_observations)

        return batch

    def clear(self):
        del self.memory
        self.memory = []

    def cleanup(self):
        self.clear()
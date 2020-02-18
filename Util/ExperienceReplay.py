class ExperienceReplay(object):
    DONE_IDX = 0
    ACTION_IDX = 1
    REWARD_IDX = 2
    OBSERVATION_IDX = 3
    FUTURE_REWARD_IDX = 4
    NEXT_OBSERVATION_IDX = 5
    TD_RESIDUAL_IDX = 6
    ADVANTAGE_IDX = 7
    ACTION_LOG_PROB_IDX = 8

    TIMESTEP_SIZE = 9

    def __init__(self, config):
        self.cfg = config
        self.size = config["experience_replay"]["size"]
        self.memory = []

    def fill_initial(self, env, agent):
        from Policies import RandomPolicy
        policy = RandomPolicy(env.observation_shape, env.action_shape, None, self.cfg)

        while len(self.memory) < self.cfg["experience_replay"]["initial_frames"]:
            episode_data = agent.run_training_episode(policy, env)
            self.register_episode(episode_data)

    def register_episode(self, episode_data, compute_future_returns=True):
        if compute_future_returns:
            episode_data.compute_future_rewards(self.cfg["policy_optimizer"]["gamma"])

        timesteps = episode_data.serialize_to_timesteps()
        for timestep in timesteps:
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
            batch = self._batch_to_columns(batch)

        return batch

    def get_random(self):
        idx = self.cfg["rng"].randint(0, len(self.memory)-1)
        return self.get(idx)

    def get_random_batch(self, batch_size, as_columns = False):
        batch = [self.get_random() for _ in range(batch_size)]

        if as_columns:
            batch = self._batch_to_columns(batch)

        return batch
    
    def _batch_to_columns(self, batch):
        actions = []
        observations = []
        next_observations = []
        rewards = []
        dones = []
        future_rewards = []
        td_residuals = []
        advantages = []
        action_log_probs = []
        
        for timestep in batch:
            dones.append(timestep[ExperienceReplay.DONE_IDX])
            actions.append(timestep[ExperienceReplay.ACTION_IDX])
            rewards.append(timestep[ExperienceReplay.REWARD_IDX])
            observations.append(timestep[ExperienceReplay.OBSERVATION_IDX])
            next_observations.append(timestep[ExperienceReplay.NEXT_OBSERVATION_IDX])
            future_rewards.append(timestep[ExperienceReplay.FUTURE_REWARD_IDX])
            td_residuals.append(timestep[ExperienceReplay.TD_RESIDUAL_IDX])
            advantages.append(timestep[ExperienceReplay.ADVANTAGE_IDX])
            action_log_probs.append(timestep[ExperienceReplay.ACTION_LOG_PROB_IDX])

        batch = [None for _ in range(ExperienceReplay.TIMESTEP_SIZE)]
        batch[ExperienceReplay.DONE_IDX] = dones
        batch[ExperienceReplay.ACTION_IDX] = actions
        batch[ExperienceReplay.REWARD_IDX] = rewards
        batch[ExperienceReplay.OBSERVATION_IDX] = observations
        batch[ExperienceReplay.NEXT_OBSERVATION_IDX] = next_observations
        batch[ExperienceReplay.FUTURE_REWARD_IDX] = future_rewards
        batch[ExperienceReplay.TD_RESIDUAL_IDX] = td_residuals
        batch[ExperienceReplay.ADVANTAGE_IDX] = advantages
        batch[ExperienceReplay.ACTION_LOG_PROB_IDX] = action_log_probs

        return batch
            
            
    def clear(self):
        del self.memory
        self.memory = []

    def cleanup(self):
        self.clear()
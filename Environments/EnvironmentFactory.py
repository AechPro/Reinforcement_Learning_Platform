from Environments import GymEnvironment

def get_from_config(cfg):
    env_id = cfg["env_id"]
    if "bsuite" in env_id:
        from Environments import BSuiteWrapper
        env_name = env_id.split(" ")[-1]
        env = BSuiteWrapper(env_name)
        return GymEnvironment(env_id=None, env=env)

    return GymEnvironment(cfg["env_id"])

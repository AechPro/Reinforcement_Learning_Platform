def get_from_config(cfg):
    type = cfg["policy_optimizer"]["type"]
    t = type.lower().strip()

    if t in ("pg", "policy gradients", "policy gradient", "policy_gradient", "policy_gradients",
                   "basic policy gradients", "basic_policy_gradients", "basic_policy_gradient"):
        from Algorithms.BasicPolicyGradients import Optimizer
        return Optimizer(cfg)

    if t in ("dqn", "deep_q_learning", "deep q learning"):
        from Algorithms.DeepQLearning import Optimizer
        return Optimizer(cfg)

    if t in ("vpg", "vanilla_policy_gradients", "vanilla policy gradients",
             "vanilla_policy_gradient", "vanilla policy gradient"):
        from Algorithms.VanillaPolicyGradients import Optimizer
        return Optimizer(cfg)
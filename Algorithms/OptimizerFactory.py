def get_from_config(cfg):
    type = cfg["policy_optimizer"]["type"]
    t = type.lower().strip()

    if t in ("pg", "policy gradients", "policy gradient", "policy_gradient", "policy_gradients",
                   "basic policy gradients", "basic_policy_gradients", "basic_policy_gradient"):
        from Algorithms.BasicPolicyGradients import Optimizer
        return Optimizer(cfg)

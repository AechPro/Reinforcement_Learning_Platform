def get_from_config(cfg):
    implementation_framework, agent_type = cfg["agent"]["type"].split(" ")
    i_f = implementation_framework.lower().strip()
    a_t = agent_type.lower().strip()

    if i_f in ("torch", "pytorch", "py_torch"):
        if a_t in ("pg", "policy gradients", "policy gradient", "policy_gradient", "policy_gradients",
                   "basic policy gradients", "basic_policy_gradients", "basic_policy_gradient"):
            from Agents.PyTorchAgents import BasicPolicyGradientsAgent
            return BasicPolicyGradientsAgent(cfg)
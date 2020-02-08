import numpy as np

from Policies.PyTorch import FeedForward
from Config import ConfigLoader
from Environments import EnvironmentFactory


def run_test():
    # cfg = {"rng": np.random}
    # input_shape = 8
    # output_shape = 8
    # instructions = \
    #     {
    #         "init_std": 0.05,
    #         "layers": [64, 64],
    #         "layer_functions": ['relu', 'relu'],
    #         "layer_extras": ['bn', 'bn'],
    #         "output_function": 'linear',
    #         "output_extras": 'bn',
    #     }

    cfg = ConfigLoader.load_config(file_name="test_config.json")
    env = EnvironmentFactory.get_from_config(cfg)
    input_shape = env.get_policy_input_shape()
    output_shape = env.get_policy_output_shape()
    instructions = cfg["policy"]
    cfg["rng"] = np.random.RandomState(cfg["seed"])

    policy = FeedForward(input_shape, output_shape, None, cfg)
    policy.build_model(instructions)

    print("BUILT POLICY LAYERS:")
    for layer in policy.model:
        print(layer)

    num = np.prod(input_shape)
    vbn = [np.random.randn(num) for _ in range(1000)]
    inp = np.ones(num)
    out = policy.activate(inp)

    print("\nOUTPUT ON ONES BEFORE VBN:", out)
    policy.compute_virtual_normalization(vbn)
    out = policy.activate(inp)
    print("OUTPUT ON ONES AFTER VBN:", out)

    policy.save("data/test")
    out = policy.activate(inp)
    print("OUTPUT ON ONES AFTER SAVE:", out)
    del policy

    policy = FeedForward(input_shape, output_shape, None, cfg)
    policy.build_model(instructions)
    policy.load("data/test")
    out = policy.activate(inp)
    print("OUTPUT ON ONES AFTER LOAD:", out)
    policy.compute_virtual_normalization(vbn)
    out = policy.activate(inp)
    print("OUTPUT ON ONES AFTER LOAD AND VBN:", out)


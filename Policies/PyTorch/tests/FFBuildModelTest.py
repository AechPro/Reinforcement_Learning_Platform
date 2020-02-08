import numpy as np

from Policies.PyTorch import FeedForward


def run_test():
    cfg = {"rng": np.random}
    input_shape = 8
    output_shape = 8
    instructions = \
    {
        "init_std": 0.05,
        "layers" : [64,64],
        "layer_functions" : ['relu', 'relu'],
        "layer_extras" : ['bn', 'bn'],
        "output_function" : 'linear',
        "output_extras" : 'bn',
    }
    policy = FeedForward(input_shape, output_shape, None, cfg)
    policy.build_model(instructions)

    print("BUILT POLICY LAYERS:")
    for layer in policy.model:
        print(layer)

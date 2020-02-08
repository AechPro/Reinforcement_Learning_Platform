import numpy as np

from Policies.PyTorch import FeedForward


def run_test():
    cfg = {"rng": np.random}
    input_shape = 2
    output_shape = 2
    instructions = \
    {
        "init_std": 0.05,
        "layers" : [1],
        "layer_functions" : ['relu'],
        "layer_extras" : ['bn'],
        "output_function" : 'linear',
        "output_extras" : 'bn',
    }
    policy = FeedForward(input_shape, output_shape, None, cfg)
    policy.build_model(instructions)

    print("BUILT POLICY LAYERS:")
    for layer in policy.model:
        print(layer)

    flat = np.random.randn(policy.num_params)

    print("POLICY FLAT BEFORE SETTING:",policy.get_trainable_flat())
    policy.set_trainable_flat(flat)
    print("POLICY FLAT AFTER SETTING:", policy.get_trainable_flat())

    print("FLAT SHOULD NOW BE:",flat)
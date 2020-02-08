import numpy as np

from Policies.PyTorch import FeedForward


def test_save_load():
    cfg = {"rng": np.random}
    input_shape = 8
    output_shape = 8
    instructions = \
        {
            "init_std": 0.05,
            "layers": [64, 64],
            "layer_functions": ['relu', 'relu'],
            "layer_extras": ['bn', 'bn'],
            "output_function": 'linear',
            "output_extras": 'bn',
        }
    policy = FeedForward(input_shape, output_shape, None, cfg)
    policy.build_model(instructions)

    print("BUILT POLICY LAYERS:")
    for layer in policy.model:
        print(layer)

    inp = np.ones(input_shape)
    out = policy.activate(inp)

    print("\nOUTPUT ON ONES BEFORE SAVING:", out)
    policy.save("data/experiments/exp_name/epochs/epoch_0/policy")
    policy.load("data/experiments/exp_name/epochs/epoch_0/policy")

    out = policy.activate(inp)
    print("OUTPUT ON ONES AFTER SAVING:", out)

def test_save_load_vbn():
    cfg = {"rng": np.random}
    input_shape = 8
    output_shape = 8
    instructions = \
        {
            "init_std": 0.05,
            "layers": [64, 64],
            "layer_functions": ['relu', 'relu'],
            "layer_extras": ['bn', 'bn'],
            "output_function": 'linear',
            "output_extras": 'bn',
        }
    policy = FeedForward(input_shape, output_shape, None, cfg)
    policy.build_model(instructions)

    print("BUILT POLICY LAYERS:")
    for layer in policy.model:
        print(layer)

    vbn = [np.random.randn(input_shape) for _ in range(1000)]
    policy.compute_virtual_normalization(vbn)

    inp = np.ones(input_shape)
    out = policy.activate(inp)

    print("\nOUTPUT ON ONES WITH VBN BEFORE SAVING:", out)
    policy.save("data/experiments/exp_name/epochs/epoch_0/policy")

    out = policy.activate(inp)
    print("\nOUTPUT ON ONES WITH VBN AFTER SAVING:",out)

    policy.set_trainable_flat(policy.get_trainable_flat() + np.random.randn(policy.num_params))
    out = policy.activate(inp)
    print("\nJIGGLED OUTPUT ON ONES WITH VBN BEFORE LOADING", out)

    policy.load("data/experiments/exp_name/epochs/epoch_0/policy")
    out = policy.activate(inp)
    print("OUTPUT ON ONES WITH VBN AFTER LOADING:", out)

def run_test():
    test_save_load()
    test_save_load_vbn()
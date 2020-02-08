import numpy as np

from Policies import PolicyActionParsers as parsers
from Policies.PyTorch import FeedForward
from Config import ConfigLoader
from Utils import MiscParsingFunctions


def test_bn():
    cfg = ConfigLoader.load_config(file_name="test_config.json")
    input_shape = 8
    output_shape = 8
    instructions = cfg["policy"]
    #parser = MiscParsingFunctions.parse_policy_action_function(cfg["policy"]["action_parser"])
    policy = FeedForward(input_shape, output_shape, None, cfg)
    policy.build_model(instructions)

    print("BUILT POLICY LAYERS:")
    for layer in policy.model:
        print(layer)

    inp = np.ones(input_shape)
    output = policy.activate(inp)

    print("POLICY OUTPUT ON ONES OF SHAPE", inp.shape, "=", output.shape, "EXPECTED", output_shape)
    action_parsers = {"linear" : parsers.linear_parse,
                      "random sample": parsers.random_sample,
                      "arg max": parsers.argmax_sample}

    for name, parser in action_parsers.items():
        policy.action_parser = parser
        print("ATTEMPTING PARSER",name)
        print("RAW POLICY OUTPUT:",policy.activate(inp)[0])
        print("SUM:",sum(policy.activate(inp)[0]))
        action = policy.get_action(inp)
        print("POLICY ACTION FROM PARSER",name,"=",action)

    print()

def test_batch_bn():
    cfg = ConfigLoader.load_config(file_name="test_config.json")
    input_shape = 8
    output_shape = 8
    instructions = cfg["policy"]
    parser = MiscParsingFunctions.parse_policy_action_function(cfg["policy"]["action_parser"])
    policy = FeedForward(input_shape, output_shape, parser, cfg)
    policy.build_model(instructions)

    print("BUILT POLICY LAYERS:")
    for layer in policy.model:
        print(layer)

    batch = [np.ones(input_shape) for _ in range(500)]
    output = policy.activate_batch(batch)

    print("OUTPUT ON ONES BATCH OF 500:", output.shape, "EXPECTED 500 OF", output_shape)

    action_parsers = {"linear": parsers.linear_parse,
                      "random sample": parsers.random_sample,
                      "arg max": parsers.argmax_sample}

    for name, parser in action_parsers.items():
        policy.action_parser = parser
        action = policy.get_actions_on_batch(batch)
        print("POLICY ACTIONS FROM PARSER", name, "=", action)
    print()

def test():
    cfg = ConfigLoader.load_config(file_name="test_config.json")
    input_shape = 8
    output_shape = 8
    instructions = cfg["policy"]
    parser = MiscParsingFunctions.parse_policy_action_function(cfg["policy"]["action_parser"])
    policy = FeedForward(input_shape, output_shape, parser, cfg)
    policy.build_model(instructions)

    print("BUILT POLICY LAYERS:")
    for layer in policy.model:
        print(layer)

    inp = np.ones(input_shape)
    output = policy.activate(inp)

    print("POLICY OUTPUT ON ONES OF SHAPE",inp.shape,"=",output.shape,"EXPECTED",output_shape)

    action_parsers = {"linear": parsers.linear_parse,
                      "random sample": parsers.random_sample,
                      "arg max": parsers.argmax_sample}

    for name, parser in action_parsers.items():
        policy.action_parser = parser
        action = policy.get_action(inp)
        print("POLICY ACTION FROM PARSER", name, "=", action)

    print()

def test_batch():
    cfg = ConfigLoader.load_config(file_name="test_config.json")
    input_shape = 8
    output_shape = 8
    instructions = cfg["policy"]
    parser = MiscParsingFunctions.parse_policy_action_function(cfg["policy"]["action_parser"])
    policy = FeedForward(input_shape, output_shape, parser, cfg)
    policy.build_model(instructions)

    print("BUILT POLICY LAYERS:")
    for layer in policy.model:
        print(layer)

    batch = [np.ones(input_shape) for _ in range(500)]
    output = policy.activate_batch(batch)

    print("OUTPUT ON ONES BATCH OF 500:",output.shape,"EXPECTED 500 OF",output_shape)

    action_parsers = {"linear": parsers.linear_parse,
                      "random sample": parsers.random_sample,
                      "arg max": parsers.argmax_sample}

    for name, parser in action_parsers.items():
        policy.action_parser = parser
        action = policy.get_actions_on_batch(batch)
        print("POLICY ACTIONS FROM PARSER", name, "=", action)

    print()


def run_test():
    test_bn()
    test_batch_bn()
    test()
    test_batch()
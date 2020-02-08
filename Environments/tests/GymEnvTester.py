import numpy as np

from Environments import GymEnvironment
from Policies import PolicyActionParsers
from Policies.PyTorch import FeedForward


def test_build():
    env = GymEnvironment("CartPole-v1")
    print("BUILT CartPole-v1","OBSERAVTION SHAPE:",env.observation_shape,"ACTION SHAPE:",env.action_shape)

def test_random_action():
    env = GymEnvironment("CartPole-v1")
    env.reset()
    action = env.get_random_action()
    print("GOT RANDOM ACTION",action)
    env.step(action)
    print("SUCCESSFULLY STEPPED ENV WITH ACTION",action)

def test_get_batch():
    env = GymEnvironment("CartPole-v1")
    env.reset()
    batch = env.get_random_batch(500)
    print("GOT BATCH OF SHAPE",batch.shape,"EXPECTED SHAPE WITH 500 ENTRIES OF",env.observation_shape)

def test_render():
    env = GymEnvironment("CartPole-v1")
    env.reset()
    while not env.needs_reset:
        env.step(env.get_random_action())
        env.render()
    env.close()
    print("RENDER TEST SUCCESSFUL!")

def test_policy():
    cfg = {"rng": np.random}

    env = GymEnvironment("CartPole-v1")
    input_shape = env.get_policy_input_shape()
    output_shape = env.get_policy_output_shape()

    policy_instructions = \
        {
            "layers": [64, 64],
            "layer_functions": ['relu', 'relu'],
            "layer_extras": ['bn', 'bn'],
            "output_function": 'softmax',
            "output_extras": 'bn',
        }
    policy = FeedForward(input_shape, output_shape, PolicyActionParsers.random_sample, cfg)
    policy.build_model(policy_instructions)

    env.reset()
    obs = env.get_random_obs()
    while not env.needs_reset:
        action = policy.get_action(obs)
        obs, rew = env.step(action)
        print("TOOK ACTION", action,"GOT REWARD",rew)

    print("POLICY TEST COMPLETE")

def test_seed():
    env = GymEnvironment("CartPole-v1")
    env.test_seeding()

    print("SEEDING TEST COMPLETE")

def run_test():
    test_build()
    test_random_action()
    test_get_batch()
    test_render()
    test_policy()
    test_seed()
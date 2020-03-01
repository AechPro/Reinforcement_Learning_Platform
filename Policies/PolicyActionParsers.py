import numpy as np
def linear_parse(policy_output, rng=None):
    return policy_output

def random_sample(policy_output, rng=None):
    actions = [i for i in range(len(policy_output))]
    if sum(policy_output) != 1.0:
        policy_output = abs(policy_output)
        policy_output /= sum(policy_output)

    return rng.choice(a=actions, p=policy_output)

def multinomial_distribution(policy_output, rng=None):
    std = np.exp(-0.5)*np.ones_like(policy_output)
    return rng.multivariate_normal(mean=policy_output, cov=std)

def argmax_sample(policy_output, rng=None):
    return np.argmax(policy_output)
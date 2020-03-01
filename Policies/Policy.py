import os

import numpy as np


class Policy(object):
    def __init__(self, input_shape, output_shape, action_parser, cfg, cfg_key):
        self.model = None
        self.flat = None
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_params = 0
        self.action_parser = action_parser
        self.cfg = cfg
        self.cfg_key = cfg_key
        self.bn_stats = None

    def build_model(self, model_instructions):
        raise NotImplementedError

    def activate(self, target_input, input_normalization=None):
        raise NotImplementedError

    def activate_batch(self, input_batch, input_normalization=None):
        raise NotImplementedError

    def compute_virtual_normalization(self, virtual_batch):
        raise NotImplementedError

    def update_internal_flat(self):
        raise NotImplementedError

    def get_trainable_flat(self):
        raise NotImplementedError

    def set_trainable_flat(self, flat):
        raise NotImplementedError

    def cleanup(self):
        raise NotImplementedError

    def update_internal_bn_stats(self):
        raise NotImplementedError

    def get_bn_stats(self):
        raise NotImplementedError

    def set_bn_stats(self, stats_list):
        raise NotImplementedError

    def get_output_log_probs(self, output, action):
        raise NotImplementedError

    def copy_from(self, other):
        raise NotImplementedError

    def save(self, file_path):
        raise NotImplementedError

    def load(self, file_path):
        raise NotImplementedError

    def get_action(self, observation, input_normalization=None):
        policy_output = self.activate(observation, input_normalization=input_normalization)[0]

        if self.cfg[self.cfg_key]["action_noise_std"] != 0:
            action_perturbation = self.cfg["rng"].randn(len(policy_output))*self.cfg["policy"]["action_noise_std"]
            policy_output = np.add(policy_output,action_perturbation)

        return self.action_parser(policy_output, self.cfg["rng"])

    def get_actions_on_batch(self, observation_batch, input_normalization=None):
        actions = []
        for obs in observation_batch:
            action = self.get_action(obs, input_normalization=input_normalization)
            actions.append(action)

        np_actions = np.asarray(actions)
        del actions
        return np_actions

    def copy(self, other):
        self.set_trainable_flat(other.get_trainable_flat().copy())

    def reshape_input(self, target_input):
        assert self.input_shape is not None, "!!!!ATTEMPTED TO RESHAPE POLICY INPUT WHEN POLICY HAS NO INPUT SHAPE!!!!"

        input_shape = self.input_shape
        if type(self.input_shape) not in (list, np.array, tuple):
            input_shape = [self.input_shape,]

        size = np.prod(input_shape)
        target_size = np.prod(np.shape(target_input))
        batch_size = int(target_size//size)
        shape = [batch_size]
        for entry in input_shape:
            shape.append(entry)

        return np.reshape(target_input, shape)
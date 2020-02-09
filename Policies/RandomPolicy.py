from abc import ABC

from Policies import Policy
import numpy as np

class RandomPolicy(Policy, ABC):
    def __init__(self, input_shape, output_shape, action_parser, cfg):
        super().__init__(input_shape, output_shape, action_parser, cfg)
        self.discrete = len(np.shape(output_shape)) > 1

    def build_model(self, model_instructions):
        pass

    def activate(self, target_input, input_normalization=None):
        if self.discrete:
            output = self.cfg["rng"].randint(0, int(np.prod(self.output_shape)))
        else:
            output = self.cfg["rng"].randn(0, self.output_shape)
        return output

    def activate_batch(self, input_batch, input_normalization=None):
        return [self.activate(None) for _ in range(len(input_batch))]

    def compute_virtual_normalization(self, virtual_batch):
        pass

    def update_internal_flat(self):
        pass

    def get_trainable_flat(self):
        return []

    def set_trainable_flat(self, flat):
        pass

    def cleanup(self):
        pass

    def update_internal_bn_stats(self):
        pass

    def get_bn_stats(self):
        return []

    def set_bn_stats(self, stats_list):
        pass

    def copy_from(self, other):
        pass

    def save(self, file_path):
        pass

    def load(self, file_path):
        pass
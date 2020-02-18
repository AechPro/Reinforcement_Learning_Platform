from abc import ABC

from Policies import Policy, PolicyActionParsers
import torch

class RandomPolicy(Policy, ABC):
    def __init__(self, input_shape, output_shape, action_parser, cfg, cfg_key="policy"):
        super().__init__(input_shape, output_shape, action_parser, cfg, cfg_key)
        self.action_parser = PolicyActionParsers.linear_parse

    def build_model(self, model_instructions):
        pass

    def activate(self, target_input, input_normalization=None):
        if len(target_input.shape) == 0:
            num = 1
        else:
            num = target_input.shape[0]

        shape = [num]
        for arg in self.input_shape:
            shape.append(arg)

        return torch.as_tensor(self.cfg["rng"].uniform(-1, 1, shape))

    def activate_batch(self, input_batch, input_normalization=None):
        return self.activate(input_batch)

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